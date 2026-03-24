import simpy
import random
import statistics
from collections import Counter
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
import csv

RANDOM_SEED = 11

SIM_TIME = 10000          # 8 days
WARMUP_SECONDS = 100     # 1 day
MEASURE_UNTIL = SIM_TIME


def production_wait_time(now: float) -> float:
    """
    7-day periodic production stop:
    - Friday 17:00  -> Saturday 07:00
    - Saturday 17:00 -> Sunday 07:00
    Assumes 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday.
    """
    SEC_PER_DAY = 86400
    day = int((now // SEC_PER_DAY) % 7)   # 0=Mon ... 6=Sun
    t = now % SEC_PER_DAY                # seconds since midnight

    def secs(h, m=0):
        return h * 3600 + m * 60

    fri = 4
    sat = 5
    sun = 6

    if day == fri:
        stop_start = secs(17)
        stop_end = SEC_PER_DAY
        if stop_start <= t < stop_end:
            return stop_end - t
        return 0.0

    if day == sat:
        if t < secs(7):
            return secs(7) - t
        if secs(17) <= t < SEC_PER_DAY:
            return SEC_PER_DAY - t + secs(7)
        return 0.0

    if day == sun:
        if t < secs(7):
            return secs(7) - t
        return 0.0

    return 0.0


def _has_free_capacity(buf):
    return (getattr(buf, "free_capacity", None) and buf.free_capacity() > 0) \
           or len(buf.items) < buf.capacity


def splitter(env, input_store, out1, out2):
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)
        if _has_free_capacity(first):
            yield first.put(part)
            toggle ^= 1
        else:
            yield second.put(part)


def forwarder(env, src, dst):
    while True:
        part = yield src.get()
        yield dst.put(part)


def merger(env, a, b, out):
    env.process(forwarder(env, a, out))
    env.process(forwarder(env, b, out))


def reset_machine_stats(m):
    m.working_time = 0
    m.failed_time_total = 0
    m.wait_input_time = 0
    m.blocked_time = 0
    m.processed_count = 0
    m.window_wait_time = 0


class DelayBuffer:
    """Single store with a global capacity cap that includes in-transit + ready."""
    def __init__(self, env, cap, delay):
        self.env = env
        self.delay = delay
        self.cap = cap
        self.store = simpy.Store(env, capacity=cap)
        self.tokens = simpy.Container(env, init=cap, capacity=cap)
        self._in_transit = 0

    def put(self, part):
        return self.env.process(self._delayed_put(part))

    def get(self):
        return self.env.process(self._get_and_release())

    @property
    def items(self):
        return self.store.items

    @property
    def capacity(self):
        return self.store.capacity

    def in_transit_count(self):
        return self._in_transit

    def free_capacity(self):
        return int(self.tokens.level)

    def _delayed_put(self, part):
        yield self.tokens.get(1)
        self._in_transit += 1
        try:
            yield self.env.timeout(self.delay)
            yield self.store.put(part)
        finally:
            self._in_transit -= 1

    def _get_and_release(self):
        part = yield self.store.get()
        yield self.tokens.put(1)
        return part


class Machine:
    def __init__(self, env, name, input_buffer, output_buffer, process_time,
                 availability, mttr, working_power, waiting_power,
                 defect_rate=None, defect_sink=None, capacity=1):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.process_time = process_time
        self.availability = availability
        self.mttr = mttr
        self.defect_rate = defect_rate
        self.defect_sink = defect_sink
        self.working_power = working_power
        self.waiting_power = waiting_power
        self.resource = simpy.Resource(env, capacity=capacity)
        self.is_up = True

        self.working_time = 0
        self.failed_time_total = 0
        self.wait_input_time = 0
        self.blocked_time = 0
        self.active_count = 0
        self.processed_count = 0
        self.window_wait_time = 0

        if availability < 100:
            avail_frac = availability / 100.0
            self.mtbf = mttr * (avail_frac / (1 - avail_frac))
            env.process(self._breakdown_cycle())
        else:
            self.mtbf = float('inf')
        for _ in range(capacity):
            env.process(self.run())

    def _breakdown_cycle(self):
        while True:
            t_up = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(t_up)
            self.is_up = False
            t_repair = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(t_repair)
            self.failed_time_total += t_repair
            self.is_up = True

    def run(self):
        while True:
            with self.resource.request() as req:
                yield req
                part = None
                while part is None:
                    if self.is_up and len(self.input_buffer.items):
                        part = yield self.input_buffer.get()
                    else:
                        if self.is_up:
                            self.wait_input_time += 1
                        yield self.env.timeout(1)

                self.processed_count += 1
                self.active_count += 1

                w = production_wait_time(self.env.now)
                self.window_wait_time += w
                if w:
                    yield self.env.timeout(w)

                pt = self.process_time() if callable(self.process_time) else self.process_time
                remaining = pt
                while remaining > 0:
                    if not self.is_up:
                        yield self.env.timeout(1)
                    else:
                        yield self.env.timeout(1)
                        self.working_time += 1
                        remaining -= 1

            start_block = self.env.now

            if self.defect_rate is not None and self.defect_sink is not None:
                if random.random() < self.defect_rate:
                    part["defect"] = 1
                    yield self.defect_sink.put(part)
                else:
                    part["defect"] = 0
                    yield self.output_buffer.put(part)
            else:
                yield self.output_buffer.put(part)

            self.blocked_time += (self.env.now - start_block)
            self.active_count -= 1

    def waiting_energy_consumption(self):
        return self.waiting_power * (self.wait_input_time +
                                     self.failed_time_total +
                                     self.blocked_time +
                                     self.window_wait_time)

    def working_energy_consumption(self):
        return self.working_power * self.working_time


def part_generator(env, output_buffer):
    part_id = 0
    while True:
        part = {"id": part_id}
        yield output_buffer.put(part)
        part_id += 1
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL, buffer_caps=None):
    random.seed(seed)
    env = simpy.Environment()

    # Buffers (all helper buffers have defined capacity)
    raw_input = simpy.Store(env, capacity=1000)
    defect_sink = simpy.Store(env, capacity=100000)
    final_sink = simpy.Store(env, capacity=100000)

    # Determine capacities from buffer_caps mapping or use defaults
    def get_cap(name, default):
        if buffer_caps is None:
            return default
        return int(buffer_caps.get(name, default))

    # From Loading robot to Conveyor belt
    PostLoadingBuffer = DelayBuffer(env, cap=get_cap("PostLoadingBuffer", 2), delay=10)
    # From Conveyor belt to Washing machine
    PostConveyorBuffer = DelayBuffer(env, cap=get_cap("PostConveyorBuffer", 2), delay=10)
    # From Washing machine to Hantering cell
    PostWashingBuffer = DelayBuffer(env, cap=get_cap("PostWashingBuffer", 2), delay=10)

    # Parallel press buffers
    PrePress1Buffer = DelayBuffer(env, cap=get_cap("PrePress1Buffer", 3), delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=get_cap("PrePress2Buffer", 3), delay=32)
    # Helper buffer for splitter before individual pre-press buffers
    # capacity matches sum of PrePress1+PrePress2 caps
    PrePressJoinBuffer = simpy.Store(env, capacity=PrePress1Buffer.capacity + PrePress2Buffer.capacity)
    # After both Presses -> shared buffer
    PostPress1_2Buffer = DelayBuffer(env, cap=get_cap("PostPress1_2Buffer", 3), delay=32)

    # Helper stores for routing in parallel section (from each press to merger)
    PostPress1Out = simpy.Store(env, capacity=PrePress1Buffer.capacity)
    PostPress2Out = simpy.Store(env, capacity=PrePress2Buffer.capacity)

    # Machines
    LoadingRobot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=PostLoadingBuffer,
        process_time=12.0,
        availability=90.49,
        mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    ConveyorBelt = Machine(
        env, "Conveyor belt",
        input_buffer=PostLoadingBuffer,
        output_buffer=PostConveyorBuffer,
        process_time=6.0,
        availability=100.0,
        mttr=1.0,
        working_power=kwh_per_sec(0.0),
        waiting_power=kwh_per_sec(0.0),
    )

    WashingMachine = Machine(
        env, "Washing machine",
        input_buffer=PostConveyorBuffer,
        output_buffer=PostWashingBuffer,
        process_time=14.0,
        availability=80.89,
        mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    HanteringCell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=PrePressJoinBuffer,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # Split parts evenly into PrePress1Buffer and PrePress2Buffer
    env.process(splitter(env, PrePressJoinBuffer, PrePress1Buffer, PrePress2Buffer))

    PressCell1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=PostPress1Out,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    PressCell2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=PostPress2Out,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    # Merge from both presses into PostPress1_2Buffer
    merger(env, PostPress1Out, PostPress2Out, PostPress1_2Buffer)

    QualityStation = Machine(
        env, "Quality station cell",
        input_buffer=PostPress1_2Buffer,
        output_buffer=final_sink,
        process_time=41.0,
        availability=85.87,
        mttr=66.0,
        working_power=kwh_per_sec(0.84),
        waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089,
        defect_sink=defect_sink,
    )

    machines_list = [
        LoadingRobot,
        ConveyorBelt,
        WashingMachine,
        HanteringCell,
        PressCell1,
        PressCell2,
        QualityStation,
    ]

    # Part generation
    env.process(part_generator(env, raw_input))

    # Warmup
    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_before = len(final_sink.items)

    wip_samples = []
    delay_buffers = [
        PostLoadingBuffer,
        PostConveyorBuffer,
        PostWashingBuffer,
        PrePress1Buffer,
        PrePress2Buffer,
        PostPress1_2Buffer,
    ]

    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env.timeout(60)

    env.process(sample_wip(env))

    env.run(until=measure_until)

    total_produced = len(final_sink.items) - produced_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    result = {"overall": {
        "throughput": throughput,
        "wip": avg_wip,
        "produced_parts": total_produced},
        "machine_energy": {}}

    for m in machines_list:
        waiting_energy = m.waiting_energy_consumption()
        working_energy = m.working_energy_consumption()
        total_energy = waiting_energy + working_energy
        result["machine_energy"][m.name] = {
            "working_time": m.working_time,
            "waiting_time": m.failed_time_total + m.blocked_time,
            "working_energy": working_energy,
            "waiting_energy": waiting_energy,
            "total_energy": total_energy}

    return result


class SystemModelBuffers:
    """
    Defines the mapping and bounds for the tunable buffer capacities.
    Buffers:
        0: PostLoadingBuffer
        1: PostConveyorBuffer
        2: PostWashingBuffer
        3: PrePress1Buffer
        4: PrePress2Buffer
        5: PostPress1_2Buffer
    Capacities are integers in [1, 5].
    """

    BUFFER_NAMES = [
        "PostLoadingBuffer",
        "PostConveyorBuffer",
        "PostWashingBuffer",
        "PrePress1Buffer",
        "PrePress2Buffer",
        "PostPress1_2Buffer",
    ]

    LOWER_BOUND = 1
    UPPER_BOUND = 5

    @classmethod
    def n_buffers(cls):
        return len(cls.BUFFER_NAMES)

    @classmethod
    def bounds(cls):
        n = cls.n_buffers()
        xl = np.full(n, cls.LOWER_BOUND, dtype=float)
        xu = np.full(n, cls.UPPER_BOUND, dtype=float)
        return xl, xu


class CandidateSolution:
    """
    Encodes a candidate solution as a vector of buffer capacities.
    """

    def __init__(self, buffer_caps):
        # buffer_caps: iterable of ints in [1,5]
        self.buffer_caps = np.array(buffer_caps, dtype=int)

    def encode(self):
        # Return as float array for pymoo
        return self.buffer_caps.astype(float)

    @classmethod
    def decode(cls, x):
        # x: numpy array of floats -> round and clip to [1,5]
        x_int = np.rint(x).astype(int)
        x_int = np.clip(
            x_int, SystemModelBuffers.LOWER_BOUND, SystemModelBuffers.UPPER_BOUND
        )
        return cls(x_int)

    def apply_to_kwargs(self, kwargs):
        kwargs = dict(kwargs) if kwargs is not None else {}
        kwargs["buffer_caps"] = {
            name: int(cap)
            for name, cap in zip(SystemModelBuffers.BUFFER_NAMES, self.buffer_caps)
        }
        return kwargs


class RunSimulationFunction:
    """
    Wrapper around the original run_simulation.
    """

    def __init__(self, warmup=None, measure_until=None, base_seed=None):
        self.warmup = warmup
        self.measure_until = measure_until
        self.base_seed = RANDOM_SEED if base_seed is None else base_seed

    def __call__(self, buffer_caps, seed_offset=0):
        seed = self.base_seed + seed_offset
        kwargs = {}
        kwargs["warmup"] = self.warmup if self.warmup is not None else WARMUP_SECONDS
        kwargs["measure_until"] = (
            self.measure_until if self.measure_until is not None else MEASURE_UNTIL
        )
        kwargs["seed"] = seed

        kwargs["buffer_caps"] = {
            name: int(cap)
            for name, cap in zip(
                SystemModelBuffers.BUFFER_NAMES, buffer_caps.buffer_caps
            )
        }

        result = run_simulation(
            kwargs["seed"],
            warmup=kwargs["warmup"],
            measure_until=kwargs["measure_until"],
            buffer_caps=kwargs["buffer_caps"],
        )
        return result


class SimulatorInterface:
    """
    Adapter for the original simulation code.
    """

    def __init__(self, run_sim_func):
        self.run_sim = run_sim_func

    def run(self, candidate, seed_offset=0):
        return self.run_sim(candidate, seed_offset=seed_offset)


class MetricsCollector:
    """
    Collects throughput, wip, produced_parts and machine energy per run.
    """

    @staticmethod
    def collect(sim_result):
        overall = sim_result["overall"]
        machine_energy = sim_result["machine_energy"]
        throughput = overall["throughput"]
        wip = overall["wip"]
        produced_parts = overall["produced_parts"]
        total_energy = sum(m["total_energy"] for m in machine_energy.values())
        return {
            "throughput": throughput,
            "wip": wip,
            "produced_parts": produced_parts,
            "total_energy": total_energy,
        }


class ObjectiveEvaluator:
    """
    Computes objective vector:
        f1 = WIP (to minimize)
        f2 = -Throughput (to minimize, since NSGA2 is minimization-based)
    """

    def __init__(self, simulator, n_replications=1):
        self.simulator = simulator
        self.n_replications = n_replications

    def evaluate(self, candidate):
        wip_list = []
        thr_list = []

        for r in range(self.n_replications):
            sim_res = self.simulator.run(candidate, seed_offset=r)
            metrics = MetricsCollector.collect(sim_res)
            wip_list.append(metrics["wip"])
            thr_list.append(metrics["throughput"])

        mean_wip = statistics.mean(wip_list)
        mean_thr = statistics.mean(thr_list)

        f1 = mean_wip
        f2 = -mean_thr
        return np.array([f1, f2], dtype=float), mean_wip, mean_thr


class ParetoArchive:
    """
    Stores non-dominated CandidateSolution objects and their objective values.
    """

    def __init__(self):
        self.solutions = []  # list of (CandidateSolution, f)

    def update(self, candidate, f):
        new_solutions = []
        dominated = False
        for c, fv in self.solutions:
            if self._dominates(fv, f):
                dominated = True
                break
            if not self._dominates(f, fv):
                new_solutions.append((c, fv))
        if not dominated:
            new_solutions.append((candidate, f))
        self.solutions = new_solutions

    @staticmethod
    def _dominates(f1, f2):
        return np.all(f1 <= f2) and np.any(f1 < f2)


class SelectionOperator:
    """
    Performs parent selection using tournament selection (delegated to pymoo).
    """

    def __init__(self, selection):
        self.selection = selection

    def select(self, pop, n_parents):
        return self.selection.do(pop, n_parents)


class VariationOperator:
    """
    Applies crossover and mutation, ensuring capacities remain in {1,...,5}.
    """

    def __init__(self, crossover, mutation):
        self.crossover = crossover
        self.mutation = mutation

    def vary(self, problem, pop):
        off = self.crossover.do(problem, pop)
        off = self.mutation.do(problem, off)
        for ind in off:
            x = ind.X
            x = np.rint(x).astype(int)
            x = np.clip(
                x, SystemModelBuffers.LOWER_BOUND, SystemModelBuffers.UPPER_BOUND
            )
            ind.X = x.astype(float)
        return off


class PopulationManager:
    """
    Manages population initialization and evaluation.
    """

    def __init__(self, problem, evaluator):
        self.problem = problem
        self.evaluator = evaluator

    def initialize(self, pop_size):
        n_var = SystemModelBuffers.n_buffers()
        pop = []
        for _ in range(pop_size):
            caps = np.random.randint(
                SystemModelBuffers.LOWER_BOUND,
                SystemModelBuffers.UPPER_BOUND + 1,
                size=n_var,
            )
            cand = CandidateSolution(caps)
            x = cand.encode()
            f, _, _ = self.evaluator.evaluate(cand)
            pop.append({"X": x, "F": f})
        return pop

    def evaluate_population(self, pop):
        for ind in pop:
            x = ind["X"]
            cand = CandidateSolution.decode(x)
            f, _, _ = self.evaluator.evaluate(cand)
            ind["F"] = f
        return pop


class Scheduler:
    """
    Dispatches simulation jobs (sequential in this simple implementation).
    """

    def __init__(self, simulator):
        self.simulator = simulator

    def run_candidate(self, candidate, seed_offset=0):
        return self.simulator.run(candidate, seed_offset=seed_offset)


class BufferOptimizationProblem(ElementwiseProblem):
    """
    Pymoo problem wrapper around the simulation-based objective evaluator.
    """

    def __init__(self, evaluator, results_recorder):
        xl, xu = SystemModelBuffers.bounds()
        super().__init__(n_var=SystemModelBuffers.n_buffers(), n_obj=2, xl=xl, xu=xu)
        self.evaluator = evaluator
        self.results_recorder = results_recorder
        self.eval_counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        cand = CandidateSolution.decode(x)
        f, mean_wip, mean_thr = self.evaluator.evaluate(cand)
        out["F"] = f
        # Record this evaluation
        self.results_recorder.record(self.eval_counter, cand, mean_wip, mean_thr)
        self.eval_counter += 1


class ParetoCallback(Callback):
    def __init__(self, archive):
        super().__init__()
        self.archive = archive

    def notify(self, algorithm):
        pop = algorithm.pop
        for ind in pop:
            x = ind.X
            f = ind.F
            cand = CandidateSolution.decode(x)
            self.archive.update(cand, f)


class UserInterface:
    """
    Minimal stub for setting constraints and visualizing results.
    """

    def __init__(self):
        self.budget = None
        self.stop_criteria = None

    def set_budget(self, n_evals):
        self.budget = n_evals

    def set_stop_criteria(self, generations):
        self.stop_criteria = generations

    def show_pareto_front(self, archive):
        print("Pareto Front (wip, -throughput, capacities):")
        for cand, f in archive.solutions:
            print(f"F={f}, caps={cand.buffer_caps}")


class ResultsRecorder:
    """
    Records all evaluated solutions with their KPI values.
    """

    def __init__(self):
        self.records = []

    def record(self, eval_id, candidate, wip, throughput):
        rec = {
            "eval_id": eval_id,
            "PostLoadingBuffer": int(candidate.buffer_caps[0]),
            "PostConveyorBuffer": int(candidate.buffer_caps[1]),
            "PostWashingBuffer": int(candidate.buffer_caps[2]),
            "PrePress1Buffer": int(candidate.buffer_caps[3]),
            "PrePress2Buffer": int(candidate.buffer_caps[4]),
            "PostPress1_2Buffer": int(candidate.buffer_caps[5]),
            "wip": float(wip),
            "throughput": float(throughput),
        }
        self.records.append(rec)

    def to_csv(self, filename):
        if not self.records:
            return
        fieldnames = [
            "eval_id",
            "PostLoadingBuffer",
            "PostConveyorBuffer",
            "PostWashingBuffer",
            "PrePress1Buffer",
            "PrePress2Buffer",
            "PostPress1_2Buffer",
            "wip",
            "throughput",
        ]
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)


class MOO_Controller:
    """
    Orchestrates the NSGA-II optimization of buffer capacities.
    """

    def __init__(
        self,
        pop_size=20,
        n_gen=5,
        n_replications=1,
        warmup=None,
        measure_until=None,
        base_seed=None,
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen

        self.run_sim_func = RunSimulationFunction(
            warmup=warmup, measure_until=measure_until, base_seed=base_seed
        )
        self.simulator = SimulatorInterface(self.run_sim_func)
        self.scheduler = Scheduler(self.simulator)
        self.evaluator = ObjectiveEvaluator(self.simulator, n_replications=n_replications)
        self.results_recorder = ResultsRecorder()
        self.problem = BufferOptimizationProblem(self.evaluator, self.results_recorder)

        self.archive = ParetoArchive()
        self.ui = UserInterface()

        # NSGA-II operators
        def tournament_comp(pop, P, **kwargs):
            # P is (n_tournaments, n_competitors)
            # Return index of winner in each tournament (0 or 1 for binary tournament)
            F = pop.get("F")
            n_tournaments, n_comp = P.shape
            S = np.full(n_tournaments, 0, dtype=int)
            for i in range(n_tournaments):
                best = P[i, 0]
                for j in range(1, n_comp):
                    cand = P[i, j]
                    if np.all(F[cand] <= F[best]) and np.any(F[cand] < F[best]):
                        best = cand
                S[i] = best
            return S

        self.selection = TournamentSelection(func_comp=tournament_comp)
        self.crossover = SBX(prob=0.9, eta=15)
        # Set mutation probability to 1/n_var to avoid NoneType error
        self.mutation = PM(prob=1.0 / SystemModelBuffers.n_buffers(), eta=20)

        class IntBufferSampling(Sampling):
            def _do(self, problem, n_samples, **kwargs):
                n_var = problem.n_var
                X = np.zeros((n_samples, n_var))
                for i in range(n_samples):
                    caps = np.random.randint(
                        SystemModelBuffers.LOWER_BOUND,
                        SystemModelBuffers.UPPER_BOUND + 1,
                        size=n_var,
                    )
                    X[i, :] = caps.astype(float)
                return X

        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntBufferSampling(),
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=True,
        )

    def run(self):
        termination = get_termination("n_gen", self.n_gen)
        callback = ParetoCallback(self.archive)

        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            callback=callback,
            verbose=True,
        )

        self.ui.show_pareto_front(self.archive)
        # Write all evaluated solutions to CSV
        self.results_recorder.to_csv("moo_simulation_results.csv")
        return res, self.archive


if __name__ == "__main__":
    controller = MOO_Controller(
        pop_size=20,
        n_gen=5,
        n_replications=1,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        base_seed=RANDOM_SEED,
    )
    result, archive = controller.run()