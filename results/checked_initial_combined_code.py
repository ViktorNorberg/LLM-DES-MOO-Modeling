import simpy
import random
import statistics
from collections import Counter
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv

RANDOM_SEED = 11

SIM_TIME = 691200        # 8 days
WARMUP_SECONDS = 86400   # 1 day
MEASURE_UNTIL = SIM_TIME


def production_wait_time(now: float) -> float:
    """
    7-day-periodic production stop windows:
    - Friday 17:00–Saturday 07:00
    - Saturday 17:00–Sunday 07:00

    day: 0=Mon ... 6=Sun
    """
    SEC_PER_DAY = 86400
    day = int((now // SEC_PER_DAY) % 7)
    t = now % SEC_PER_DAY

    def until_7():
        end = 7 * 3600
        return max(0.0, end - t) if t < end else 0.0

    def until_24():
        end = 24 * 3600
        return max(0.0, end - t) if t < end else 0.0

    # Friday
    if day == 4:
        # 17:00–24:00
        if 17 * 3600 <= t < 24 * 3600:
            return until_24()
        return 0.0

    # Saturday
    if day == 5:
        # 00:00–07:00
        if 0 <= t < 7 * 3600:
            return until_7()
        # 17:00–24:00
        if 17 * 3600 <= t < 24 * 3600:
            return until_24()
        return 0.0

    # Sunday
    if day == 6:
        # 00:00–07:00
        if 0 <= t < 7 * 3600:
            return until_7()
        return 0.0

    # Other weekdays: production allowed
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
        self.store = simpy.Store(env, capacity=cap)   # holds 'ready' items
        self.tokens = simpy.Container(env, init=cap, capacity=cap)  # global slots
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


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Buffers (all with defined capacities, matching given spec)
    PostLoadingBuffer = DelayBuffer(env, cap=2, delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=2, delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=2, delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=3, delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=3, delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=3, delay=32)  # PostPress1&Press2Buffer

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000)     # large but defined
    sink = simpy.Store(env, capacity=100000)        # final sink
    defects = simpy.Store(env, capacity=100000)     # defect sink

    # Helper stores for parallel routing of presses
    # capacities set explicitly
    hantering_out = simpy.Store(env, capacity=6)    # 3 + 3 to match both pre-press buffers
    post_press1_out = simpy.Store(env, capacity=3)
    post_press2_out = simpy.Store(env, capacity=3)

    # Machines
    Loading_robot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=PostLoadingBuffer,
        process_time=12.0,
        availability=90.49, mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    Conveyor_belt = Machine(
        env, "Conveyor belt",
        input_buffer=PostLoadingBuffer,
        output_buffer=PostConveyorBuffer,
        process_time=6.0,
        availability=100.0, mttr=1.0,
        working_power=kwh_per_sec(0.00),
        waiting_power=kwh_per_sec(0.00),
    )

    Washing_machine = Machine(
        env, "Washing machine",
        input_buffer=PostConveyorBuffer,
        output_buffer=PostWashingBuffer,
        process_time=14.0,
        availability=80.89, mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    Hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=hantering_out,  # output goes to splitter
        process_time=25.0,
        availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    Presses_cell_1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=post_press1_out,
        process_time=175.0,
        availability=87.79, mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    Presses_cell_2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=post_press2_out,
        process_time=176.0,
        availability=87.69, mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    Quality_station_cell = Machine(
        env, "Quality station cell",
        input_buffer=PostPress12Buffer,
        output_buffer=sink,
        process_time=41.0,
        availability=85.87, mttr=66.0,
        working_power=kwh_per_sec(0.84),
        waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089,
        defect_sink=defects,
    )

    machines_list = [
        Loading_robot,
        Conveyor_belt,
        Washing_machine,
        Hantering_cell,
        Presses_cell_1,
        Presses_cell_2,
        Quality_station_cell,
    ]

    # Routing logic
    # 1) From Hantering_cell to PrePress1Buffer & PrePress2Buffer via splitter.
    env.process(splitter(env, hantering_out, PrePress1Buffer, PrePress2Buffer))

    # 2) From PostPress1&Press2 (parallel) to PostPress12Buffer via merger
    merger(env, post_press1_out, post_press2_out, PostPress12Buffer)

    # Start part generation into raw_input
    env.process(part_generator(env, raw_input))

    # Run warm-up
    env.run(until=warmup)

    # Reset statistics after warm-up
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)

    wip_samples = []
    delay_buffers = [
        PostLoadingBuffer,
        PostConveyorBuffer,
        PostWashingBuffer,
        PrePress1Buffer,
        PrePress2Buffer,
        PostPress12Buffer,
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

    total_produced = len(sink.items) - produced_count_before
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


def run_simulation_with_caps(seed, capacities, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    """
    Wrapper around the original run_simulation that injects buffer capacities.
    This function is a modified copy of run_simulation where the capacities
    of the DelayBuffers are taken from the 'capacities' dict.
    """

    random.seed(seed)
    env = simpy.Environment()

    # Extract capacities from dict
    cap_PostLoadingBuffer = capacities["PostLoadingBuffer"]
    cap_PostConveyorBuffer = capacities["PostConveyorBuffer"]
    cap_PostWashingBuffer = capacities["PostWashingBuffer"]
    cap_PrePress1Buffer = capacities["PrePress1Buffer"]
    cap_PrePress2Buffer = capacities["PrePress2Buffer"]
    cap_PostPress12Buffer = capacities["PostPress12Buffer"]

    # Buffers with adjustable capacities
    PostLoadingBuffer = DelayBuffer(env, cap=cap_PostLoadingBuffer, delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=cap_PostConveyorBuffer, delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=cap_PostWashingBuffer, delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=cap_PrePress1Buffer, delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=cap_PrePress2Buffer, delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=cap_PostPress12Buffer, delay=32)

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Helper stores for parallel routing of presses
    hantering_out = simpy.Store(env, capacity=cap_PrePress1Buffer + cap_PrePress2Buffer)
    post_press1_out = simpy.Store(env, capacity=cap_PrePress1Buffer)
    post_press2_out = simpy.Store(env, capacity=cap_PrePress2Buffer)

    # Machines
    Loading_robot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=PostLoadingBuffer,
        process_time=12.0,
        availability=90.49, mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    Conveyor_belt = Machine(
        env, "Conveyor belt",
        input_buffer=PostLoadingBuffer,
        output_buffer=PostConveyorBuffer,
        process_time=6.0,
        availability=100.0, mttr=1.0,
        working_power=kwh_per_sec(0.00),
        waiting_power=kwh_per_sec(0.00),
    )

    Washing_machine = Machine(
        env, "Washing machine",
        input_buffer=PostConveyorBuffer,
        output_buffer=PostWashingBuffer,
        process_time=14.0,
        availability=80.89, mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    Hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=hantering_out,
        process_time=25.0,
        availability=97.79, mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    Presses_cell_1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=post_press1_out,
        process_time=175.0,
        availability=87.79, mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    Presses_cell_2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=post_press2_out,
        process_time=176.0,
        availability=87.69, mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    Quality_station_cell = Machine(
        env, "Quality station cell",
        input_buffer=PostPress12Buffer,
        output_buffer=sink,
        process_time=41.0,
        availability=85.87, mttr=66.0,
        working_power=kwh_per_sec(0.84),
        waiting_power=kwh_per_sec(0.58),
        defect_rate=0.089,
        defect_sink=defects,
    )

    machines_list = [
        Loading_robot,
        Conveyor_belt,
        Washing_machine,
        Hantering_cell,
        Presses_cell_1,
        Presses_cell_2,
        Quality_station_cell,
    ]

    # Routing logic
    env.process(splitter(env, hantering_out, PrePress1Buffer, PrePress2Buffer))
    merger(env, post_press1_out, post_press2_out, PostPress12Buffer)

    # Start part generation into raw_input
    env.process(part_generator(env, raw_input))

    # Run warm-up
    env.run(until=warmup)

    # Reset statistics after warm-up
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)

    wip_samples = []
    delay_buffers = [
        PostLoadingBuffer,
        PostConveyorBuffer,
        PostWashingBuffer,
        PrePress1Buffer,
        PrePress2Buffer,
        PostPress12Buffer,
    ]

    def sample_wip(env_local):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env_local.timeout(60)

    env.process(sample_wip(env))
    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = float(np.mean(wip_samples)) if wip_samples else 0.0

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


class SimulationAdapter:
    """
    Adapter to connect the optimization variables (buffer capacities)
    to the existing run_simulation function.
    """

    def __init__(self, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
        self.warmup = warmup
        self.measure_until = measure_until

    def _run_single(self, seed, capacities):
        """
        Run a single simulation with given random seed and buffer capacities.
        """
        return run_simulation_with_caps(seed, capacities, self.warmup, self.measure_until)

    def evaluate(self, capacities, runs=3, base_seed=RANDOM_SEED):
        """
        Evaluate a configuration (capacities) by running the simulation multiple times
        and averaging throughput and WIP.
        capacities: dict with keys:
            PostLoadingBuffer, PostConveyorBuffer, PostWashingBuffer,
            PrePress1Buffer, PrePress2Buffer, PostPress12Buffer
        """
        throughputs = []
        wips = []
        for i in range(runs):
            seed = base_seed + i
            res = self._run_single(seed, capacities)
            overall = res["overall"]
            throughputs.append(overall["throughput"])
            wips.append(overall["wip"])
        mean_throughput = float(np.mean(throughputs)) if throughputs else 0.0
        mean_wip = float(np.mean(wips)) if wips else 0.0
        return mean_wip, mean_throughput


class ProductionLineProblem(Problem):
    """
    pymoo Problem definition for optimizing buffer capacities
    to minimize WIP and maximize throughput (implemented as minimizing -throughput).
    Decision variables:
        x[0] -> PostLoadingBuffer capacity
        x[1] -> PostConveyorBuffer capacity
        x[2] -> PostWashingBuffer capacity
        x[3] -> PrePress1Buffer capacity
        x[4] -> PrePress2Buffer capacity
        x[5] -> PostPress12Buffer capacity
    """

    def __init__(self, adapter: SimulationAdapter, runs_per_eval=3):
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=0,
            xl=np.array([1, 1, 1, 1, 1, 1]),
            xu=np.array([5, 5, 5, 5, 5, 5]),
            type_var=int
        )
        self.adapter = adapter
        self.runs_per_eval = runs_per_eval

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X is a 2D array of shape (n_individuals, 6)
        """
        n = X.shape[0]
        F = np.zeros((n, self.n_obj), dtype=float)

        for i in range(n):
            x = X[i, :]
            capacities = {
                "PostLoadingBuffer": int(x[0]),
                "PostConveyorBuffer": int(x[1]),
                "PostWashingBuffer": int(x[2]),
                "PrePress1Buffer": int(x[3]),
                "PrePress2Buffer": int(x[4]),
                "PostPress12Buffer": int(x[5]),
            }
            mean_wip, mean_throughput = self.adapter.evaluate(capacities, runs=self.runs_per_eval)
            # Objectives: minimize WIP, maximize throughput -> minimize -throughput
            F[i, 0] = mean_wip
            F[i, 1] = -mean_throughput

        out["F"] = F


def run_nsga2_optimization(
    population_size=20,
    n_generations=5,
    runs_per_eval=3,
    warmup=WARMUP_SECONDS,
    measure_until=MEASURE_UNTIL,
    random_seed=RANDOM_SEED
):
    """
    Run NSGA-II optimization on the production line simulation.
    Returns the pymoo result object containing the Pareto front and decision variables,
    and the algorithm history for CSV export.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    adapter = SimulationAdapter(warmup=warmup, measure_until=measure_until)
    problem = ProductionLineProblem(adapter=adapter, runs_per_eval=runs_per_eval)

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(eta=20)

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_generations)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=random_seed,
        save_history=True,
        verbose=True
    )

    return res, algorithm


def export_history_to_csv(algorithm, filename="moo_simulation_results.csv"):
    """
    Export all solutions from every generation with their KPIs to a CSV file.
    Columns:
        generation, solution_index, PostLoadingBuffer, PostConveyorBuffer,
        PostWashingBuffer, PrePress1Buffer, PrePress2Buffer, PostPress12Buffer,
        wip, throughput
    """
    fieldnames = [
        "generation",
        "solution_index",
        "PostLoadingBuffer",
        "PostConveyorBuffer",
        "PostWashingBuffer",
        "PrePress1Buffer",
        "PrePress2Buffer",
        "PostPress12Buffer",
        "wip",
        "throughput"
    ]

    rows = []
    history = algorithm.history

    for gen_idx, entry in enumerate(history):
        pop = entry.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None:
            continue
        for i in range(len(X)):
            x = X[i]
            f = F[i]
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in optimization
            row = {
                "generation": gen_idx,
                "solution_index": i,
                "PostLoadingBuffer": int(x[0]),
                "PostConveyorBuffer": int(x[1]),
                "PostWashingBuffer": int(x[2]),
                "PrePress1Buffer": int(x[3]),
                "PrePress2Buffer": int(x[4]),
                "PostPress12Buffer": int(x[5]),
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    # Run NSGA-II optimization
    result, algorithm = run_nsga2_optimization(
        population_size=20,
        n_generations=5,
        runs_per_eval=3,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        random_seed=RANDOM_SEED
    )

    # Export all solutions from every generation to CSV
    export_history_to_csv(algorithm, filename="moo_simulation_results.csv")

    # Print Pareto-optimal solutions as a table
    X = result.X
    F = result.F

    header = [
        "PostLoadingBuffer",
        "PostConveyorBuffer",
        "PostWashingBuffer",
        "PrePress1Buffer",
        "PrePress2Buffer",
        "PostPress12Buffer",
        "wip",
        "throughput"
    ]
    print("\t".join(header))
    for i in range(len(X)):
        x = X[i]
        f = F[i]
        wip = f[0]
        throughput = -f[1]
        row = [
            str(int(x[0])),
            str(int(x[1])),
            str(int(x[2])),
            str(int(x[3])),
            str(int(x[4])),
            str(int(x[5])),
            f"{wip:.4f}",
            f"{throughput:.4f}"
        ]
        print("\t".join(row))