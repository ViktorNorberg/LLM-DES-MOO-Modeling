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

SIM_TIME = 691200          # 8 days
WARMUP_SECONDS = 86400     # 1 day
MEASURE_UNTIL = SIM_TIME


def production_wait_time(now: float) -> float:
    """
    Compute how long (in seconds) the machine must wait until production is allowed.

    New stop rules (7-day periodic):
    - Friday 17:00  -> Saturday 07:00
    - Saturday 17:00 -> Sunday 07:00
    """
    SEC_PER_DAY = 86400
    day = int((now // SEC_PER_DAY) % 7)  # 0=Mon ... 6=Sun
    time_of_day = now % SEC_PER_DAY

    def secs(h, m=0, s=0):
        return h * 3600 + m * 60 + s

    if day == 4:  # Friday
        stop1_start = secs(17)
        stop1_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        return 0.0

    if day == 5:  # Saturday
        stop1_start = secs(0)
        stop1_end = secs(7)
        stop2_start = secs(17)
        stop2_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        if stop2_start <= time_of_day < stop2_end:
            return max(0.0, stop2_end - time_of_day)
        return 0.0

    if day == 6:  # Sunday
        stop_start = secs(0)
        stop_end = secs(7)
        if stop_start <= time_of_day < stop_end:
            return max(0.0, stop_end - time_of_day)
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


def run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    PostLoadingBuffer = DelayBuffer(env, cap=2, delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=2, delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=2, delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=3, delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=3, delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=3, delay=32)

    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    Loading_robot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=PostLoadingBuffer,
        process_time=12.0,
        availability=90.49,
        mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    Conveyor_belt = Machine(
        env, "Conveyor belt",
        input_buffer=PostLoadingBuffer,
        output_buffer=PostConveyorBuffer,
        process_time=6.0,
        availability=100.0,
        mttr=1.0,
        working_power=kwh_per_sec(0.0),
        waiting_power=kwh_per_sec(0.0),
    )

    Washing_machine = Machine(
        env, "Washing machine",
        input_buffer=PostConveyorBuffer,
        output_buffer=PostWashingBuffer,
        process_time=14.0,
        availability=80.89,
        mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    Hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=None,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    hantering_out = simpy.Store(env, capacity=6)
    Hantering_cell.output_buffer = hantering_out

    env.process(splitter(env, hantering_out, PrePress1Buffer, PrePress2Buffer))

    Presses_cell_1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=press1_out,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    Presses_cell_2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=press2_out,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    merger(env, press1_out, press2_out, PostPress12Buffer)

    Quality_station_cell = Machine(
        env, "Quality station cell",
        input_buffer=PostPress12Buffer,
        output_buffer=sink,
        process_time=41.0,
        availability=85.87,
        mttr=66.0,
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

    env.process(part_generator(env, raw_input))

    env.run(until=warmup)

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

    bottleneck_data = {}
    for m in machines_list:
        m_th = m.processed_count / hours if hours > 0 else 0.0
        util = (m.working_time / (measure_until - warmup)) * 100.0 if (measure_until > warmup) else 0.0
        bottleneck_data[m.name] = {
            "throughput": m_th,
            "utilization": util,
            "processed_count": m.processed_count}

    result["bottleneck"] = {
        "top_3": sorted(bottleneck_data.items(),
                        key=lambda kv: kv[1]["utilization"],
                        reverse=True)[:3],
        "all": bottleneck_data
    }
    return result


def run_simulation_with_buffers(seed,
                                post_loading_cap,
                                post_conveyor_cap,
                                post_washing_cap,
                                pre_press1_cap,
                                pre_press2_cap,
                                post_press12_cap,
                                warmup=WARMUP_SECONDS,
                                measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    PostLoadingBuffer = DelayBuffer(env, cap=post_loading_cap, delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=post_conveyor_cap, delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=post_washing_cap, delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=pre_press1_cap, delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=pre_press2_cap, delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=post_press12_cap, delay=32)

    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    Loading_robot = Machine(
        env, "Loading robot",
        input_buffer=raw_input,
        output_buffer=PostLoadingBuffer,
        process_time=12.0,
        availability=90.49,
        mttr=68.0,
        working_power=kwh_per_sec(0.72),
        waiting_power=kwh_per_sec(0.25),
    )

    Conveyor_belt = Machine(
        env, "Conveyor belt",
        input_buffer=PostLoadingBuffer,
        output_buffer=PostConveyorBuffer,
        process_time=6.0,
        availability=100.0,
        mttr=1.0,
        working_power=kwh_per_sec(0.0),
        waiting_power=kwh_per_sec(0.0),
    )

    Washing_machine = Machine(
        env, "Washing machine",
        input_buffer=PostConveyorBuffer,
        output_buffer=PostWashingBuffer,
        process_time=14.0,
        availability=80.89,
        mttr=269.0,
        working_power=kwh_per_sec(35.24),
        waiting_power=kwh_per_sec(4.28),
    )

    Hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=None,
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    hantering_out = simpy.Store(env, capacity=6)
    Hantering_cell.output_buffer = hantering_out

    env.process(splitter(env, hantering_out, PrePress1Buffer, PrePress2Buffer))

    Presses_cell_1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=press1_out,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    Presses_cell_2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=press2_out,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    merger(env, press1_out, press2_out, PostPress12Buffer)

    Quality_station_cell = Machine(
        env, "Quality station cell",
        input_buffer=PostPress12Buffer,
        output_buffer=sink,
        process_time=41.0,
        availability=85.87,
        mttr=66.0,
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

    env.process(part_generator(env, raw_input))

    env.run(until=warmup)

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

    return {
        "throughput": throughput,
        "wip": avg_wip,
        "produced_parts": total_produced
    }


class SimulationRunner:
    def __init__(self, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL, seeds=None):
        self.warmup = warmup
        self.measure_until = measure_until
        if seeds is None:
            self.seeds = [RANDOM_SEED + i for i in range(3)]
        else:
            self.seeds = seeds

    def run_with_params(self, params, seed):
        post_loading_cap, post_conveyor_cap, post_washing_cap, pre_press1_cap, pre_press2_cap, post_press12_cap = params
        return run_simulation_with_buffers(
            seed=seed,
            post_loading_cap=int(post_loading_cap),
            post_conveyor_cap=int(post_conveyor_cap),
            post_washing_cap=int(post_washing_cap),
            pre_press1_cap=int(pre_press1_cap),
            pre_press2_cap=int(pre_press2_cap),
            post_press12_cap=int(post_press12_cap),
            warmup=self.warmup,
            measure_until=self.measure_until
        )


class ObjectiveAggregator:
    @staticmethod
    def compute_throughput(raw_metrics_list):
        return statistics.mean(m["throughput"] for m in raw_metrics_list)

    @staticmethod
    def compute_wip(raw_metrics_list):
        return statistics.mean(m["wip"] for m in raw_metrics_list)


class Evaluator:
    def __init__(self, simulation_runner, seeds=None):
        self.simulation_runner = simulation_runner
        if seeds is None:
            self.seeds = simulation_runner.seeds
        else:
            self.seeds = seeds

    def evaluate_candidate(self, candidate_params):
        raw_metrics_list = []
        for s in self.seeds:
            metrics = self.simulation_runner.run_with_params(candidate_params, s)
            raw_metrics_list.append(metrics)
        th = ObjectiveAggregator.compute_throughput(raw_metrics_list)
        wip = ObjectiveAggregator.compute_wip(raw_metrics_list)
        return th, wip


class ParetoArchive:
    def __init__(self):
        self.pareto_front = []

    @staticmethod
    def _dominates(a_obj, b_obj):
        better_or_equal = all(a <= b for a, b in zip(a_obj, b_obj))
        strictly_better = any(a < b for a, b in zip(a_obj, b_obj))
        return better_or_equal and strictly_better

    def update_with(self, candidate_params, objectives):
        new_entry = {"params": candidate_params, "objs": objectives}
        non_dominated = []
        dominated = False
        for e in self.pareto_front:
            if self._dominates(e["objs"], objectives):
                dominated = True
                break
            if not self._dominates(objectives, e["objs"]):
                non_dominated.append(e)
        if not dominated:
            non_dominated.append(new_entry)
        self.pareto_front = non_dominated

    def export_front(self):
        return self.pareto_front


class ResultsStorage:
    def __init__(self):
        self.run_history = []
        self.final_pareto_front = None
        self.best_solutions = None

    def save_run(self, population, objectives):
        self.run_history.append({
            "population": population,
            "objectives": objectives
        })

    def set_final_front(self, front):
        self.final_pareto_front = front

    def export(self):
        return {
            "history": self.run_history,
            "final_pareto_front": self.final_pareto_front,
            "best_solutions": self.best_solutions
        }


class BufferCapacityProblem(Problem):
    def __init__(self, evaluator):
        super().__init__(n_var=6,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([1, 1, 1, 1, 1, 1]),
                         xu=np.array([6, 6, 6, 6, 6, 6]),
                         elementwise_evaluation=True)
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        params = [int(round(v)) for v in x]
        throughput, wip = self.evaluator.evaluate_candidate(params)
        f1 = -throughput
        f2 = wip
        out["F"] = np.array([f1, f2])


class OptimizationController:
    def __init__(self,
                 population_size=20,
                 generations=4,
                 mutation_rate=0.1,
                 warmup=WARMUP_SECONDS,
                 measure_until=MEASURE_UNTIL,
                 seeds=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.simulation_runner = SimulationRunner(warmup=warmup,
                                                  measure_until=measure_until,
                                                  seeds=seeds)
        self.evaluator = Evaluator(self.simulation_runner)
        self.problem = BufferCapacityProblem(self.evaluator)
        self.archive = ParetoArchive()
        self.results_storage = ResultsStorage()

    def run_loop(self):
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20, prob=self.mutation_rate),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.generations)

        res = minimize(self.problem,
                       algorithm,
                       termination,
                       seed=RANDOM_SEED,
                       save_history=True,
                       verbose=False)

        X = res.X
        F = res.F

        for x, f in zip(X, F):
            params = [int(round(v)) for v in x]
            throughput = -float(f[0])
            wip = float(f[1])
            self.archive.update_with(params, (throughput, wip))

        for algo_gen in res.history:
            pop = algo_gen.pop.get("X")
            objs = algo_gen.pop.get("F")
            self.results_storage.save_run(population=pop, objectives=objs)

        front = self.archive.export_front()
        self.results_storage.set_final_front(front)
        self.results_storage.best_solutions = front
        return front, self.results_storage, res


def write_results_to_csv(results_storage, res, filename="moo_simulation_results.csv"):
    fieldnames = [
        "generation",
        "solution_index",
        "post_loading_cap",
        "post_conveyor_cap",
        "post_washing_cap",
        "pre_press1_cap",
        "pre_press2_cap",
        "post_press12_cap",
        "throughput",
        "wip"
    ]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        history = res.history
        for gen_index, algo_gen in enumerate(history):
            pop = algo_gen.pop.get("X")
            objs = algo_gen.pop.get("F")
            for i, (x, f) in enumerate(zip(pop, objs)):
                params = [int(round(v)) for v in x]
                throughput = -float(f[0])
                wip = float(f[1])
                row = {
                    "generation": gen_index,
                    "solution_index": i,
                    "post_loading_cap": params[0],
                    "post_conveyor_cap": params[1],
                    "post_washing_cap": params[2],
                    "pre_press1_cap": params[3],
                    "pre_press2_cap": params[4],
                    "post_press12_cap": params[5],
                    "throughput": throughput,
                    "wip": wip
                }
                writer.writerow(row)


if __name__ == "__main__":
    controller = OptimizationController(
        population_size=20,
        generations=4,
        mutation_rate=0.1,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        seeds=[RANDOM_SEED + i for i in range(3)]
    )

    pareto_front, results_storage, res = controller.run_loop()

    write_results_to_csv(results_storage, res, filename="moo_simulation_results.csv")

    print("Final Pareto Front (buffer capacities -> throughput, wip):")
    print("PostLoad, PostConv, PostWash, PreP1, PreP2, PostP12, Throughput, WIP")
    for entry in pareto_front:
        params = entry["params"]
        th, wip = entry["objs"]
        print(f"{params[0]}, {params[1]}, {params[2]}, {params[3]}, {params[4]}, {params[5]}, {th:.3f}, {wip:.3f}")