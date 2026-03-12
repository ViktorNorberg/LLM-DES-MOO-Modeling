import random
import statistics
import numpy as np
from collections import Counter
import csv

import simpy

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

RANDOM_SEED = 11

SIM_TIME = 3600          # 8 days
WARMUP_SECONDS = 100     # 1 day
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

    # Friday (4): stop 17:00–24:00, then continue on Saturday 00:00–07:00
    # Saturday (5): stop 17:00–24:00, then continue on Sunday 00:00–07:00
    if day == 4:  # Friday
        stop1_start = secs(17)
        stop1_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        return 0.0

    if day == 5:  # Saturday
        # 00:00–07:00 is still the Friday–Saturday block (already handled by previous day),
        # but since function must be purely local in time, model only Sat 17–24 and Sun 00–07
        stop1_start = secs(0)
        stop1_end = secs(7)
        stop2_start = secs(17)
        stop2_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        if stop2_start <= time_of_day < stop2_end:
            return max(0.0, stop2_end - time_of_day)
        return 0.0

    if day == 6:  # Sunday: 00–07 blocked from Sat, but model only local window 00–07
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


def run_simulation_with_capacities(seed,
                                   post_loading_cap=2,
                                   post_conveyor_cap=2,
                                   post_washing_cap=2,
                                   pre_press1_cap=3,
                                   pre_press2_cap=3,
                                   post_press12_cap=3,
                                   warmup=WARMUP_SECONDS,
                                   measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Buffers (capacities from parameters)
    PostLoadingBuffer = DelayBuffer(env, cap=int(post_loading_cap), delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=int(post_conveyor_cap), delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=int(post_washing_cap), delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=int(pre_press1_cap), delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=int(pre_press2_cap), delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=int(post_press12_cap), delay=32)

    # Helper buffer between parallel presses and common PostPress12Buffer
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    # Machines following given routing and station table
    # Loading robot -> Conveyor belt
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

    # PostLoadingBuffer -> Conveyor belt -> PostConveyorBuffer
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

    # Conveyor belt -> Washing machine -> PostWashingBuffer
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

    # Conveyor belt -> Washing machine -> PostWashingBuffer -> Hantering cell
    Hantering_cell = Machine(
        env, "Hantering cell",
        input_buffer=PostWashingBuffer,
        output_buffer=None,  # will be reassigned to hantering_out for splitting
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # After Hantering_cell, split evenly into PrePress1Buffer and PrePress2Buffer
    # Represent Hantering_cell output as an explicit store for splitting
    hantering_out = simpy.Store(env, capacity=6)
    Hantering_cell.output_buffer = hantering_out

    # Start splitter process for parallel presses
    env.process(splitter(env, hantering_out, PrePress1Buffer, PrePress2Buffer))

    # Hantering cell -> Presses cell 1
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

    # Parallel: Presses cell 2 || Presses cell 1
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

    # Merge parallel presses into common buffer PostPress1&Press2Buffer
    merger(env, press1_out, press2_out, PostPress12Buffer)

    # Presses cell 2 -> Quality station cell (take from merged flow)
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


class SimulationConfig:
    def __init__(self,
                 post_loading_cap,
                 post_conveyor_cap,
                 post_washing_cap,
                 pre_press1_cap,
                 pre_press2_cap,
                 post_press12_cap,
                 runs=5,
                 warmup=WARMUP_SECONDS,
                 sim_time=SIM_TIME,
                 seed_base=RANDOM_SEED):
        self.post_loading_cap = int(post_loading_cap)
        self.post_conveyor_cap = int(post_conveyor_cap)
        self.post_washing_cap = int(post_washing_cap)
        self.pre_press1_cap = int(pre_press1_cap)
        self.pre_press2_cap = int(pre_press2_cap)
        self.post_press12_cap = int(post_press12_cap)
        self.runs = runs
        self.warmup = warmup
        self.sim_time = sim_time
        self.seed_base = seed_base

    def to_buffer_dict(self):
        return {
            "post_loading_cap": self.post_loading_cap,
            "post_conveyor_cap": self.post_conveyor_cap,
            "post_washing_cap": self.post_washing_cap,
            "pre_press1_cap": self.pre_press1_cap,
            "pre_press2_cap": self.pre_press2_cap,
            "post_press12_cap": self.post_press12_cap,
        }


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def single_run(self, seed_offset):
        seed = self.config.seed_base + seed_offset
        res = run_simulation_with_capacities(
            seed=seed,
            post_loading_cap=self.config.post_loading_cap,
            post_conveyor_cap=self.config.post_conveyor_cap,
            post_washing_cap=self.config.post_washing_cap,
            pre_press1_cap=self.config.pre_press1_cap,
            pre_press2_cap=self.config.pre_press2_cap,
            post_press12_cap=self.config.post_press12_cap,
            warmup=self.config.warmup,
            measure_until=self.config.sim_time
        )
        return res

    def run_replications(self):
        throughputs = []
        wips = []
        for i in range(self.config.runs):
            res = self.single_run(i)
            throughputs.append(res["overall"]["throughput"])
            wips.append(res["overall"]["wip"])
        return self.aggregate_results(throughputs, wips)

    @staticmethod
    def aggregate_results(throughputs, wips):
        mean_throughput = statistics.mean(throughputs) if throughputs else 0.0
        mean_wip = statistics.mean(wips) if wips else 0.0
        return {"throughput": mean_throughput, "wip": mean_wip}

    def compute_objectives(self):
        agg = self.run_replications()
        # Objective 1: minimize WIP
        f1 = agg["wip"]
        # Objective 2: maximize throughput -> minimize negative throughput
        f2 = -agg["throughput"]
        return np.array([f1, f2], dtype=float)


class BufferCapacityProblem(Problem):
    def __init__(self,
                 runs_per_individual=5,
                 warmup=WARMUP_SECONDS,
                 sim_time=SIM_TIME,
                 seed_base=RANDOM_SEED,
                 history_recorder=None):
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=0,
            xl=np.array([1, 1, 1, 1, 1, 1]),
            xu=np.array([3, 3, 3, 3, 3, 3]),
            type_var=int
        )
        self.runs_per_individual = runs_per_individual
        self.warmup = warmup
        self.sim_time = sim_time
        self.seed_base = seed_base
        self.history_recorder = history_recorder

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            caps = X[i, :]
            cfg = SimulationConfig(
                post_loading_cap=caps[0],
                post_conveyor_cap=caps[1],
                post_washing_cap=caps[2],
                pre_press1_cap=caps[3],
                pre_press2_cap=caps[4],
                post_press12_cap=caps[5],
                runs=self.runs_per_individual,
                warmup=self.warmup,
                sim_time=self.sim_time,
                seed_base=self.seed_base
            )
            runner = SimulationRunner(cfg)
            objs = runner.compute_objectives()
            F[i, :] = objs
            if self.history_recorder is not None:
                wip = float(objs[0])
                throughput = -float(objs[1])
                self.history_recorder.record_individual(
                    capacities={
                        "PostLoadingBuffer": int(caps[0]),
                        "PostConveyorBuffer": int(caps[1]),
                        "PostWashingBuffer": int(caps[2]),
                        "PrePress1Buffer": int(caps[3]),
                        "PrePress2Buffer": int(caps[4]),
                        "PostPress12Buffer": int(caps[5]),
                    },
                    throughput=throughput,
                    wip=wip
                )
        out["F"] = F


class HistoryRecorder:
    def __init__(self):
        self.current_generation = 0
        self.records = []

    def start_generation(self, gen):
        self.current_generation = gen

    def record_individual(self, capacities, throughput, wip):
        rec = {
            "generation": self.current_generation,
            "PostLoadingBuffer": capacities["PostLoadingBuffer"],
            "PostConveyorBuffer": capacities["PostConveyorBuffer"],
            "PostWashingBuffer": capacities["PostWashingBuffer"],
            "PrePress1Buffer": capacities["PrePress1Buffer"],
            "PrePress2Buffer": capacities["PrePress2Buffer"],
            "PostPress12Buffer": capacities["PostPress12Buffer"],
            "throughput": throughput,
            "wip": wip
        }
        self.records.append(rec)

    def to_csv(self, filename="moo_simulation_results.csv"):
        if not self.records:
            return
        fieldnames = [
            "generation",
            "PostLoadingBuffer",
            "PostConveyorBuffer",
            "PostWashingBuffer",
            "PrePress1Buffer",
            "PrePress2Buffer",
            "PostPress12Buffer",
            "throughput",
            "wip"
        ]
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)


class GenerationPrinter(Callback):
    def __init__(self, history_recorder: HistoryRecorder):
        super().__init__()
        self.n_gen = 0
        self.history_recorder = history_recorder

    def notify(self, algorithm):
        self.n_gen = algorithm.n_gen
        self.history_recorder.start_generation(self.n_gen)
        print(f"Running generation {self.n_gen}")


def run_moo_optimization(
        runs_per_individual=5,
        n_generations=10,
        population_size=20,
        warmup=WARMUP_SECONDS,
        sim_time=SIM_TIME,
        seed_base=RANDOM_SEED):
    history_recorder = HistoryRecorder()

    problem = BufferCapacityProblem(
        runs_per_individual=runs_per_individual,
        warmup=warmup,
        sim_time=sim_time,
        seed_base=seed_base,
        history_recorder=history_recorder
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(eta=15, prob=0.9)
    mutation = PM(eta=20)
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    callback = GenerationPrinter(history_recorder)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        callback=callback,
        verbose=False
    )

    pareto_X = res.X
    pareto_F = res.F

    pareto_solutions = []
    for x, f in zip(pareto_X, pareto_F):
        capacities = {
            "PostLoadingBuffer": int(x[0]),
            "PostConveyorBuffer": int(x[1]),
            "PostWashingBuffer": int(x[2]),
            "PrePress1Buffer": int(x[3]),
            "PrePress2Buffer": int(x[4]),
            "PostPress12Buffer": int(x[5]),
        }
        wip = float(f[0])
        throughput = -float(f[1])
        pareto_solutions.append({
            "capacities": capacities,
            "throughput": throughput,
            "wip": wip
        })

    best_idx = int(np.argmax([s["throughput"] for s in pareto_solutions]))
    best_solution = pareto_solutions[best_idx]

    print("\nPareto-optimal solutions (capacities, throughput, wip):")
    for s in pareto_solutions:
        print(s)

    print("\nSelected preferred solution:")
    print(best_solution)

    history_recorder.to_csv("moo_simulation_results.csv")

    return pareto_solutions, best_solution


def main():
    pareto_solutions, best_solution = run_moo_optimization(
        runs_per_individual=3,
        n_generations=5,
        population_size=10
    )

    best_caps = best_solution["capacities"]

    print("\nRunning main simulation with best found capacities...")
    res = run_simulation_with_capacities(
        seed=RANDOM_SEED,
        post_loading_cap=best_caps["PostLoadingBuffer"],
        post_conveyor_cap=best_caps["PostConveyorBuffer"],
        post_washing_cap=best_caps["PostWashingBuffer"],
        pre_press1_cap=best_caps["PrePress1Buffer"],
        pre_press2_cap=best_caps["PrePress2Buffer"],
        post_press12_cap=best_caps["PostPress12Buffer"],
        warmup=WARMUP_SECONDS,
        measure_until=SIM_TIME
    )

    print("\nMain simulation KPIs with optimized capacities:")
    print(f"Throughput = {res['overall']['throughput']:.2f} parts/hour")
    print(f"WIP = {res['overall']['wip']:.2f} parts")
    print(f"Produced parts = {res['overall']['produced_parts']}")
    print("\nAll MOO solutions with KPIs have been saved to 'moo_simulation_results.csv'.")


if __name__ == "__main__":
    main()