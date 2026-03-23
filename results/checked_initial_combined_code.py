import simpy
import random
import statistics
from collections import Counter
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv

RANDOM_SEED = 11

SIM_TIME = 10000          # 8 days
WARMUP_SECONDS = 3600     # 1 day
MEASURE_UNTIL = SIM_TIME

# Global buffer capacities to be set by MOO
BUFFER_CAPS = {
    "PostLoadingBuffer": 2,
    "PostConveyorBuffer": 2,
    "PostWashingBuffer": 2,
    "PrePress1Buffer": 3,
    "PrePress2Buffer": 3,
    "PostPress12Buffer": 3,
}


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

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Use global BUFFER_CAPS for capacities
    caps = BUFFER_CAPS

    # Buffers (capacities from BUFFER_CAPS)
    PostLoadingBuffer = DelayBuffer(env, cap=caps["PostLoadingBuffer"], delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=caps["PostConveyorBuffer"], delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=caps["PostWashingBuffer"], delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=caps["PrePress1Buffer"], delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=caps["PrePress2Buffer"], delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=caps["PostPress12Buffer"], delay=32)

    # Helper buffer between parallel presses and common PostPress12Buffer
    press1_out = simpy.Store(env, capacity=caps["PostPress12Buffer"])
    press2_out = simpy.Store(env, capacity=caps["PostPress12Buffer"])

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

    hantering_out = simpy.Store(env, capacity=caps["PrePress1Buffer"] + caps["PrePress2Buffer"])
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


class SimulationModelAdapter:
    def __init__(self, warmup, measure_until, runs, base_seed=11):
        self.warmup = warmup
        self.measure_until = measure_until
        self.runs = runs
        self.base_seed = base_seed

    def run_with_config(self, caps):
        overall_results = []
        energy_per_part_list = []
        machine_results = {}
        bottleneck_results = []

        for i in range(self.runs):
            seed = self.base_seed + i
            global BUFFER_CAPS
            BUFFER_CAPS = caps.copy()

            res = run_simulation(seed, warmup=self.warmup, measure_until=self.measure_until)
            overall_results.append(res["overall"])
            for machine_name, _data in res["bottleneck"]["top_3"]:
                bottleneck_results.append(machine_name)
            for mname, mdata in res["machine_energy"].items():
                machine_results.setdefault(mname, []).append(mdata)
            total_energy_run = sum(machine_results[mname][i]["total_energy"] for mname in machine_results)
            total_energy_kwh = total_energy_run
            produced_parts = overall_results[i]["produced_parts"]
            energy_per_part_list.append(total_energy_kwh / produced_parts if produced_parts > 0 else 0)

        mean_overall = {
            "throughput": statistics.mean(o["throughput"] for o in overall_results),
            "wip": statistics.mean(o["wip"] for o in overall_results),
        }
        mean_energy_per_part = statistics.mean(energy_per_part_list) if energy_per_part_list else 0.0

        return {
            "throughput": mean_overall["throughput"],
            "wip": mean_overall["wip"],
            "energy_per_part": mean_energy_per_part,
            "bottleneck_counter": Counter(bottleneck_results)
        }


class SimulationRunner:
    def __init__(self, warmup, measure_until, runs=3, base_seed=11, parallel_evals=False):
        self.model = SimulationModelAdapter(warmup, measure_until, runs, base_seed)
        self.parallel_evals = parallel_evals

    def evaluate_config(self, x):
        caps = {
            "PostLoadingBuffer": int(x[0]),
            "PostConveyorBuffer": int(x[1]),
            "PostWashingBuffer": int(x[2]),
            "PrePress1Buffer": int(x[3]),
            "PrePress2Buffer": int(x[4]),
            "PostPress12Buffer": int(x[5]),
        }
        metrics = self.model.run_with_config(caps)
        return metrics


class ProductionLineMOOProblem(Problem):
    def __init__(self, warmup, measure_until, runs=3, base_seed=11, history=None):
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=0,
            xl=np.array([1] * 6),
            xu=np.array([3] * 6),
            type_var=int
        )
        self.runner = SimulationRunner(warmup, measure_until, runs, base_seed)
        self.history = history

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            metrics = self.runner.evaluate_config(x)
            throughput = metrics["throughput"]
            wip = metrics["wip"]
            # Objective 1: minimize WIP
            f1 = wip
            # Objective 2: maximize throughput -> minimize negative throughput
            f2 = -throughput
            F.append([f1, f2])

            if self.history is not None:
                self.history.append({
                    "PostLoadingBuffer": int(x[0]),
                    "PostConveyorBuffer": int(x[1]),
                    "PostWashingBuffer": int(x[2]),
                    "PrePress1Buffer": int(x[3]),
                    "PrePress2Buffer": int(x[4]),
                    "PostPress12Buffer": int(x[5]),
                    "throughput": throughput,
                    "wip": wip
                })

        out["F"] = np.array(F)


class MOOOptimizer:
    def __init__(self, warmup, measure_until, population_size=10, generations=5, runs_per_eval=3, base_seed=11):
        self.population_size = population_size
        self.generations = generations
        self.seeds = base_seed
        self.history = []
        self.problem = ProductionLineMOOProblem(
            warmup=warmup,
            measure_until=measure_until,
            runs=runs_per_eval,
            base_seed=base_seed,
            history=self.history
        )
        self.algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=IntegerRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=PolynomialMutation(eta=20),
            eliminate_duplicates=True
        )
        self.termination = get_termination("n_gen", self.generations)
        self.result = None

    def optimize(self):
        self.result = minimize(
            self.problem,
            self.algorithm,
            self.termination,
            seed=self.seeds,
            verbose=True
        )
        return self.result

    def get_pareto_front(self):
        if self.result is None:
            return None
        X = self.result.X
        F = self.result.F
        pareto_solutions = []
        for x, f in zip(X, F):
            sol = {
                "PostLoadingBuffer": int(x[0]),
                "PostConveyorBuffer": int(x[1]),
                "PostWashingBuffer": int(x[2]),
                "PrePress1Buffer": int(x[3]),
                "PrePress2Buffer": int(x[4]),
                "PostPress12Buffer": int(x[5]),
                # F[0] = wip, F[1] = -throughput
                "throughput": -float(f[1]),
                "wip": float(f[0]),
            }
            pareto_solutions.append(sol)
        return pareto_solutions

    def get_history(self):
        return self.history


if __name__ == "__main__":
    optimizer = MOOOptimizer(
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        population_size=10,
        generations=5,
        runs_per_eval=3,
        base_seed=11
    )

    optimizer.optimize()
    history = optimizer.get_history()

    pareto_front = optimizer.get_pareto_front()
    print("\n=== Pareto Front Solutions (Buffer Capacities, Throughput, WIP) ===")
    for i, sol in enumerate(pareto_front):
        print(f"Solution {i + 1}: "
              f"PostLoading={sol['PostLoadingBuffer']}, "
              f"PostConveyor={sol['PostConveyorBuffer']}, "
              f"PostWashing={sol['PostWashingBuffer']}, "
              f"PrePress1={sol['PrePress1Buffer']}, "
              f"PrePress2={sol['PrePress2Buffer']}, "
              f"PostPress12={sol['PostPress12Buffer']} | "
              f"Throughput={sol['throughput']:.3f}, WIP={sol['wip']:.3f}")

    # Write all evaluated solutions (history) to CSV
    csv_filename = "moo_simulation_results.csv"
    fieldnames = [
        "PostLoadingBuffer",
        "PostConveyorBuffer",
        "PostWashingBuffer",
        "PrePress1Buffer",
        "PrePress2Buffer",
        "PostPress12Buffer",
        "throughput",
        "wip"
    ]

    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)