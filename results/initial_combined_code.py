import random
import statistics
from collections import Counter
import simpy
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import pandas as pd

RANDOM_SEED = 11
SIM_TIME = 86400          # 8 days
WARMUP_SECONDS = 3600   # 1 day
MEASURE_UNTIL = SIM_TIME

def production_wait_time(now: float) -> float:
    SEC_PER_DAY = 86400
    day = int((now // SEC_PER_DAY) % 7)
    time_of_day = now % SEC_PER_DAY
    def secs(h, m=0, s=0):
        return h * 3600 + m * 60 + s
    if day == 4:
        stop1_start = secs(17)
        stop1_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        return 0.0
    if day == 5:
        stop1_start = secs(0)
        stop1_end = secs(7)
        stop2_start = secs(17)
        stop2_end = secs(24)
        if stop1_start <= time_of_day < stop1_end:
            return max(0.0, stop1_end - time_of_day)
        if stop2_start <= time_of_day < stop2_end:
            return max(0.0, stop2_end - time_of_day)
        return 0.0
    if day == 6:
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

def run_simulation(buffer_caps, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    PostLoadingBuffer = DelayBuffer(env, cap=buffer_caps[0], delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=buffer_caps[1], delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=buffer_caps[2], delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=buffer_caps[3], delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=buffer_caps[4], delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=buffer_caps[5], delay=32)

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

    delay_buffers = [
        PostLoadingBuffer,
        PostConveyorBuffer,
        PostWashingBuffer,
        PrePress1Buffer,
        PrePress2Buffer,
        PostPress12Buffer,
    ]

    env.process(part_generator(env, raw_input))

    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

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

    return throughput, avg_wip

class SimulationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=2,
                         n_constr=0,
                         xl=1,
                         xu=10)

    def _evaluate(self, x, out, *args, **kwargs):
        results = np.array([run_simulation(xi) for xi in x])
        out["F"] = -results

algorithm = NSGA2(pop_size=10)

problem = SimulationProblem()
res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=True)

optimal_solutions = res.X
optimal_throughput_wip = -res.F

results_df = pd.DataFrame(optimal_solutions, columns=[f'Buffer_{i+1}' for i in range(6)])
results_df['Throughput'] = optimal_throughput_wip[:,0]
results_df['WIP'] = optimal_throughput_wip[:,1]

results_df.to_csv('moo_simulation_results.csv', index=False)