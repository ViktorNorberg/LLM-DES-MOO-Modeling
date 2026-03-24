import simpy
import random
import statistics
from collections import Counter
 
RANDOM_SEED = 11

SIM_TIME = 691200          # 8 days
WARMUP_SECONDS = 86400     # 1 day
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
        stop_end   = SEC_PER_DAY
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
    return getattr(buf, "free_capacity", None) and buf.free_capacity() > 0 \
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
                 defect_rate = None, defect_sink = None, capacity=1):
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

    # Buffers (all helper buffers have defined capacity)
    raw_input = simpy.Store(env, capacity=1000)
    defect_sink = simpy.Store(env, capacity=100000)
    final_sink = simpy.Store(env, capacity=100000)

    # From Loading robot to Conveyor belt
    PostLoadingBuffer = DelayBuffer(env, cap=1, delay=10)
    # From Conveyor belt to Washing machine
    PostConveyorBuffer = DelayBuffer(env, cap=2, delay=10)
    # From Washing machine to Hantering cell
    PostWashingBuffer = DelayBuffer(env, cap=1, delay=10)

    # Parallel press buffers
    PrePress1Buffer = DelayBuffer(env, cap=1, delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=1, delay=32)
    # Helper buffer for splitter before individual pre-press buffers
    PrePressJoinBuffer = simpy.Store(env, capacity=2)  # matches PrePress1+PrePress2 cap
    # After both Presses -> shared buffer
    PostPress1_2Buffer = DelayBuffer(env, cap=1, delay=32)

    # Helper stores for routing in parallel section (from each press to merger)
    PostPress1Out = simpy.Store(env, capacity=1)
    PostPress2Out = simpy.Store(env, capacity=1)

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
 
if __name__ == "__main__":
    runs = 10
    overall_results = []
    machine_results = {}
    energy_per_part_list = []
 
    for i in range(runs):
        seed = RANDOM_SEED + i
        res = run_simulation(seed)
        overall_results.append(res["overall"])

        for mname, mdata in res["machine_energy"].items():
            machine_results.setdefault(mname, []).append(mdata)

        total_energy_run = sum(machine_results[mname][i]["total_energy"]
                               for mname in machine_results)
        total_energy_kwh = total_energy_run
        produced_parts = overall_results[i]["produced_parts"]
        energy_per_part_list.append(
            total_energy_kwh / produced_parts if produced_parts > 0 else 0
        )
 
    mean_energy_per_part = statistics.mean(energy_per_part_list)
    mean_overall = {
        "throughput": statistics.mean(o["throughput"] for o in overall_results),
        "wip": statistics.mean(o["wip"] for o in overall_results)
    }
 
    print(f"\n=== Mean Overall KPIs over {runs} runs ===")
    print(f"Throughput = {mean_overall['throughput']:.2f} parts/hour")
    print(f"WIP = {mean_overall['wip']:.2f} parts")
    print(f"Mean Energy Consumption per Part = {mean_energy_per_part:.4f} kWh/part")