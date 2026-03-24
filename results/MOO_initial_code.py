from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import numpy as np
import random

# Assumes the simulation code (including run_simulation) is imported or present in the same module.


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
        This function assumes that run_simulation can be modified to accept
        buffer capacities, or that a wrapper around run_simulation is used.
        Here we show a wrapper pattern that redefines run_simulation_with_caps.
        """
        # capacities is a dict mapping buffer names to capacity values
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
    Returns the pymoo result object containing the Pareto front and decision variables.
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
        save_history=False,
        verbose=True
    )

    return res


if __name__ == "__main__":
    # Example of running the optimization and printing Pareto-optimal solutions
    result = run_nsga2_optimization(
        population_size=20,
        n_generations=5,
        runs_per_eval=3
    )

    X = result.X
    F = result.F

    print("\nPareto-optimal buffer configurations and objectives:")
    for i in range(len(X)):
        x = X[i]
        f = F[i]
        capacities = {
            "PostLoadingBuffer": int(x[0]),
            "PostConveyorBuffer": int(x[1]),
            "PostWashingBuffer": int(x[2]),
            "PrePress1Buffer": int(x[3]),
            "PrePress2Buffer": int(x[4]),
            "PostPress12Buffer": int(x[5]),
        }
        wip = f[0]
        throughput = -f[1]
        print(f"Solution {i + 1}: capacities={capacities}, WIP={wip:.2f}, Throughput={throughput:.4f} parts/hour")