import numpy as np
import random
import statistics
from collections import Counter
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBXCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ----------------------------------------------------------------------
# Wrapper around the provided run_simulation to allow variable buffers
# ----------------------------------------------------------------------

def run_simulation_with_buffers(seed,
                                cap_PostLoadingBuffer,
                                cap_PostConveyorBuffer,
                                cap_PostWashingBuffer,
                                cap_PrePress1Buffer,
                                cap_PrePress2Buffer,
                                cap_PostPress12Buffer,
                                warmup=WARMUP_SECONDS,
                                measure_until=MEASURE_UNTIL):
    """
    Re-implements run_simulation but exposes buffer capacities as parameters.
    This is a copy of the original run_simulation with buffer caps replaced
    by the given arguments.
    """
    random.seed(seed)
    env = simpy.Environment()

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Buffers (capacities driven by decision variables)
    PostLoadingBuffer = DelayBuffer(env, cap=int(cap_PostLoadingBuffer), delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=int(cap_PostConveyorBuffer), delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=int(cap_PostWashingBuffer), delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=int(cap_PrePress1Buffer), delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=int(cap_PrePress2Buffer), delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=int(cap_PostPress12Buffer), delay=32)

    # Helper buffer between parallel presses and common PostPress12Buffer
    press1_out = simpy.Store(env, capacity=3)
    press2_out = simpy.Store(env, capacity=3)

    # Machines following given routing and station table
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

    def sample_wip(env_):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env_.timeout(60)

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


# ----------------------------------------------------------------------
# Multi-objective optimization problem definition using pymoo
# ----------------------------------------------------------------------

class BufferCapacityOptimizationProblem(Problem):
    def __init__(self,
                 n_var=6,
                 n_obj=2,
                 xl=1,
                 xu=10,
                 n_int=6,
                 base_seed=RANDOM_SEED,
                 n_replications=3,
                 warmup=WARMUP_SECONDS,
                 measure_until=MEASURE_UNTIL):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.full(n_var, xl),
                         xu=np.full(n_var, xu),
                         type_var=int)
        self.base_seed = base_seed
        self.n_replications = n_replications
        self.warmup = warmup
        self.measure_until = measure_until

    def _evaluate(self, X, out, *args, **kwargs):
        # X is population of candidate solutions, shape (n_pop, n_var)
        n_pop = X.shape[0]
        F = np.zeros((n_pop, self.n_obj))

        for i in range(n_pop):
            caps = X[i, :].astype(int)
            cap_PostLoadingBuffer = caps[0]
            cap_PostConveyorBuffer = caps[1]
            cap_PostWashingBuffer = caps[2]
            cap_PrePress1Buffer = caps[3]
            cap_PrePress2Buffer = caps[4]
            cap_PostPress12Buffer = caps[5]

            throughputs = []
            wips = []

            for r in range(self.n_replications):
                seed = self.base_seed + r + i * 10000
                res = run_simulation_with_buffers(
                    seed,
                    cap_PostLoadingBuffer,
                    cap_PostConveyorBuffer,
                    cap_PostWashingBuffer,
                    cap_PrePress1Buffer,
                    cap_PrePress2Buffer,
                    cap_PostPress12Buffer,
                    warmup=self.warmup,
                    measure_until=self.measure_until
                )
                tp = res["overall"]["throughput"]
                wip = res["overall"]["wip"]
                throughputs.append(tp)
                wips.append(wip)

            mean_tp = statistics.mean(throughputs) if throughputs else 0.0
            mean_wip = statistics.mean(wips) if wips else 0.0

            # Multi-objective: maximize throughput, minimize WIP
            # pymoo minimizes objectives, so negate throughput
            F[i, 0] = -mean_tp
            F[i, 1] = mean_wip

        out["F"] = F


# ----------------------------------------------------------------------
# Main MOO optimization routine
# ----------------------------------------------------------------------

def optimize_buffers(
        pop_size=20,
        n_gen=10,
        n_replications=3,
        base_seed=RANDOM_SEED,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL):
    """
    Run NSGA-II on buffer capacities:
    Decision vars: 6 integer capacities in [1,10]
    Obj1: maximize throughput (negated in evaluation)
    Obj2: minimize WIP
    """

    problem = BufferCapacityOptimizationProblem(
        n_var=6,
        n_obj=2,
        xl=1,
        xu=10,
        n_int=6,
        base_seed=base_seed,
        n_replications=n_replications,
        warmup=warmup,
        measure_until=measure_until
    )

    sampling = IntegerRandomSampling()
    crossover = SBXCrossover(eta=15, prob=0.9)
    mutation = PM(eta=20, prob=None)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=base_seed,
        verbose=True
    )

    # Extract Pareto-optimal buffer configurations and objective values
    pareto_X = res.X.astype(int)
    pareto_F = res.F

    # Convert back first objective to throughput (positive) for reporting
    pareto_solutions = []
    for x, f in zip(pareto_X, pareto_F):
        throughput = -f[0]
        wip = f[1]
        sol = {
            "PostLoadingBuffer": int(x[0]),
            "PostConveyorBuffer": int(x[1]),
            "PostWashingBuffer": int(x[2]),
            "PrePress1Buffer": int(x[3]),
            "PrePress2Buffer": int(x[4]),
            "PostPress12Buffer": int(x[5]),
            "throughput": float(throughput),
            "wip": float(wip)
        }
        pareto_solutions.append(sol)

    # Simple example of selecting one preferred trade-off:
    # here we pick the solution with max throughput / wip ratio
    best_idx = None
    best_score = -np.inf
    for i, sol in enumerate(pareto_solutions):
        if sol["wip"] > 0:
            score = sol["throughput"] / sol["wip"]
        else:
            score = sol["throughput"]
        if score > best_score:
            best_score = score
            best_idx = i

    selected_solution = pareto_solutions[best_idx] if pareto_solutions else None

    return {
        "pareto_solutions": pareto_solutions,
        "selected_solution": selected_solution,
        "res": res
    }


# ----------------------------------------------------------------------
# Example usage and integration entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Parameters for MOO
    POP_SIZE = 10          # increase for better search
    N_GEN = 5              # increase for deeper search
    N_REP = 2              # replications per candidate (increase for robustness)

    moo_result = optimize_buffers(
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        n_replications=N_REP,
        base_seed=RANDOM_SEED,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL
    )

    print("\n=== Pareto-Optimal Buffer Configurations ===")
    for i, sol in enumerate(moo_result["pareto_solutions"]):
        print(f"Solution {i+1}: Caps = "
              f"[PostLoad={sol['PostLoadingBuffer']}, "
              f"PostConv={sol['PostConveyorBuffer']}, "
              f"PostWash={sol['PostWashingBuffer']}, "
              f"PreP1={sol['PrePress1Buffer']}, "
              f"PreP2={sol['PrePress2Buffer']}, "
              f"PostP12={sol['PostPress12Buffer']}], "
              f"Throughput = {sol['throughput']:.3f} parts/hour, "
              f"WIP = {sol['wip']:.3f}")

    if moo_result["selected_solution"] is not None:
        s = moo_result["selected_solution"]
        print("\n=== Selected Trade-off Solution ===")
        print("Buffer capacities:")
        print(f"  PostLoadingBuffer  = {s['PostLoadingBuffer']}")
        print(f"  PostConveyorBuffer = {s['PostConveyorBuffer']}")
        print(f"  PostWashingBuffer  = {s['PostWashingBuffer']}")
        print(f"  PrePress1Buffer    = {s['PrePress1Buffer']}")
        print(f"  PrePress2Buffer    = {s['PrePress2Buffer']}")
        print(f"  PostPress12Buffer  = {s['PostPress12Buffer']}")
        print(f"Performance:")
        print(f"  Throughput = {s['throughput']:.3f} parts/hour")
        print(f"  WIP        = {s['wip']:.3f} parts")

        # Example on how to integrate:
        # You can now plug these capacities into your production run_simulation_with_buffers
        # or update default capacities in your existing model.