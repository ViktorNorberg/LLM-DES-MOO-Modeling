import simpy
import random
import statistics
from collections import Counter

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv


# Baseline availabilities from the given simulation model
BASE_AVAILABILITIES = {
    "M1": 90.49,
    "M2": 100.0,
    "M3_parallel": 80.89,
    "M4_parallel": 97.79,
    "M5": 87.79,
}

MACHINE_ORDER = [
    "M1",
    "M2",
    "M3_parallel",
    "M4_parallel",
    "M5",
]



def run_simulation(seed, flags, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    # Machine topology example including both serial and parallel section:
    #   M1  ->  M2  ->  [ M3 || M4 ]  ->  M5
    #
    # Serial segments:
    #   raw_input -> M1 -> buffer1 -> M2 -> buffer2
    #
    # Parallel segment (Pattern: splitter + separate pre/post buffers, depending on machine count):
    #   splitter(buffer2, preM3buffer, preM4buffer)
    #   preM3buffer -> M3 -> postM3buffer
    #   preM4buffer -> M4 -> postM4buffer
    #   add buffers if needed for more parallel machines
    #   merger(postM3buffer, postM4buffer) -> buffer3
    #
    # final serial segment:
    #   buffer3 -> M5 -> sink

   # Compute new availabilities and check constraint (no availability > 100)
    new_avail = {}
    for name, flag in zip(MACHINE_ORDER, flags):
        base = BASE_AVAILABILITIES[name]
        if flag == 1:
            val = base + 5.0
        else:
            val = base
        if val > 100.0:
            return None
        new_avail[name] = val
    
    

    #Buffers between serial machines
    buffer1 = DelayBuffer(env, cap=2, delay=10)  # between M1 and M2
    buffer2 = DelayBuffer(env, cap=2, delay=10)  # between M2 and parallel section
    buffer3 = DelayBuffer(env, cap=2, delay=10)  # between parallel section and M5

    # Raw input and sinks
    raw_input = simpy.Store(env, capacity=1000) # large to avoid starvation
    sink = simpy.Store(env, capacity=100000)   # final sink
    defects = simpy.Store(env, capacity=100000)  # defect sink

    # Helper stores for routing in parallel section
    # - Helper stores are simple simpy.Store, not DelayBuffer.
    # - They can be used:
    #     - as outputs of parallel machines before a merger, or
    #     - as temporary queues for splitters/mergers.
    # The capacity of helper stores can not be changed
    branch1_out = simpy.Store(env, capacity=2)  # output of M3
    branch2_out = simpy.Store(env, capacity=2)  # output of M4

    M1 = Machine(env, "M1", input_buffer=raw_input, output_buffer=buffer1,
        process_time=5, availability=new_avail["M1"], mttr=74,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M2 = Machine(env, "M2", input_buffer=buffer1, output_buffer=buffer2,
        process_time=20, availability=new_avail["M2"], mttr=100,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M3_parallel = Machine(env, "M3parallel", input_buffer=buffer2, output_buffer=branch1_out,
        process_time=15, availability=new_avail["M3_parallel"], mttr=80,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    M4_parallel = Machine(env, "M4parallel", input_buffer=buffer2, output_buffer=branch2_out,
        process_time=15, availability=new_avail["M4_parallel"], mttr=80,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
    )

    # Merge outputs of parallel machines into buffer3.
    merger(env, branch1_out, branch2_out, buffer3)

    M5 = Machine(env, "M5", input_buffer=buffer3, output_buffer=sink,
        process_time=25, availability=new_avail["M5"], mttr=90,
        working_power=kwh_per_sec(1.28), waiting_power=kwh_per_sec(1.25),
        defect_rate=0.089, defect_sink=defects,
    )

    machines_list = [M1, M2, M3_parallel, M4_parallel, M5]

    # Start part generation.
    env.process(part_generator(env, raw_input))

    # Run the model to fill pipelines/buffers and reach steady-state
    env.run(until=warmup)

    # Zero machine counters so everything after is measured stats
    for m in machines_list:
        reset_machine_stats(m)

    # Zero sinks for measured production counts
    produced_count_before = len(sink.items)

    wip_samples = []
    delay_buffers = [buffer1, buffer2, buffer3]

    def sample_wip(env):
        while True:
            # WIP definition: items in delay buffers + items in process
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
            "produced_parts":total_produced},
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

class AvailabilityIncreaseProblem(Problem):

    """
    Multi-objective optimization problem for machine availability increases.
    Decision variables (binary flags, 0 or 1):
        x[0] = M1 availability increase 5% flag
        x[1] = M2 availability increase 5% flag
        x[2] = M3_parallel availability increase 5% flag
        x[3] = M4_parallel availability increase 5% flag
        x[4] = M5 availability increase 5% flag
    
    Objectives:
        f1 = average WIP (to be minimized)
        f2 = -average throughput (negative because pymoo minimizes)
    """

    def __init__(
        self,
        n_var=5,
        n_obj=2,
        n_constr=0,
        xl=None,
        xu=None,
        n_replications=5,
        base_seed=RANDOM_SEED,
    ):
        if xl is None:
            xl = np.array([0] * n_var, dtype=int)
        if xu is None:
            xu = np.array([1] * n_var, dtype=int)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
        )

        self.n_replications = n_replications
        self.base_seed = base_seed

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a 2D array of shape (n_individuals, n_var)
        X = np.asarray(X)
        n_individuals = X.shape[0]

        F = np.zeros((n_individuals, 2), dtype=float)
        

        for i in range(n_individuals):
            flags= X[i, :]

            throughputs = []
            wips = []

            for r in range(self.n_replications):
                seed = self.base_seed + r + random.randint(0, 1000000)
                res = run_simulation(seed, flags, WARMUP_SECONDS, MEASURE_UNTIL)
                throughputs.append(res["overall"]["throughput"])
                wips.append(res["overall"]["wip"])

            avg_throughput = statistics.mean(throughputs)
            avg_wip = statistics.mean(wips)

            F[i, 0] = avg_wip
            F[i, 1] = -avg_throughput

        out["F"] = F

def run_nsga2_optimization(
    pop_size=50,
    n_gen=6,
    n_replications=5,
    base_seed=RANDOM_SEED,
    verbose=True
):
    """
    Returns:
        res: pymoo result object containing the Pareto front and decision variables.
    """

    problem = AvailabilityIncreaseProblem(
        n_var=5,
        n_obj=2,
        xl=np.array([0]*5),
        xu=np.array([1]*5),
        n_replications=n_replications,
        base_seed=base_seed
    )

    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(prob=1.0 / problem.n_var, eta=20)

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
        save_history=True,
        verbose=verbose
    )

    return res

def export_history_to_csv(result, filename="moo_simulation_results.csv"):
    """
    Export all feasible solutions from every generation (including initial population)
    with their KPIs and decision variables to a CSV file.

    If the MOO algortihm is optimizing which machine to increase availability for,
    the column names for the decision variables in the csv should be like "washing_machine_availability_increase5%_flag", "press1_availability_increase5%_flag" etc
    the column name must include information on how much the availability is increased for each machine.
    """



    fieldnames = [
        "gen",
        "ind",
        "M1_availability_increase5percent_flag",
        "M2_availability_increase5percent_flag",
        "M3_availability_increase5percent_flag",
        "M4_availability_increase5percent_flag",
        "M5_availability_increase5percent_flag",
        "wip",
        "throughput"
    ]

    rows = []
    history = result.history

    for gen_idx, algo in enumerate(history):
        pop = algo.pop
        X = pop.get("X")
        F = pop.get("F")

        for ind_idx, (x, f) in enumerate(zip(X, F)):
            wip = float(f[0])
            throughput = float(-f[1])  # stored as -throughput in objectives
            flags = [int(v) for v in x[:5]]
            row = {
                "gen": gen_idx,
                "ind": ind_idx,
                "M1_availability_increase5percent_flag": flags[0],
                "M2_availability_increase5percent_flag": flags[1],
                "M3_availability_increase5percent_flag": flags[2],
                "M4_availability_increase5percent_flag": flags[3],
                "M5_availability_increase5percent_flag": flags[4],
                "wip": wip,
                "throughput": throughput
            }
            rows.append(row)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run NSGA-II optimization on the simulation model
    result = run_nsga2_optimization(pop_size=50, n_gen=6, n_replications=5, verbose=True)

    # Export all solutions from every generation to CSV
    export_history_to_csv(result, filename="moo_simulation_results.csv")
    