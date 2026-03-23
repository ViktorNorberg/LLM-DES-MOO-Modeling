from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import numpy as np
import multiprocessing as mp
import statistics
from collections import Counter
import random

# -------------------------------------------------------------------------
# Import or paste the existing simulation code here, or ensure it is
# available on the PYTHONPATH so that run_simulation can be imported.
#
# from your_simulation_module import run_simulation, RANDOM_SEED
#
# For direct integration, place this MOO code in the same file as the
# simulation code, below the run_simulation definition.
# -------------------------------------------------------------------------

# Names of the DelayBuffers in the simulation that we want to optimize.
# These must match the variable names used in run_simulation.
BUFFER_NAMES = [
    "PostLoadingBuffer",
    "PostConveyorBuffer",
    "PostWashingBuffer",
    "PrePress1Buffer",
    "PrePress2Buffer",
    "PostPress12Buffer",
    "PostHanteringBuffer",
]

# Helper to run one simulation with a given buffer configuration.
# We assume that run_simulation can be modified or wrapped to accept
# buffer capacities as an argument. To keep compatibility with the
# given code, we implement a wrapper that redefines run_simulation
# internally with the chosen capacities.


def run_simulation_with_caps(seed, caps,
                             warmup=WARMUP_SECONDS,
                             measure_until=MEASURE_UNTIL):
    """
    Wrapper around the original run_simulation that applies
    the given buffer capacities to all DelayBuffer instances.
    caps: list/array of ints, len == len(BUFFER_NAMES)
    """
    random.seed(seed)
    env = simpy.Environment()

    # Buffers (raw + normal) with capacities as specified
    raw_input = simpy.Store(env, capacity=1000)

    # Map decision variables to capacities (1-10) and keep original delays
    cap_dict = {name: int(c) for name, c in zip(BUFFER_NAMES, caps)}

    PostLoadingBuffer = DelayBuffer(env, cap=cap_dict["PostLoadingBuffer"], delay=10)
    PostConveyorBuffer = DelayBuffer(env, cap=cap_dict["PostConveyorBuffer"], delay=10)
    PostWashingBuffer = DelayBuffer(env, cap=cap_dict["PostWashingBuffer"], delay=10)
    PrePress1Buffer = DelayBuffer(env, cap=cap_dict["PrePress1Buffer"], delay=32)
    PrePress2Buffer = DelayBuffer(env, cap=cap_dict["PrePress2Buffer"], delay=32)
    PostPress12Buffer = DelayBuffer(env, cap=cap_dict["PostPress12Buffer"], delay=32)

    # Sinks
    sink = simpy.Store(env, capacity=100000)
    defects = simpy.Store(env, capacity=100000)

    # Helper buffers for splitting/merging press cells (explicit, with capacities)
    Press1_out_helper = simpy.Store(env, capacity=3)
    Press2_out_helper = simpy.Store(env, capacity=3)

    # Machines according to table and routing:
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
        output_buffer=None,  # will be set to PostHanteringBuffer
        process_time=25.0,
        availability=97.79,
        mttr=74.0,
        working_power=kwh_per_sec(0.74),
        waiting_power=kwh_per_sec(0.50),
    )

    # Split from a dedicated post-Hantering buffer so we can evenly split
    PostHanteringBuffer = DelayBuffer(env, cap=cap_dict["PostHanteringBuffer"], delay=0)
    Hantering_cell.output_buffer = PostHanteringBuffer

    # Splitter for parallel presses
    env.process(splitter(env, PostHanteringBuffer, PrePress1Buffer, PrePress2Buffer))

    Presses_cell_1 = Machine(
        env, "Presses cell 1",
        input_buffer=PrePress1Buffer,
        output_buffer=Press1_out_helper,
        process_time=175.0,
        availability=87.79,
        mttr=73.0,
        working_power=kwh_per_sec(1.28),
        waiting_power=kwh_per_sec(1.25),
    )

    Presses_cell_2 = Machine(
        env, "Presses cell 2",
        input_buffer=PrePress2Buffer,
        output_buffer=Press2_out_helper,
        process_time=176.0,
        availability=87.69,
        mttr=74.0,
        working_power=kwh_per_sec(1.27),
        waiting_power=kwh_per_sec(1.25),
    )

    # Merge outputs of two press cells into common post-press buffer
    merger(env, Press1_out_helper, Press2_out_helper, PostPress12Buffer)

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
        PostHanteringBuffer,
    ]

    env.process(part_generator(env, raw_input))

    env.run(until=warmup)

    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)
    wip_samples = []

    def sample_wip(env_):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(
                b.in_transit_count() for b in delay_buffers if hasattr(b, "in_transit_count")
            )
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env_.timeout(60)

    env.process(sample_wip(env))

    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    result = {
        "overall": {
            "throughput": throughput,
            "wip": avg_wip,
            "produced_parts": total_produced
        },
        "machine_energy": {}
    }

    for m in machines_list:
        waiting_energy = m.waiting_energy_consumption()
        working_energy = m.working_energy_consumption()
        total_energy = waiting_energy + working_energy
        result["machine_energy"][m.name] = {
            "working_time": m.working_time,
            "waiting_time": m.failed_time_total + m.blocked_time,
            "working_energy": working_energy,
            "waiting_energy": waiting_energy,
            "total_energy": total_energy
        }

    bottleneck_data = {}
    for m in machines_list:
        m_th = m.processed_count / hours if hours > 0 else 0.0
        util = (m.working_time / (measure_until - warmup)) * 100.0 if (measure_until > warmup) else 0.0
        bottleneck_data[m.name] = {
            "throughput": m_th,
            "utilization": util,
            "processed_count": m.processed_count
        }

    result["bottleneck"] = {
        "top_3": sorted(
            bottleneck_data.items(),
            key=lambda kv: kv[1]["utilization"],
            reverse=True
        )[:3],
        "all": bottleneck_data
    }

    return result


def evaluate_individual(x, base_seed=RANDOM_SEED, n_replications=3):
    """
    Evaluate one individual x (buffer capacities).
    Returns objectives: [f1, f2] where
    f1 = average WIP (to minimize)
    f2 = -average throughput (since pymoo minimizes)
    Also returns a dict with additional KPIs for logging.
    """
    caps = np.clip(np.round(x).astype(int), 1, 10)
    throughputs = []
    wips = []
    energies = []
    bottlenecks = []

    for r in range(n_replications):
        seed = base_seed + r
        res = run_simulation_with_caps(seed, caps)
        throughputs.append(res["overall"]["throughput"])
        wips.append(res["overall"]["wip"])

        total_energy_run = sum(m["total_energy"] for m in res["machine_energy"].values())
        energies.append(total_energy_run)

        for mname, _data in res["bottleneck"]["top_3"]:
            bottlenecks.append(mname)

    avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
    avg_wip = statistics.mean(wips) if wips else 0.0
    avg_energy = statistics.mean(energies) if energies else 0.0
    bottleneck_counter = Counter(bottlenecks)

    f1 = avg_wip
    f2 = -avg_throughput

    info = {
        "caps": caps.tolist(),
        "avg_throughput": avg_throughput,
        "avg_wip": avg_wip,
        "avg_energy": avg_energy,
        "bottleneck_counter": dict(bottleneck_counter)
    }
    return np.array([f1, f2], dtype=float), info


class ProductionLineProblem(ElementwiseProblem):
    def __init__(self, n_var=len(BUFFER_NAMES), xl=1, xu=10):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=np.full(n_var, xl),
            xu=np.full(n_var, xu),
            elementwise_evaluation=True
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f, info = evaluate_individual(x)
        out["F"] = f
        out["info"] = info


class LoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["F"] = []
        self.data["X"] = []
        self.data["info"] = []

    def notify(self, algorithm):
        self.data["F"].append(algorithm.pop.get("F"))
        self.data["X"].append(algorithm.pop.get("X"))
        self.data["info"].append(algorithm.pop.get("info"))


def run_nsga2_optimization(pop_size=50, n_gen=5, seed=RANDOM_SEED, n_processes=1):
    problem = ProductionLineProblem()

    sampling = IntegerRandomSampling()
    crossover = TwoPointCrossover()
    mutation = PolynomialMutation(eta=20, prob=1.0 / problem.n_var)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    callback = LoggingCallback()

    if n_processes > 1:
        pool = mp.Pool(processes=n_processes)
        from pymoo.core.problem import StarmapParallelization
        runner = StarmapParallelization(pool.starmap)
        problem.elementwise_runner = runner
    else:
        pool = None

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=True,
        callback=callback
    )

    if pool is not None:
        pool.close()
        pool.join()

    X = res.X
    F = res.F
    infos = res.opt.get("info")

    pareto_solutions = []
    for x, f, info in zip(X, F, infos):
        sol = {
            "buffer_caps": np.clip(np.round(x).astype(int), 1, 10).tolist(),
            "wip": float(f[0]),
            "throughput": float(-f[1]),
            "avg_energy": float(info["avg_energy"]),
            "bottleneck_counter": info["bottleneck_counter"],
        }
        pareto_solutions.append(sol)

    return pareto_solutions, res, callback


if __name__ == "__main__":
    pareto_solutions, res, callback = run_nsga2_optimization(
        pop_size=50,
        n_gen=5,
        seed=RANDOM_SEED,
        n_processes=1  # increase for parallel evaluation if desired
    )

    with open("moo_pareto_solutions.txt", "w") as f:
        for i, sol in enumerate(pareto_solutions):
            f.write(f"Solution {i}:\n")
            f.write(f"  Buffer capacities (PostLoading, PostConv, PostWash, PreP1, PreP2, PostP12, PostHant): {sol['buffer_caps']}\n")
            f.write(f"  Throughput: {sol['throughput']:.4f} parts/hour\n")
            f.write(f"  WIP: {sol['wip']:.4f} parts\n")
            f.write(f"  Avg Energy: {sol['avg_energy']:.4f} kWh\n")
            f.write(f"  Bottlenecks: {sol['bottleneck_counter']}\n")
            f.write("\n")