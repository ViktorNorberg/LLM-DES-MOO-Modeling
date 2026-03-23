from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import numpy as np
import statistics
from collections import Counter
import random


# ---------------- Simulation Adapter and Runner ---------------- #

class SimulationModelAdapter:
    """
    Adapter around the existing run_simulation function.
    It assumes that run_simulation reads buffer capacities from its own
    DelayBuffer instantiations. To integrate buffer capacities as decision
    variables, we wrap run_simulation in a function that accepts capacities
    and passes them into a modified run_simulation_with_caps.
    """

    def __init__(self, warmup, measure_until, runs, base_seed=11):
        self.warmup = warmup
        self.measure_until = measure_until
        self.runs = runs
        self.base_seed = base_seed

    def run_with_config(self, caps):
        """
        caps: dict with keys:
            PostLoadingBuffer, PostConveyorBuffer, PostWashingBuffer,
            PrePress1Buffer, PrePress2Buffer, PostPress12Buffer
        Returns aggregated metrics over self.runs replications.
        """
        from __main__ import run_simulation  # use existing function

        overall_results = []
        energy_per_part_list = []
        machine_results = {}
        bottleneck_results = []

        for i in range(self.runs):
            seed = self.base_seed + i
            # run_simulation currently does not accept capacities.
            # To integrate capacities, the user should modify run_simulation
            # to read global variables or a configuration object.
            # Here we assume run_simulation has been adapted externally
            # to use the capacities stored in a global dict BUFFER_CAPS.
            global BUFFER_CAPS
            BUFFER_CAPS = caps

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
        """
        x: array-like of 6 integers in [1,3]
        Order: [PostLoadingBuffer, PostConveyorBuffer, PostWashingBuffer,
                PrePress1Buffer, PrePress2Buffer, PostPress12Buffer]
        """
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


# ---------------- Problem Definition for pymoo ---------------- #

class ProductionLineMOOProblem(Problem):
    """
    Multi-objective optimization problem for the production line.
    Decision variables:
        6 integer buffer capacities in [1,3]
    Objectives:
        f1 = -throughput (to maximize throughput)
        f2 = wip (to minimize WIP)
    """

    def __init__(self, warmup, measure_until, runs=3, base_seed=11):
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=0,
            xl=np.array([1] * 6),
            xu=np.array([3] * 6),
            type_var=int
        )
        self.runner = SimulationRunner(warmup, measure_until, runs, base_seed)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            metrics = self.runner.evaluate_config(x)
            throughput = metrics["throughput"]
            wip = metrics["wip"]
            f1 = -throughput
            f2 = wip
            F.append([f1, f2])
        out["F"] = np.array(F)


# ---------------- NSGA-II Optimizer Wrapper ---------------- #

class MOOOptimizer:
    def __init__(self, warmup, measure_until, population_size=10, generations=5, runs_per_eval=3, base_seed=11):
        self.population_size = population_size
        self.generations = generations
        self.seeds = base_seed
        self.problem = ProductionLineMOOProblem(
            warmup=warmup,
            measure_until=measure_until,
            runs=runs_per_eval,
            base_seed=base_seed
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
                "throughput": -float(f[0]),
                "wip": float(f[1]),
            }
            pareto_solutions.append(sol)
        return pareto_solutions


# ---------------- Example Usage Integration ---------------- #

if __name__ == "__main__":
    # Import SIM_TIME and WARMUP_SECONDS from the existing simulation module if needed.
    from __main__ import WARMUP_SECONDS, MEASURE_UNTIL

    optimizer = MOOOptimizer(
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        population_size=10,
        generations=5,
        runs_per_eval=3,
        base_seed=11
    )

    res = optimizer.optimize()
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