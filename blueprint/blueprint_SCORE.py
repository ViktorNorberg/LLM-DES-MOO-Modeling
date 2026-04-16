import simpy
import random
import statistics
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import csv

# --- Configuration & Baselines (unchanged) ---
BASE_DATA = {
    "M1": {"avail": 90.49, "mttr": 74,  "proc": 5},
    "M2": {"avail": 100.0, "mttr": 100, "proc": 20},
    "M3_parallel": {"avail": 80.89, "mttr": 80, "proc": 15},
    "M4_parallel": {"avail": 97.79, "mttr": 80, "proc": 15},
    "M5": {"avail": 87.79, "mttr": 90, "proc": 25},
}
MACHINE_ORDER = ["M1", "M2", "M3_parallel", "M4_parallel", "M5"]
RANDOM_SEED = 42

def kwh_per_sec(kw):
    return kw / 3600.0

def run_simulation(seed, flags, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL):
    random.seed(seed)
    env = simpy.Environment()

    avail_flags = flags[0:5]
    mttr_flags = flags[5:10]
    proc_flags = flags[10:15]

    config = {}
    for i, name in enumerate(MACHINE_ORDER):
        base = BASE_DATA[name]
        # Logic: 10% increase/decrease based on flags
        new_avail = min(100.0, base["avail"] * (1.1 if avail_flags[i] == 1 else 1.0))
        new_mttr = base["mttr"] * (0.9 if mttr_flags[i] == 1 else 1.0)
        new_proc = base["proc"] * (0.9 if proc_flags[i] == 1 else 1.0)
        config[name] = {"avail": new_avail, "mttr": new_mttr, "proc": new_proc}

    # Note: Machine, DelayBuffer, merger, part_generator classes must be present in your environment
    # ... [SimPy Boilerplate Setup] ...
    
    pass # Placeholder for the full SimPy logic you already have

class CostBenefitProblem(Problem):
    def __init__(self, n_replications=5, base_seed=RANDOM_SEED):
        super().__init__(
            n_var=15, 
            n_obj=2,
            n_constr=1, # <--- Updated to 1 constraint
            xl=np.array([0]*15),
            xu=np.array([1]*15),
            elementwise_evaluation=False,
        )
        self.n_replications = n_replications
        self.base_seed = base_seed

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=int)
        F = np.zeros((X.shape[0], 2))
        G = np.zeros((X.shape[0], 1)) # <--- Array to store constraint values

        for i in range(X.shape[0]):
            flags = X[i, :]
            
            # Objective 1: Minimize number of changes (sum of flags)
            num_changes = np.sum(flags)
            
            # Constraint Formulation: g(x) <= 0
            # Condition: num_changes >= 1  =>  1 - num_changes <= 0
            G[i, 0] = 1 - num_changes
            
            # Objective 2: Maximize Throughput (via simulation)
            # Optimization: Skip simulation if constraint is violated (num_changes == 0)
            if num_changes > 0:
                throughputs = []
                for r in range(self.n_replications):
                    seed = self.base_seed + r + random.randint(0, 1000)
                    res = run_simulation(seed, flags)
                    throughputs.append(res["overall"]["throughput"] if res else 0)
                avg_throughput = statistics.mean(throughputs)
            else:
                avg_throughput = 0.0 # Penalize objective if infeasible

            F[i, 0] = num_changes             # Minimize flags
            F[i, 1] = -avg_throughput         # Maximize throughput

        out["F"] = F
        out["G"] = G # <--- Pass constraints back to pymoo

def export_history_to_csv(result, filename="moo_cost_benefit_results.csv"):
    fieldnames = ["gen", "ind"]
    for cat in ["Avail_Inc_10per", "MTTR_Red_10per", "Proc_Red_10per"]:
        for m in MACHINE_ORDER:
            fieldnames.append(f"{m}_{cat}")
    # Added 'feasible' column to help you track rejected solutions
    fieldnames.extend(["active_flags", "throughput", "feasible"]) 

    rows = []
    for gen_idx, algo in enumerate(result.history):
        X, F = algo.pop.get("X"), algo.pop.get("F")
        G = algo.pop.get("G") # Get the constraints
        
        for ind_idx in range(len(X)):
            x = X[ind_idx]
            f = F[ind_idx]
            
            # Check if individual is feasible (all g <= 0)
            is_feasible = bool(np.all(G[ind_idx] <= 0)) if G is not None else True
            
            row = {
                "gen": gen_idx, 
                "ind": ind_idx, 
                "active_flags": int(f[0]), 
                "throughput": float(-f[1]),
                "feasible": is_feasible
            }
            for i, val in enumerate(x):
                row[fieldnames[i+2]] = int(val)
            rows.append(row)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    problem = CostBenefitProblem(n_replications=3)
    
    algorithm = NSGA2(
        pop_size=50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0/15, eta=20),
        eliminate_duplicates=True
    )
    
    res = minimize(
        problem, 
        algorithm, 
        get_termination("n_gen", 20), 
        seed=1, 
        save_history=True, 
        verbose=True
    )
    
    export_history_to_csv(res)