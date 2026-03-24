from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import numpy as np
import random
import statistics

# ---------------------------------------------------------------------------
# Assumed to be imported from the existing simulation module:
# from your_simulation_module import run_simulation, RANDOM_SEED
# Here we assume run_simulation(seed, warmup, measure_until) is available.
# ---------------------------------------------------------------------------

# ---------------------- SystemModelBuffers ---------------------------------


class SystemModelBuffers:
    """
    Defines the mapping and bounds for the tunable buffer capacities.
    Buffers:
        0: PostLoadingBuffer
        1: PostConveyorBuffer
        2: PostWashingBuffer
        3: PrePress1Buffer
        4: PrePress2Buffer
        5: PostPress1_2Buffer
    Capacities are integers in [1, 5].
    """

    BUFFER_NAMES = [
        "PostLoadingBuffer",
        "PostConveyorBuffer",
        "PostWashingBuffer",
        "PrePress1Buffer",
        "PrePress2Buffer",
        "PostPress1_2Buffer",
    ]

    LOWER_BOUND = 1
    UPPER_BOUND = 5

    @classmethod
    def n_buffers(cls):
        return len(cls.BUFFER_NAMES)

    @classmethod
    def bounds(cls):
        n = cls.n_buffers()
        xl = np.full(n, cls.LOWER_BOUND, dtype=float)
        xu = np.full(n, cls.UPPER_BOUND, dtype=float)
        return xl, xu


# ---------------------- CandidateSolution ----------------------------------


class CandidateSolution:
    """
    Encodes a candidate solution as a vector of buffer capacities.
    """

    def __init__(self, buffer_caps):
        # buffer_caps: iterable of ints in [1,5]
        self.buffer_caps = np.array(buffer_caps, dtype=int)

    def encode(self):
        # Return as float array for pymoo
        return self.buffer_caps.astype(float)

    @classmethod
    def decode(cls, x):
        # x: numpy array of floats -> round and clip to [1,5]
        x_int = np.rint(x).astype(int)
        x_int = np.clip(
            x_int, SystemModelBuffers.LOWER_BOUND, SystemModelBuffers.UPPER_BOUND
        )
        return cls(x_int)

    def apply_to_kwargs(self, kwargs):
        """
        Prepare keyword arguments for the simulator to use these capacities.
        This assumes the simulation's run_simulation function can accept
        a 'buffer_caps' argument as a dict or list.
        """
        kwargs = dict(kwargs) if kwargs is not None else {}
        kwargs["buffer_caps"] = {
            name: int(cap)
            for name, cap in zip(SystemModelBuffers.BUFFER_NAMES, self.buffer_caps)
        }
        return kwargs


# ---------------------- RunSimulationFunction / SimulatorInterface ---------


class RunSimulationFunction:
    """
    Wrapper around the original run_simulation.
    """

    def __init__(self, warmup=None, measure_until=None, base_seed=None):
        self.warmup = warmup
        self.measure_until = measure_until
        self.base_seed = RANDOM_SEED if base_seed is None else base_seed

    def __call__(self, buffer_caps, seed_offset=0):
        """
        buffer_caps: CandidateSolution
        seed_offset: int
        """
        seed = self.base_seed + seed_offset
        # The original run_simulation does not accept buffer capacities.
        # To integrate, you must modify run_simulation to accept buffer_caps
        # and apply them when creating DelayBuffer instances.
        # Here we assume such an interface exists:
        # run_simulation(seed, warmup=self.warmup, measure_until=self.measure_until,
        #                buffer_caps={...})
        kwargs = {}
        kwargs["warmup"] = self.warmup if self.warmup is not None else WARMUP_SECONDS
        kwargs["measure_until"] = (
            self.measure_until if self.measure_until is not None else MEASURE_UNTIL
        )
        kwargs["seed"] = seed

        # Apply buffer capacities
        kwargs["buffer_caps"] = {
            name: int(cap)
            for name, cap in zip(
                SystemModelBuffers.BUFFER_NAMES, buffer_caps.buffer_caps
            )
        }

        # The following assumes you have updated run_simulation signature to:
        # run_simulation(seed, warmup=WARMUP_SECONDS, measure_until=MEASURE_UNTIL,
        #                buffer_caps=None)
        result = run_simulation(
            kwargs["seed"],
            warmup=kwargs["warmup"],
            measure_until=kwargs["measure_until"],
            buffer_caps=kwargs["buffer_caps"],
        )
        return result


class SimulatorInterface:
    """
    Adapter for the original simulation code.
    """

    def __init__(self, run_sim_func):
        self.run_sim = run_sim_func

    def run(self, candidate, seed_offset=0):
        return self.run_sim(candidate, seed_offset=seed_offset)


# ---------------------- MetricsCollector / ObjectiveEvaluator --------------


class MetricsCollector:
    """
    Collects throughput, wip, produced_parts and machine energy per run.
    """

    @staticmethod
    def collect(sim_result):
        overall = sim_result["overall"]
        machine_energy = sim_result["machine_energy"]
        throughput = overall["throughput"]
        wip = overall["wip"]
        produced_parts = overall["produced_parts"]
        total_energy = sum(m["total_energy"] for m in machine_energy.values())
        return {
            "throughput": throughput,
            "wip": wip,
            "produced_parts": produced_parts,
            "total_energy": total_energy,
        }


class ObjectiveEvaluator:
    """
    Computes objective vector:
        f1 = WIP (to minimize)
        f2 = -Throughput (to minimize, since NSGA2 is minimization-based)
    """

    def __init__(self, simulator, n_replications=1):
        self.simulator = simulator
        self.n_replications = n_replications

    def evaluate(self, candidate):
        wip_list = []
        thr_list = []

        for r in range(self.n_replications):
            sim_res = self.simulator.run(candidate, seed_offset=r)
            metrics = MetricsCollector.collect(sim_res)
            wip_list.append(metrics["wip"])
            thr_list.append(metrics["throughput"])

        mean_wip = statistics.mean(wip_list)
        mean_thr = statistics.mean(thr_list)

        f1 = mean_wip
        f2 = -mean_thr
        return np.array([f1, f2], dtype=float)


# ---------------------- ParetoArchive --------------------------------------


class ParetoArchive:
    """
    Stores non-dominated CandidateSolution objects and their objective values.
    """

    def __init__(self):
        self.solutions = []  # list of (CandidateSolution, f)

    def update(self, candidate, f):
        new_solutions = []
        dominated = False
        for c, fv in self.solutions:
            if self._dominates(fv, f):
                dominated = True
                break
            if not self._dominates(f, fv):
                new_solutions.append((c, fv))
        if not dominated:
            new_solutions.append((candidate, f))
        self.solutions = new_solutions

    @staticmethod
    def _dominates(f1, f2):
        return np.all(f1 <= f2) and np.any(f1 < f2)


# ---------------------- PopulationManager / Selection / Variation ----------


class SelectionOperator:
    """
    Performs parent selection using tournament selection (delegated to pymoo).
    """

    def __init__(self, selection):
        self.selection = selection

    def select(self, pop, n_parents):
        return self.selection.do(pop, n_parents)


class VariationOperator:
    """
    Applies crossover and mutation, ensuring capacities remain in {1,...,5}.
    """

    def __init__(self, crossover, mutation):
        self.crossover = crossover
        self.mutation = mutation

    def vary(self, problem, pop):
        off = self.crossover.do(problem, pop)
        off = self.mutation.do(problem, off)
        # Ensure integer and bounds
        for ind in off:
            x = ind.X
            x = np.rint(x).astype(int)
            x = np.clip(
                x, SystemModelBuffers.LOWER_BOUND, SystemModelBuffers.UPPER_BOUND
            )
            ind.X = x.astype(float)
        return off


class PopulationManager:
    """
    Manages population initialization and evaluation.
    """

    def __init__(self, problem, evaluator):
        self.problem = problem
        self.evaluator = evaluator

    def initialize(self, pop_size):
        n_var = SystemModelBuffers.n_buffers()
        pop = []
        for _ in range(pop_size):
            caps = np.random.randint(
                SystemModelBuffers.LOWER_BOUND,
                SystemModelBuffers.UPPER_BOUND + 1,
                size=n_var,
            )
            cand = CandidateSolution(caps)
            x = cand.encode()
            f = self.evaluator.evaluate(cand)
            pop.append({"X": x, "F": f})
        return pop

    def evaluate_population(self, pop):
        for ind in pop:
            x = ind["X"]
            cand = CandidateSolution.decode(x)
            f = self.evaluator.evaluate(cand)
            ind["F"] = f
        return pop


# ---------------------- Scheduler (simple, sequential) ---------------------


class Scheduler:
    """
    Dispatches simulation jobs (sequential in this simple implementation).
    """

    def __init__(self, simulator):
        self.simulator = simulator

    def run_candidate(self, candidate, seed_offset=0):
        return self.simulator.run(candidate, seed_offset=seed_offset)


# ---------------------- MOO Problem for pymoo ------------------------------


class BufferOptimizationProblem(ElementwiseProblem):
    """
    Pymoo problem wrapper around the simulation-based objective evaluator.
    """

    def __init__(self, evaluator):
        xl, xu = SystemModelBuffers.bounds()
        super().__init__(n_var=SystemModelBuffers.n_buffers(), n_obj=2, xl=xl, xu=xu)
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        cand = CandidateSolution.decode(x)
        f = self.evaluator.evaluate(cand)
        out["F"] = f


# ---------------------- Callback for ParetoArchive -------------------------


class ParetoCallback(Callback):
    def __init__(self, archive):
        super().__init__()
        self.archive = archive

    def notify(self, algorithm):
        pop = algorithm.pop
        for ind in pop:
            x = ind.X
            f = ind.F
            cand = CandidateSolution.decode(x)
            self.archive.update(cand, f)


# ---------------------- UserInterface (minimal stub) -----------------------


class UserInterface:
    """
    Minimal stub for setting constraints and visualizing results.
    """

    def __init__(self):
        self.budget = None
        self.stop_criteria = None

    def set_budget(self, n_evals):
        self.budget = n_evals

    def set_stop_criteria(self, generations):
        self.stop_criteria = generations

    def show_pareto_front(self, archive):
        print("Pareto Front (WIP, -Throughput, capacities):")
        for cand, f in archive.solutions:
            print(f"F={f}, caps={cand.buffer_caps}")


# ---------------------- MOO_Controller -------------------------------------


class MOO_Controller:
    """
    Orchestrates the NSGA-II optimization of buffer capacities.
    """

    def __init__(
        self,
        pop_size=20,
        n_gen=5,
        n_replications=1,
        warmup=None,
        measure_until=None,
        base_seed=None,
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen

        self.run_sim_func = RunSimulationFunction(
            warmup=warmup, measure_until=measure_until, base_seed=base_seed
        )
        self.simulator = SimulatorInterface(self.run_sim_func)
        self.scheduler = Scheduler(self.simulator)
        self.evaluator = ObjectiveEvaluator(self.simulator, n_replications=n_replications)
        self.problem = BufferOptimizationProblem(self.evaluator)

        self.archive = ParetoArchive()
        self.ui = UserInterface()

        # NSGA-II operators
        self.selection = TournamentSelection(func_comp=None)
        self.crossover = SBX(prob=0.9, eta=15)
        self.mutation = PM(prob=None, eta=20)

        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=self._sampling,
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            eliminate_duplicates=True,
        )

    def _sampling(self, problem, n_samples, **kwargs):
        xl, xu = problem.xl, problem.xu
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var))
        for i in range(n_samples):
            caps = np.random.randint(
                SystemModelBuffers.LOWER_BOUND,
                SystemModelBuffers.UPPER_BOUND + 1,
                size=n_var,
            )
            X[i, :] = caps.astype(float)
        return X

    def run(self):
        termination = get_termination("n_gen", self.n_gen)
        callback = ParetoCallback(self.archive)

        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            callback=callback,
            verbose=True,
        )

        self.ui.show_pareto_front(self.archive)
        return res, self.archive


# ---------------------- Example main integration ---------------------------

if __name__ == "__main__":
    # Example of running the MOO controller.
    controller = MOO_Controller(
        pop_size=20,
        n_gen=5,
        n_replications=1,
        warmup=WARMUP_SECONDS,
        measure_until=MEASURE_UNTIL,
        base_seed=RANDOM_SEED,
    )
    result, archive = controller.run()