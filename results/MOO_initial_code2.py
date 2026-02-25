from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np

# Define the simulation optimization problem
class SimulationProblem(Problem):
    
    def __init__(self):
        super().__init__(n_var=6, 
                         n_obj=2, 
                         n_constr=0, 
                         xl=1, 
                         xu=10)
        
    def _evaluate(self, x, out, *args, **kwargs):
        # Here we will replace delay buffers capacities with input variables x
        def run_modified_simulation(x):
            buffer_caps = x.astype(int)

            # Configure the simulation environment with new buffer sizes
            def run_simulation_with_buffers(buffer_caps):
                env = simpy.Environment()

                raw_input = simpy.Store(env, capacity=1000)
                sink = simpy.Store(env, capacity=100000)
                defects = simpy.Store(env, capacity=100000)

                # Buffers with dynamically assigned capacities
                PostLoadingBuffer = DelayBuffer(env, cap=buffer_caps[0], delay=10)
                PostConveyorBuffer = DelayBuffer(env, cap=buffer_caps[1], delay=10)
                PostWashingBuffer = DelayBuffer(env, cap=buffer_caps[2], delay=10)
                PrePress1Buffer = DelayBuffer(env, cap=buffer_caps[3], delay=32)
                PrePress2Buffer = DelayBuffer(env, cap=buffer_caps[4], delay=32)
                PostPress12Buffer = DelayBuffer(env, cap=buffer_caps[5], delay=32)
                
                # Define the rest of the machines and processes as per the original script
                # Use the same definitions and processes as the original code
                # ...

                def part_generator(env, output_buffer):
                    part_id = 0
                    while True:
                        part = {"id": part_id}
                        yield output_buffer.put(part)
                        part_id += 1
                        yield env.timeout(1)

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

            # Call the simulation with specific buffer caps
            return run_simulation_with_buffers(buffer_caps)

        results = np.array([run_modified_simulation(xi) for xi in x])
        out["F"] = -results  # Maximize throughput, Minimize WIP

# Define the MOO algorithm
algorithm = NSGA2(pop_size=100)

# Solve the problem
problem = SimulationProblem()
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True)

# Process results
optimal_solutions = res.X
optimal_throughput_wip = res.F