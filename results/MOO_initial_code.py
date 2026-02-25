import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

class SimulationProblem:
    def __init__(self):
        self.buffer_sizes = [2, 2, 2, 3, 3, 3]  # Default buffer sizes
    
    def run_simulation(self, buffer_sizes):
        # Replace the buffer sizes in the main simulation code with values from buffer_sizes
        PostLoadingBuffer = DelayBuffer(self.env, cap=buffer_sizes[0], delay=10)
        PostConveyorBuffer = DelayBuffer(self.env, cap=buffer_sizes[1], delay=10)
        PostWashingBuffer = DelayBuffer(self.env, cap=buffer_sizes[2], delay=10)
        PrePress1Buffer = DelayBuffer(self.env, cap=buffer_sizes[3], delay=32)
        PrePress2Buffer = DelayBuffer(self.env, cap=buffer_sizes[4], delay=32)
        PostPress12Buffer = DelayBuffer(self.env, cap=buffer_sizes[5], delay=32)

        # The rest of the setup and running the simulation remains the same
        
        # After running simulation gather objectives
        return throughput, wip  # Collect and return required outputs

    def evaluate(self, x):
        buffer_sizes = x.astype(int)
        throughput, wip = self.run_simulation(buffer_sizes)
        return -throughput, wip  # Return negative throughput to maximize it

problem = SimulationProblem()

algorithm = NSGA2()
res = minimize(problem, algorithm, ('n_gen', 100), verbose=True)

Scatter().add(res.F).show()