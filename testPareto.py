from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("moo_simulation_results.csv")

WIP = df["WIP"].values
Throughput = df["Throughput"].values

plt.plot(WIP, Throughput, "o", label="Solutions")


# Identify the mask (True/False) for Pareto solutions
# sense=["min", "max"] tells it what to do with each column
mask = paretoset(df[["WIP", "Throughput"]], sense=["min", "max"])

pareto_solutions = df[mask]

print("Pareto-optimal solutions:")
print(pareto_solutions)


plt.show()