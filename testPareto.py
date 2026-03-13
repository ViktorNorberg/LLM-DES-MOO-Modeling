from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import pandas as pd

"""
selected_objectives = ["wip", "throughput"]
directions = ["min", "max"]



def _find_pareto_front(selected_objectives, directions):

    df = pd.read_csv("moo_simulation_results.csv")

    mask = paretoset(df[[selected_objectives[0], selected_objectives[1]]], sense=[directions[0], directions[1]])

    pareto_solutions = df[mask]

    return pareto_solutions

pareto_solutions = _find_pareto_front(selected_objectives, directions)

print(pareto_solutions)

MOO_pareto_csv_path = Path("moo_pareto_csv.csv")
pareto_solutions.to_csv(MOO_pareto_csv_path, index=False)

with open(MOO_pareto_csv_path, "r") as file:
    pareto_solutions_string = file.read()



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
#print(pareto_solutions.to_string(index=False))


#plt.show()

df = pd.read_csv("moo_simulation_results.csv")
df2 = pd.read_csv("moo_pareto_solutions.csv")

WIP = df["wip"].values
Throughput = df["throughput"].values

WIP2 = df2["wip"].values
Throughput2 = df2["throughput"].values

#plot both the original solutions and the pareto solutions


plt.plot(WIP, Throughput, "o", label="Solutions")
plt.plot(WIP2, Throughput2, "s", label="Pareto Solutions")
plt.xlabel("WIP")
plt.ylabel("Throughput")
plt.legend()
plt.show()




# Load your data
df = pd.read_csv("moo_simulation_results.csv")
# Add a column to distinguish the types if you want them on the same plot
df['Type'] = 'Standard Solution'

# If you want to include the Pareto solutions in the same interactive plot:
df2 = pd.read_csv("moo_pareto_solutions.csv")
df2['Type'] = 'Pareto Optimal'

# Combine them
combined_df = pd.concat([df, df2], ignore_index=True)

# Create the interactive scatter plot
fig = px.scatter(
    combined_df, 
    x="wip", 
    y="throughput", 
    color="Type",
    # Add all the columns you want to see when hovering
    hover_data=[
        "generation", 
        "PostLoadingBuffer", 
        "PostConveyorBuffer", 
        "PostWashingBuffer", 
        "PrePress1Buffer", 
        "PrePress2Buffer", 
        "PostPress12Buffer"
    ],
    title="Interactive MOO Simulation Results",
    labels={"wip": "Work In Progress (WIP)", "throughput": "Throughput (parts/hr)"},
    template="plotly_white"
)

# Show the plot
fig.show()

"""


def visualize_MOO_results(selected_objectives):
   
    # 1. Load the data
    df = pd.read_csv("moo_simulation_results.csv")
    df2 = pd.read_csv("moo_pareto_solutions.csv")

    # 2. Extract the column names from the CSV before adding "Type"
    # This captures everything: WIP, Throughput, and all Buffers
    hover_cols = list(df.columns)

    # 3. Add the "Type" column for legend/coloring
    df['Type'] = 'Standard Solution'
    df2['Type'] = 'Pareto Optimal'

    df3 = pd.read_csv("suggested_improvements.csv")
    df3['Type'] = 'LLM chosen points'

    # 4. Combine them
    combined_df = pd.concat([df, df2, df3], ignore_index=True)

    # 5. Create the interactive scatter plot
    fig = px.scatter(
        combined_df, 
        x=selected_objectives[0], 
        y=selected_objectives[1], 
        color="Type",
        hover_data=hover_cols,  # Explicitly includes every column from the CSV
        title="Interactive MOO Simulation Results",
        labels={
            selected_objectives[0]: selected_objectives[0].replace('_', ' ').title(),
            selected_objectives[1]: selected_objectives[1].replace('_', ' ').title()
        },
        template="plotly_white"
    )
    
    # Optional: Make markers slightly larger for easier hovering
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkGrey')))
    
    fig.show()



def adder(a,b):
    x = a + b
    visualize_MOO_results(selected_objectives=["wip", "throughput"])
    return x


x = adder(2,3)

print(x)

