
import os
from pathlib import Path
from helpers.runner import run_python_code
from agents.optimizer import Modeloptimizer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def test_objective_selection():
    possible_metrics = ["WIP", "Throughput", "Energy Consumption"]
    selected_objectives = []
    directions = []

    print("\n--- Objective Selection ---")
    print("You need to select 2 objectives to perform Multi-Objective Optimization.")

    for i in range(1, 3):  # Runs twice: once for i=1, once for i=2
        while True:  # Inner loop to handle invalid input for this specific slot
            print(f"\nSelect Objective #{i}:")
            for idx, metric in enumerate(possible_metrics, 1):
                print(f"{idx}. {metric}")
            
            choice = input(f"Enter number (1-{len(possible_metrics)}): ")
            
            try:
                m_idx = int(choice) - 1
                if 0 <= m_idx < len(possible_metrics):
                    metric_name = possible_metrics[m_idx]
                    
                    # Check if they already picked this
                    if metric_name in selected_objectives:
                        print(f"❌ '{metric_name}' is already selected. Please pick a different objective.")
                        continue
                    
                    # Ask for Direction
                    print(f"How should we optimize '{metric_name}'?")
                    print("1. Minimize")
                    print("2. Maximize")
                    dir_choice = input("Choice (1 or 2): ")
                    direction = "min" if dir_choice == "1" else "max"
                    
                    # Store results
                    selected_objectives.append(metric_name)
                    directions.append(direction)
                    
                    print(f"✅ Slot {i} set to: {direction.upper()} {metric_name}")
                    break  # Exit inner while loop, move to next 'i' in for loop
                else:
                    print("❌ Invalid selection. Out of range.")
            except ValueError:
                print("❌ Please enter a numerical value.")

    print(f"\nFinal configuration: {selected_objectives[0]} ({directions[0]}) vs {selected_objectives[1]} ({directions[1]})")
    return selected_objectives, directions


selected_objectives, directions = test_objective_selection()
objectives = [f"{dir.upper()} {obj}" for dir, obj in zip(directions, selected_objectives)]
print(f"Selected Objectives: {selected_objectives} with directions {directions}")
print(f"objectives: {objectives}")


"""
optimizer = Modeloptimizer(client)




_ = input("code has been generated and checked. Inspect the code in 'results/checked_initial_combined_code.py' and make sure it looks good. If you are satisfied with the code, press enter to run the code and get the optimization results. ")

# Define your test inputs
code_path = os.path.join("results", "checked_initial_combined_code.py")
# 2. Open and read the file
with open(code_path, "r") as file:
    clean_checked_combined_code = file.read()


_ = run_python_code(clean_checked_combined_code)
    
csv_path = Path("moo_simulation_results.csv")
with open(csv_path, "r") as file:
    MOO_pareto_csv = file.read()



user_input = input("What are your current priorities for the production system? (e.g. prioritize high throughput, minimize energy consumption, etc.) ")
suggestions =optimizer._suggest_improvements(clean_checked_combined_code, user_input, MOO_pareto_csv)
print(suggestions)

"""