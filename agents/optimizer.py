import os

from pathlib import Path
from openai import OpenAI
import json
from helpers.other_helpers import remove_code_wrappers, remove_code_wrappers, save_model
from helpers.mermaid_renderer import render_mermaid_to_png
from helpers.runner import run_python_code
from paretoset import paretoset
from agents.visualizer import Modelvisualizer
from agents.inspector import Modelinspector
import pandas as pd
import plotly.express as px
import time


class Blueprintoptimizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def optimize(self, model_code, MOO_blueprint_buffer, MOO_blueprint_availability):
        print("\nOptimizer activated:")
        path = Path("results")
        
        
        # Let the user choose the two objectives for the MOO algorithm and their corresponding optimization directions
        selected_objectives, directions = self._choose_objectives()
        objectives = f" {selected_objectives[0]} ({directions[0]}) and {selected_objectives[1]} ({directions[1]})"
        
        
        # Let the user specify the input variables for the MOO algorithm, their ranges, and whether they are discrete or continuous
        decision_variables, choice = self.choose_decision_variables()

        # Let the user choose the MOO algorithm to use, the population size, and the number of generations for the MOO algorithm
        population_size, generations, SIM_TIME, WARMUP_SECONDS = self.choose_hyper_parameters()
              
        #Generate MOO code
        print("\nGenerating MOO code...")

        if choice == "1":
            MOO_blueprint = MOO_blueprint_buffer
        else:
            MOO_blueprint = MOO_blueprint_availability

        MOO_code = self._generate_code(model_code, MOO_blueprint, objectives, decision_variables, population_size, generations)
        clean_initial_MOO_code = remove_code_wrappers(MOO_code)
        save_model(clean_initial_MOO_code, path, "MOO_initial_code.py")

        #Combine the MOO code with the simulation code
        print("\nCombining the MOO code with the simulation code...")
        combined_code = self._combiner(model_code, MOO_code, selected_objectives, SIM_TIME, WARMUP_SECONDS)
        clean_initial_combined_model = remove_code_wrappers(combined_code)
        save_model(clean_initial_combined_model, path, "initial_combined_code.py")

        #Generate UML diagram for the MOO algorithm
        print("\nGenerating a vizualization of the MOO algorithm...")
        visualizer = Modelvisualizer(self.client)
        UML_diagram = visualizer._generate_MOO_UML(clean_initial_combined_model, objectives, decision_variables)
        self.save_UML(UML_diagram, path)

        #repair and run the code
        self.repair_and_run_code(clean_initial_combined_model, path, objectives, decision_variables, self.client)
        

        #Extract Pareto-optimal solutions from the MOO results
        pareto_solutions = self._find_pareto_front(selected_objectives, directions)
        print("\nSee the results of the MOO algorithm in 'moo_simulation_results.csv'")
        print("And the pareto optimal solutions here: 'moo_pareto_solutions.csv'")
   
        #Ask the user for their priorities and suggest improvements based on the Pareto-optimal solutions
        print("")
        user_input = input("What are your current priorities for the production system? (e.g. prioritize high throughput, minimize energy consumption, etc.) ")

        #Generate suggestions for improvements based on the Pareto-optimal solutions and the user's priorities
        print("\nGenerating suggestions for improvements based on the Pareto-optimal solutions and user input...")
        suggestions = self._suggest_improvements(model_code, user_input, pareto_solutions, selected_objectives)

        #visualize the results
        self.json_to_csv(suggestions)
        self.visualize_MOO_results(selected_objectives)

        print("")
        print("Suggested changes based on the MOO algorithm")
        print(suggestions)
        print("\n\n")

        explanation = self._explain_suggestions(suggestions, model_code)

        print(explanation)
        print("")

        return suggestions
    

    def _explain_suggestions(self, suggestions, model_code,
        model = "gpt-5-mini"):
        prompt = (
            "You are an AI assistant that explains suggestions for improving a production system. "
            "The suggestions are based on the results of a multi-objective optimization (MOO) analysis, and are provided in a json format. "
            "Please explain the reasoning behind these suggestions in a way that is understandable for a human production manager. "
            f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
            f"Here are the suggestions in a json format:\n\n {suggestions}\n"
            "Please provide a clear, concise and short explanation for each suggestion, focusing on how it will improve the production system based on the MOO results and the user's priorities."
            "Do not end you answer with a question. "
        )
        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]) 
        return resp.choices[0].message.content
                             



    def _suggest_improvements(self, model_code, user_input, pareto_solutions, selected_objectives,
        model = "gpt-5-mini",
        response_format={"type": "json_object"}):
        prompt = (
                "You are an AI assistant that suggests improvements to a production system based on the results of a multi-objective optimization (MOO) analysis. "
                "The improvements should be based on the results of a MOO analysis, which are provided in a csv format. "
                "Choose three datapoints on the provided pareto front and suggest specific, implementable instructions to improve the system based on those datapoints. "
                f"When choosing the datapoints, consider the these instructions from the perspective of a production manager: \n {user_input}\n"
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"Here are the results of the MOO analysis:\n\n {pareto_solutions}\n"
                "Only answer with the instructions in a json format."
                "The instructions should only contain the input variable settings and corresponding objectives values, no explanations"
                f"The objectives are: {selected_objectives}"
                "The json format should be like this, though the variable and objective names and values should be chosen from the MOO results: { 'instructions': [ {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 1, 'PostPress12Buffer': 1, 'throughput': 28.114285714285717, 'wip': 10.779661016949152}, {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 3, 'PostPress12Buffer': 1, 'throughput': 30.17142857142857, 'wip': 12.836158192090396}] }"
                f"Make sure that the variable names in the instructions are the same as the ones in the \n\n {pareto_solutions} \n"
                "Make sure to only output the json object and nothing else. No explanations, no markdown fences."
                )
        
        resp = self.client.chat.completions.create(
            model=model,
            response_format = response_format, 
            messages=[{"role": "user", "content": prompt}])
        try:
            operator_output = resp.choices[0].message.content.strip()
            instructions_json = json.loads(operator_output)
            return instructions_json
        except Exception as e:
            raise
    

    def _generate_code(self, model_code, MOO_blueprint, objectives, decision_variables, population_size, generations,
            model ="gpt-5.1"):
            prompt = (
                 "You are an AI assistant that generates code for a multi-objective optimization (MOO) algorithm"
                 "Your task is to generate an MOO algorithm that optimizes a production line simulation model in Python."
                 f"This is the Python simulation model:\n\n```python\n {model_code}\n```\n\n"
                 f"These are the target objectives of the MOO algorithm: {objectives}\n"
                 f"These are the decision variables of the MOO algorithm that can be adjusted: {decision_variables}\n"
                 f"If a point violates the constraint, dont evaluate its objective values, and the datapoint should not be in the results csv file"
                 f"The MOO algortihm should have a population size of {population_size} and {generations} generations. "
                 "Please modify the following blueprint MOO code according to your instructions."
                 f"```python\n{MOO_blueprint}\n```\n\n"
                 "Only output the MOO code, no explanations, no markdown fences, don't include the simulation code"
                 "Make sure that the MOO code is compatible with the existing simulation code, and that it can be easily integrated"
                 
            )
            resp = self.client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.2)
            try:                
                return resp.choices[0].message.content
            except Exception as e:                
                raise e     
            
    
    def _combiner(self, model_code, MOO_code, selected_objectives, SIM_TIME, WARMUP_SECONDS,
        model = "gpt-5.1"):
        prompt = (
            "You are an AI assistant that combines python code"
            "The goal is to combine the existing simulation code with the MOO algorithm code. "
            "The combined code should be a single, working Python file that integrates both the simulation and the MOO algorithm. "
            "Do not output any explanations or markdown fences. "
            "Only output the combined Python code."
            f"Here is the existing simulation code:\n\n```python\n {model_code}\n```\n\n"
            f"Here is the MOO algorithm code:\n\n```python\n {MOO_code}\n```\n\n"
            "Make sure that the combined code is properly integrated, with the MOO algorithm being called in the right place, and that all necessary imports and dependencies are included. "
            f"The simulation will run for {SIM_TIME} seconds with a warmup period of {WARMUP_SECONDS} seconds."
            "When the combined code is run the MOO algorithm should optimize the simulation code and output all results as a table of the different solutions found by the MOO algorithm, with their corresponding KPI values."
            "The final combined python code should return a csv file with all the different solutions from every generation found by the MOO algorithm, except the one that violate any constraint"
            f"Make sure that the csv file contain KPI values for the selected objectives: {selected_objectives}, their column names should be the same as the objective names. "
            "Make sure the file name of the csv file is exactly: 'moo_simulation_results.csv'. and that it has self explanatory column names. ")
        
        resp= self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2)
        
        return resp.choices[0].message.content



    def _choose_objectives(self):
        """
        Allows the user to select from three predefined optimization scenarios.
        Returns a tuple of (selected_objectives, directions).
        """
        print("\n" + "="*30)
        print("-OPTIMIZATION INITIALIZED-")
        print("---OBJECTIVE SELECTION---")
        print("="*30)
        print("Please choose one of the following optimization scenarios:")
        print("1. WIP (Min) vs. Throughput (Max)")
        print("2. WIP (Min) vs. Energy Consumption per part (Min)")
        print("3. Throughput (Max) vs. Energy Consumption per part (Min)")

        # Dictionary to map choices to the required lists
        scenarios = {
            "1": (["wip", "throughput"], ["min", "max"]),
            "2": (["wip", "energy consumption per part"], ["min", "min"]),
            "3": (["throughput", "energy consumption per part"], ["max", "min"])
        }

        while True:
            choice = input("\nEnter scenario number (1, 2, or 3): ").strip()

            if choice in scenarios:
                selected_objectives, directions = scenarios[choice]
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        # Final confirmation output
        obj1, obj2 = selected_objectives
        dir1, dir2 = directions
        print(f"\n[CONFIRMED] Scenario {choice} loaded:")
        print(f" -> Objective 1: {obj1.upper()} ({dir1.upper()})")
        print(f" -> Objective 2: {obj2.upper()} ({dir2.upper()})")
        print("-" * 30 + "\n")

        return selected_objectives, directions
    

    def choose_decision_variables(self):

        print("\n" + "="*32)
        print("---DECISION VARIABLES SELECTION---")
        print("="*32)
        print("Please choose one of the following approaches to optimizing your selected objectives:")
        print("1. Find the best buffer configuration")
        print("2. Decrease machine process time by som chosen percentage ")
        print("3. Increase machine availability by some chosen percentage ")
        print("4. Decrease machine MTTR by some chosen percentage ")

        # Dictionary to map choices to the required lists
        scenarios = {
            "1": ("All buffer capacities. Discrete values on the range "),
            "2": ("All machine process times"), 
            "3": ("All machine availabilities "),
            "4": ("All machine MTTR")
        }

        choice = None 

        while True:
                choice = input("\nEnter scenario number (1, 2, 3 or 4): ").strip()

                if choice == "1":
                    _range = input("\nEnter the input range of the buffer capacities (e.g. 1-10)")
                    constraint = input("\nOPTIONAL: add a contraint of maximum total buffer capacity (e.g. 30), leave blank if none: ")
                    if constraint:
                        constraint = " The total buffer capacity of all buffers is constrained to: " + constraint
                        decision_variables = scenarios[choice] + _range + constraint
                    else:
                        decision_variables = scenarios[choice] + _range
                    break
                if choice == "2":
                    percentage_nr = input("\nEnter the increase in percentage (e.g 5): ")
                    percentage = f"can decrease with {percentage_nr} percent or stay at the same level"
                    constraint = input("\nOPTIONAL: add a contraint of how many machines can have decreased process times, leave blank if no constraint: ")
                    if constraint:
                        constraint = " At most " + constraint + " machines can have decreased process times."
                        decision_variables = scenarios[choice] + percentage + constraint
                    else:
                        decision_variables = scenarios[choice] + percentage
                    break
                if choice == "3":
                    percentage_nr = input("\nEnter the increase in percentage (e.g 5): ")
                    percentage = f"can increase with {percentage_nr} percent or stay at the same level"
                    constraint = input("\nOPTIONAL: add a contraint of how many machines can have increased availability, leave blank if no constraint: ")
                    if constraint:
                        constraint = " At most " + constraint + " machines can have increased availability."
                        decision_variables = scenarios[choice] + percentage + constraint
                    else:
                        decision_variables = scenarios[choice] + percentage
                    break

                if choice == "4":
                    percentage_nr = input("\nEnter the increase in percentage (e.g 5): ")
                    percentage = f"can decrease with {percentage_nr} percent or stay at the same level"
                    constraint = input("\nOPTIONAL: add a contraint of how many machines can have decreased MTTR, leave blank if no constraint: ")
                    if constraint:
                        constraint = " At most " + constraint + " machines can have decreased MTTR."
                        decision_variables = scenarios[choice] + percentage + constraint
                    else:
                        decision_variables = scenarios[choice] + percentage
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3 or 4.")

        print(f"\n[CONFIRMED] Scenario {choice} loaded:")
        print(f"The MOO algorithm will use these input variables to optimize the objectives:")
        print(f"{decision_variables}")
        print("-" * 30 + "\n")
        return decision_variables, choice


    def choose_hyper_parameters(self):
        print("\n" + "="*33)
        print("---HYPER PARAMETER SELECTION---")
        print("="*33)

        population_size = input("What population size would you like to use for the MOO algorithm? (e.g. 100) ")
        generations = input("How many generations should the MOO algorithm run for? (e.g. 50) ")
        SIM_TIME = input("How long should the simulation run for in seconds? (e.g. 691200 ) ")
        WARMUP_SECONDS = input("How long should the warmup period be for the simulation in seconds? (e.g. 86400) ")
        print("-" * 30 + "\n")
        
        return population_size, generations, SIM_TIME, WARMUP_SECONDS


    def _find_pareto_front(self,selected_objectives, directions):

        df = pd.read_csv("moo_simulation_results_buffer.csv")

        mask = paretoset(df[[selected_objectives[0], selected_objectives[1]]], sense=[directions[0], directions[1]])

        pareto_solutions = df[mask]

        MOO_pareto_csv_path = Path("moo_pareto_solutions.csv")

        pareto_solutions.to_csv(MOO_pareto_csv_path, index=False)

        with open(MOO_pareto_csv_path, "r") as file:
            pareto_solutions_string = file.read()

        return pareto_solutions_string
    

    
    def repair_and_run_code(self, code, path, objectives, decision_variables, client):
        # Iterative Debugging Loop
        max_attempts = 6
        attempt = 0
        error_message = None
        inspector = Modelinspector(client)

        while attempt < max_attempts:
            
            
            print(f"\nInspecting MOO and simulation code (Attempt {attempt + 1})...")

            # pass the error_message if it exists
            code = inspector._inspect_MOO(code, objectives, decision_variables, error_message)
            code = remove_code_wrappers(code)
            save_model(code, path, "checked_initial_combined_code.py")

            input("\nPlease review the code manually: 'results/checked_initial_combined_code.py'. Make changes if necessary. Press enter to run the MOO algorithm.")

            print("\nRunning the MOO algorithm...")
            #read the code again after manual review
            with open(os.path.join(path, "checked_initial_combined_code.py"), "r", encoding='utf-8') as f:
                code = f.read()
            
            try:
                #measure the time it takes to run the code
                start_time = time.time()
                _ = run_python_code(code)
                end_time = time.time()
                print(f"\nRun successful! Time taken: {end_time - start_time:.2f} seconds")
                break 
            except Exception as e:
                error_message = str(e)
                print(f"\nAttempt {attempt + 1} failed, this is the error message:\n\n {error_message}\n\n, repairing code...")
                attempt += 1
                if attempt == max_attempts:
                    print("\nMaximum fix attempts reached. Please fix the code manually.")
                    return None
                
    def json_to_csv(self, json_data, filename="suggested_improvements.csv"):
        try:
            root_key = list(json_data.keys())[0]
            records = json_data[root_key]

            df = pd.DataFrame(records)

            df.to_csv(filename, index=False)
            
            print(f"\nThe suggested datapoints for adaption can be seen in: '{filename}'")
            
        except Exception as e:
            print(f"Error converting JSON to CSV: {e}")

    

    def visualize_MOO_results(self, selected_objectives):
        
        # Load data
        df = pd.read_csv("moo_simulation_results_buffer.csv")
        df['Type'] = 'Standard Solution'

        df2 = pd.read_csv("moo_pareto_solutions.csv")
        df2['Type'] = 'Pareto Optimal'

        df3 = pd.read_csv("suggested_improvements.csv")
        df3['Type'] = 'LLM chosen points'

        # Combine them
        combined_df = pd.concat([df, df2, df3], ignore_index=True)

        hover_cols = [col for col in combined_df.columns if col not in selected_objectives and col != 'Type']
        
        # Create interactive scatter plot
        fig = px.scatter(
            combined_df, 
            x=selected_objectives[0], 
            y=selected_objectives[1], 
            color="Type",
            hover_data=hover_cols, # Now strictly contains metadata columns
            title="Interactive MOO Simulation Results",
            labels={
                selected_objectives[0]: selected_objectives[0].replace('_', ' ').title(),
                selected_objectives[1]: selected_objectives[1].replace('_', ' ').title()
            },
            template="plotly_white"
        )
        
        # Improve layout
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkGrey')))
        
        fig.show()

    def save_UML(self, UML_diagram, path):

        mmd_path = os.path.join(path, "MOO_visualization.mmd")
        png_path = os.path.join(path, "MOO_visualization.png")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(UML_diagram)
        render_mermaid_to_png(mmd_path, png_path, client=self.client)