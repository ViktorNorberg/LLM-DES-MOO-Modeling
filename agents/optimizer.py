import os

from pathlib import Path
from openai import OpenAI
import json
from helpers.other_helpers import remove_code_wrappers, remove_code_wrappers, save_model
from helpers.mermaid_renderer import render_mermaid_to_png
from helpers.runner import run_MOO_code
from paretoset import paretoset
from agents.visualizer import Modelvisualizer
import pandas as pd
import plotly.express as px


class Modeloptimizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def optimize(self, model_code):
        print("\nOptimizer activated:")
        path = Path("results")
        
        # Let the user choose the two objectives for the MOO algorithm and their corresponding optimization directions
        selected_objectives, directions = self._choose_objectives()
        objectives = f" {selected_objectives[0]} ({directions[0]}) and {selected_objectives[1]} ({directions[1]})"

        # Let the user specify the input variables for the MOO algorithm, their ranges, and whether they are discrete or continuous
        input_variables = input("What are the input variables that can be changed in the model? (e.g. buffer sizes, processing times, etc.) Please specify the range of values, and if the variable is discrete or continuous, for each variable as well. (e.g. Buffer 1 on the range [1,10] (discrete variable), buffer 2 on the range [1,10] (discrete variable) etc.) ")
        
        # Let the user choose the MOO algorithm to use, the population size, and the number of generations for the MOO algorithm
        algorithm = input("Which MOO algorithm would you like to use? (e.g. NSGA-II, MOEA/D, AGEMOEA etc.) ")
        population_size = input("What population size would you like to use for the MOO algorithm? (e.g. 100) ")
        generations = input("How many generations should the MOO algorithm run for? (e.g. 50) ")
        SIM_TIME = input("How long should the simulation run for in seconds? (e.g. 10000) ")
        WARMUP_SECONDS = input("How long should the warmup period be for the simulation in seconds? (e.g. 100) ")

        #Generate UML diagram for the MOO algorithm
        print("Generating UML diagram for the MOO algorithm...")
        visualizer = Modelvisualizer(self.client)
        UML_diagram = visualizer._generate_MOO_UML(model_code, objectives, input_variables)
        self.save_UML(UML_diagram, path)
        
        #Generate MOO code
        print("Generating MOO code...")
        MOO_code = self._generate_code(model_code, objectives, input_variables, UML_diagram, algorithm, population_size, generations)
        clean_initial_MOO_code = remove_code_wrappers(MOO_code)
        save_model(clean_initial_MOO_code, path, "MOO_initial_code.py")

        #Combine the MOO code with the simulation code
        print("Combining the MOO code with the simulation code...")
        combined_code = self._combiner(model_code, MOO_code, selected_objectives, SIM_TIME, WARMUP_SECONDS)
        clean_initial_combined_model = remove_code_wrappers(combined_code)
        save_model(clean_initial_combined_model, path, "initial_combined_code.py")

        #repair and run the code
        self.repair_and_run_code(clean_initial_combined_model, path, objectives, input_variables)

        #Extract Pareto-optimal solutions from the MOO results
        pareto_solutions = self._find_pareto_front(selected_objectives, directions)
        print("See the results of the MOO algorithm in 'moo_simulation_results.csv'")
        print("And the pareto optimal solutions here: 'moo_pareto_solutions.csv'")
   
        #Ask the user for their priorities and suggest improvements based on the Pareto-optimal solutions
        user_input = input("What are your current priorities for the production system? (e.g. prioritize high throughput, minimize energy consumption, etc.) ")

        #Generate suggestions for improvements based on the Pareto-optimal solutions and the user's priorities
        print("Generating suggestions for improvements based on the Pareto-optimal solutions and user input...")
        suggestions = self._suggest_improvements(model_code, user_input, pareto_solutions)

        #visualize the results
        self.json_to_csv(suggestions)
        self.visualize_MOO_results(selected_objectives)

        print(suggestions)

        explanation = self._explain_suggestions(suggestions, pareto_solutions, model_code)

        print(explanation)

        return suggestions
    

    def _explain_suggestions(self, suggestions, pareto_solutions, model_code,
        model = "gpt-5-mini"):
        prompt = (
            "You are an AI assistant that explains suggestions for improving a production system. "
            "The suggestions are based on the results of a multi-objective optimization (MOO) analysis, and are provided in a json format. "
            "Please explain the reasoning behind these suggestions in a way that is understandable for a human production manager. "
            f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
            f"Here are the results of the MOO analysis from which the suggestions are derived: \n\n {pareto_solutions}\n"
            f"Here are the suggestions in a json format:\n\n {suggestions}\n"
            "Please provide a clear, concise and short explanation for each suggestion, focusing on how it will improve the production system based on the MOO results and the user's priorities."
            "Do not end you answer with a question. "
        )
        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]) 
        return resp.choices[0].message.content
                             



    def _suggest_improvements(self, model_code, user_input, pareto_solutions,
        model = "gpt-5-mini",
        response_format={"type": "json_object"}):
        prompt = (
                "You are an AI assistant that suggests improvements to a production system based on the results of a multi-objective optimization (MOO) analysis. "
                "The improvements should be based on the results of a MOO analysis, which are provided in a csv format. "
                "Choose three datapoints on the provided pareto fron and suggest specific, implementable instructions to improve the system based on those datapoints. "
                f"When choosing the datapoints, consider the these instructions from the perspective of a production manager: \n {user_input}\n"
                "Based on the three datapoints you choose, suggest specific, implementable instructions to improve the system. "
                "They should be easily implementable with the existing model. "
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"Here are the results of the MOO analysis:\n\n {pareto_solutions}\n"
                "Only answer with the instructions in a json format."
                "The instructions should only contain the input variable settings and corresponding objectives values, no explanations"
                "The json format should be like this, though the variable names and values should be chosen from the MOO results: { 'instructions': [ {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 1, 'PostPress12Buffer': 1, 'throughput': 28.114285714285717, 'wip': 10.779661016949152}, {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 3, 'PostPress12Buffer': 1, 'throughput': 30.17142857142857, 'wip': 12.836158192090396}] }"
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
    

    def _generate_code(self, model_code, objectives, input_variables, UMLmmd, algorithm, population_size, generations,
            model ="gpt-5.1"):
            prompt = (
                 "You are an AI assistant that generates code for a multi-objective optimization (MOO) algorithm"
                 "Your task is to generate an MOO algorithm that optimizes a production line simulation model in Python."
                 f"This is the Python simulation model:\n\n```python\n {model_code}\n```\n\n"
                 f"These are the target objectives of the MOO algorithm: {objectives}\n"
                 f"These are the input variables for the MOO algorithm that can be adjusted: {input_variables}\n"
                 f"Use the {algorithm} algorithm for the MOO, with a population size of {population_size} and {generations} generations. "
                 "The MOO algortihm is to be written in python using the pymoo library. "
                 f"use the UML mmd {UMLmmd} file as guidence to how to implement you MOO algorithm"
                 "only output the code, no explanations, no markdown fences"
                 "make sure that the MOO code is compatible with the existing simulation code, and that it can be easily integrated with the existing code"
                 
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
            "The final combined python code should return a csv file with all the different solutions from every generation found by the MOO algorithm"
            f"Make sure that the csv file contain KPI values for the selected objectives: {selected_objectives}, the column names should be the same as the objective names. "
            "Name the csv file 'moo_simulation_results.csv'")
        
        resp= self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2)
        
        return resp.choices[0].message.content

    def _inspector(self, combined_code, objectives, input_variables, error_message=None, 
        model="gpt-5.1"):

        if error_message:

            prompt = (
                f"You are a Senior Python Developer. Please evaluate the following Python code and the error message from the last execution attempt. "
                f"This was the original code: {combined_code}"
                f"The previous code failed with the following error:\n"
                f"--- ERROR ---\n{error_message}\n--------------\n"
                f"Please analyze the error and the code, and provide a corrected version. "
                f"Make sure that the MOO algorithm optimizes these objectives: {objectives}, by changing these input variables: {input_variables}. "
                "Only output the corrected code, no explanations, no markdown fences."
                "Make minimal adjustments to the original by only changing the parts of the code that are causing the error, and keep the rest of the code intact. "
            )

        else:
            prompt = (
            "Please evaluate if the following Python code is correct and will run without errors. "
            "If it is correct, do nothing. If it is incorrect, please adapt it so that it runs correctly. Only answer with the code.\n\n"
            f"```python\n{combined_code}\n```"
            f"Make sure that the MOO algorithm optimizes these objectives: {objectives}, by changing these input variables: {input_variables}. "
            "Output ONLY the corrected Python code. No explanations, no markdown fences."
            )

        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.1
        )
        return resp.choices[0].message.content

    
    def _choose_objectives(self):
                
        possible_metrics = ["wip", "throughput", "energy consumption", "lead time", "utilization"]
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
                            print(f"'{metric_name}' is already selected. Please pick a different objective.")
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
                        
                        print(f"Slot {i} set to: {direction.upper()} {metric_name}")
                        break  # Exit inner while loop, move to next 'i' in for loop
                    else:
                        print("Invalid selection. Out of range.")
                except ValueError:
                    print("Please enter a numerical value.")

        print(f"\nFinal configuration: {selected_objectives[0]} ({directions[0]}) vs {selected_objectives[1]} ({directions[1]})\n\n")
        return selected_objectives, directions
    


    def _find_pareto_front(self,selected_objectives, directions):

        df = pd.read_csv("moo_simulation_results.csv")

        mask = paretoset(df[[selected_objectives[0], selected_objectives[1]]], sense=[directions[0], directions[1]])

        pareto_solutions = df[mask]

        MOO_pareto_csv_path = Path("moo_pareto_solutions.csv")

        pareto_solutions.to_csv(MOO_pareto_csv_path, index=False)

        with open(MOO_pareto_csv_path, "r") as file:
            pareto_solutions_string = file.read()

        return pareto_solutions_string
    

    
    def repair_and_run_code(self, code, path, objectives, input_variables):
        # Iterative Debugging Loop
        max_attempts = 6
        attempt = 0
        error_message = None

        while attempt < max_attempts:
            print(f"Inspecting code (Attempt {attempt + 1})...")

            # pass the error_message if it exists
            code = self._inspector(code, objectives, input_variables, error_message)
            code = remove_code_wrappers(code)
            save_model(code, path, "checked_initial_combined_code.py")

            input("Please review the code manually: 'results/checked_initial_combined_code.py'. Make changes if necessary. Press enter to run the MOO algorithm. \n" \
            "Tip: Run the code in a separate environment with a short simulation time for testing and look at the resultsn\n" \
            "scenario 1: the code casts an error. Then continue this workflow by pressing enter \n" \
            "scenario 2: the code runs but the results are no good. Then you may try to find the problem yourself or rerunning the entire workflow\n" \
            "scenario 3: the code runs and yield seemingly good results. Then continue this workflow by pressing enter \n"
            "before continuing in scenario 1 or 3, make sure to change back the simulation time to 8 days \n" \
            "Follow this process to avoid unsatisfactory results from the long MOO run")

            print("Running the MOO algorithm...")
            #read the code again after manual review
            with open(os.path.join(path, "checked_initial_combined_code.py"), "r", encoding='utf-8') as f:
                code = f.read()
            
            try:
                # Assuming run_python_code raises an Exception on failure
                # or returns a result indicating failure.
                _ = run_MOO_code(code)
                print("Run successful!")
                break 
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {attempt + 1} failed, this is the error message:\n\n {error_message}\n\n, repairing code...")
                attempt += 1
                if attempt == max_attempts:
                    print("Maximum fix attempts reached. Please fix the code manually.")
                    return None
                
    def json_to_csv(self, json_data, filename="suggested_improvements.csv"):
        try:
            # 1. Dynamically find the key that holds the list (e.g., 'instructions')
            # This takes the first key it finds in the dictionary
            root_key = list(json_data.keys())[0]
            records = json_data[root_key]

            # 2. Convert to DataFrame
            # Pandas automatically extracts column names from the dictionary keys
            df = pd.DataFrame(records)

            # 3. Save to the main folder
            df.to_csv(filename, index=False)
            
            print(f"Successfully saved {len(df)} datapoints to '{filename}'")
            #print(f"Columns identified: {list(df.columns)}")
            
        except Exception as e:
            print(f"Error converting JSON to CSV: {e}")

    def visualize_MOO_results(self, selected_objectives):
        
        # Load data
        df = pd.read_csv("moo_simulation_results.csv")
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
        
        # Improve layout: force the legend to be visible and markers to be distinct
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkGrey')))
        
        fig.show()

    def save_UML(self, UML_diagram, path):

        mmd_path = os.path.join(path, "UML.mmd")
        png_path = os.path.join(path, "UML.png")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(UML_diagram)
        render_mermaid_to_png(mmd_path, png_path, client=self.client)