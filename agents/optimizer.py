import os

from pathlib import Path
from openai import OpenAI
import json
from helpers.other_helpers import remove_code_wrappers, remove_code_wrappers, save_model
from helpers.mermaid_renderer import render_mermaid_to_png
from helpers.runner import run_python_code
from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


class Modeloptimizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def optimize(self, model_code):
        print("\nOptimizer activated:")
        path = Path("results")
        
        
        selected_objectives, directions = self._choose_objectives()
        objectives = f" {selected_objectives[0]} ({directions[0]}) and {selected_objectives[1]} ({directions[1]})"

        #Generate UML diagram for the MOO algorithm
        
        input_variables = input("What are the input variables that can be changed in the model? (e.g. buffer sizes, processing times, etc.) Please specify the range of values, and if the variable is discrete or continuous, for each variable as well. (e.g. Buffer 1 on the range [1,10] (discrete variable), buffer 2 on the range [1,10] (discrete variable) etc.) ")
        
        
        UML_diagram = self._generate_UML(model_code, objectives, input_variables)

        mmd_path = os.path.join(path, "UML.mmd")
        png_path = os.path.join(path, "UML.png")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(UML_diagram)
        render_mermaid_to_png(mmd_path, png_path)
        


        #Generate MOO code
        print("generating MOO code...")
        MOO_code = self._generate_code(model_code, objectives, input_variables, UML_diagram)
        clean_initial_MOO_code = remove_code_wrappers(MOO_code)
        save_model(clean_initial_MOO_code, path, "MOO_initial_code.py")

        #Combine the MOO code with the existing code
        print("Combining the MOO code with the existing code...")
        combined_code = self._combiner(model_code, MOO_code, selected_objectives)
        clean_initial_combined_model = remove_code_wrappers(combined_code)
        save_model(clean_initial_combined_model, path, "initial_combined_code.py")

        #repair and run the code
        self.repair_and_run_code(clean_initial_combined_model, path, objectives, input_variables)

        #Extract Pareto-optimal solutions from the MOO results
        pareto_solutions = self._find_pareto_front(selected_objectives, directions)

        #visualize the results
        self.visualize_MOO_results(selected_objectives)
    
        #Ask the user for their priorities and suggest improvements based on the Pareto-optimal solutions
        user_input = input("What are your current priorities for the production system? (e.g. prioritize high throughput, minimize energy consumption, etc.) ")
        print("Generating suggestions for improvements based on the Pareto-optimal solutions...")
        suggestions = self._suggest_improvements(model_code, user_input, pareto_solutions)
        return suggestions
    


    def _suggest_improvements(self, model_code, user_input, MOO_pareto_csv,
        model = "gpt-5-mini",
        response_format={"type": "json_object"}):
        prompt = (
                "You are an AI assistant that suggests improvements to a production system based on the results of a multi-objective optimization (MOO) analysis. "
                "The improvements should be based on the results of a MOO analysis, which are provided in a csv format. "
                "Choose three datapoints on the provided pareto fron and suggest specific, implementable instructions to improve the system based on those datapoints. "
                f"When choosing the datapoints, consider the these instructions from the perspective of a production manager: \n {user_input}\n"
                "Based on the three datapoints you choose, suggest specific, implementable instructions to improve the system. "
                "These should be short and concise statements, e.g. Increase the buffer size of buffer_1 to 8. buffer_2 to 10... etc. "
                "They should be easily implementable with the existing model. "
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"Here are the results of the MOO analysis:\n\n {MOO_pareto_csv}\n"
                "Only answer with the instructions in a json format."
                "The instructions should only contain the input variable settings and corresponding objectives values, no explanations"
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
    
    def _generate_UML(self, model_code: str, objectives, input_variables,
        model: str ="gpt-5-mini") -> str:
        prompt = (
                "You are an AI assistant that generates UML activity diagrams from natural language."
                "The UML should include the classes, their attributes, and methods. "
                "Focus on the parts of the code that are relevant to the following objectives, input variables, and output variables.\n\n"
                "The provided python code is of a simulation model of a production system"
                "Please generate a UML digram of an MOO algorithm that will optimize the following simulation"
                "Your UML diagram should be focused on how the MOO algorithm will interact with the existing code, and how it will optimize it. "
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"Objectives: {objectives}\n"
                f"Input Variables: {input_variables}\n"

                 "HARD REQUIREMENTS:\n"
                "1) Output ONLY Mermaid code: no markdown fences, no ```python blocks, no explanations.\n"
                "2) The first non-empty line MUST be exactly: flowchart TD\n"
                "Only answer with the UML diagram in a mermaid code format."
                "3) IMPORTANT: Never use square brackets [ ] or parentheses ( ) inside a node label. "
                "Use curly braces { } or just plain text for ranges (e.g., {1-10} instead of [1,10]).\n"
                "4) Every node must be formatted as: NodeID[\"**Node Name**<br/>Description\"]"

                "Label format:\n"
                "- Use valid identifiers (letters, digits, underscore) for node IDs.\n"
                "- ALL node labels must use real line breaks inside the brackets.\n"
                "- Never output '\\n' or '\\\\n' anywhere in any label\n"
                "- The first line of every label must be the node name written in **bold**, using Markdown syntax.\n"
                )

        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]) 
            #temperature=0.1)
        try:
            return resp.choices[0].message.content
        except Exception as e:
            raise

    def _generate_code(self, model_code, objectives, input_variables, UMLmmd,
             model ="gpt-5.1"):
            prompt = (
                 "You are an AI assistant that generates code for a multi-objective optimization (MOO) algorithm"
                 "Your task is to generate an MOO algorithm that optimizes a production line simulation model in Python."
                 f"This is the Python simulation model:\n\n```python\n {model_code}\n```\n\n"
                 f"These are the target objectives of the MOO algorithm: {objectives}\n"
                 f"These are the input variables for the MOO algorithm that can be adjusted: {input_variables}\n"
                 "The MOO algortihm are to be written in python using the pymoo library. "
                 f"use the UML mmd {UMLmmd} file as guidence to how to implement you MOO algorithm"
                 "Make the algorithm print what generation is currently running"
                 "only output the code, no explanations, no markdown fences"
                 "make sure that the MOO code is compatible with the existing simulation code, and that it can be easily integrated with the existing code"
                 ""
            )
            resp = self.client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.2)
            try:                
                return resp.choices[0].message.content
            except Exception as e:                
                raise e     
            
    
    def _combiner(self, model_code, MOO_code, selected_objectives,
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
        max_attempts = 5
        attempt = 0
        error_message = None

        while attempt < max_attempts:
            print(f"Inspecting code (Attempt {attempt + 1})...")
            # We pass the error_message if it exists
            code = self._inspector(code, objectives, input_variables, error_message)
            code = remove_code_wrappers(code)
            save_model(code, path, "checked_initial_combined_code.py")

            input("Please review the code manually: 'results/checked_initial_combined_code.py'.")
            print("Running the MOO algorithm...")
            #read the code again after manual review
            with open(os.path.join(path, "checked_initial_combined_code.py"), "r") as f:
                code = f.read()
            
            try:
                # Assuming run_python_code raises an Exception on failure
                # or returns a result indicating failure.
                _ = run_python_code(code)
                print("✅ Run successful!")
                break # Exit the loop on success
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {attempt + 1} failed, this is the error message: {error_message}, repairing code...")
                attempt += 1
                if attempt == max_attempts:
                    print("Maximum fix attempts reached. Please check the code manually.")
                    return None

    def visualize_MOO_results(self, selected_objectives):
        
        # Load your data
        df = pd.read_csv("moo_simulation_results.csv")
        df['Type'] = 'Standard Solution'

        # If you want to include the Pareto solutions in the same interactive plot:
        df2 = pd.read_csv("moo_pareto_solutions.csv")
        df2['Type'] = 'Pareto Optimal'

        # Combine them
        combined_df = pd.concat([df, df2], ignore_index=True)

        hover_cols = [col for col in combined_df.columns if col not in selected_objectives and col != 'Type']
        
        # Create the interactive scatter plot
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
