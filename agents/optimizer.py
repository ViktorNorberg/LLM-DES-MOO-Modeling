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

class Modeloptimizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def optimize(self, model_code):
        print("\nOptimizer activated:")
        path = Path("results")
        
        #Generate UML diagram for the MOO algorithm
        objectives = input("What are the objectives you want to optimize for? (e.g. maximize throughput, minimize energy consumption, etc.) ")
        input_variables = input("What are the input variables that can be changed in the model? (e.g. buffer sizes, processing times, etc.) Please specify the range of values, and if the variable is discrete or continuous, for each variable as well. (e.g. Buffer 1 on the range [1,10] (discrete variable), buffer 2 on the range [1,10] (discrete variable) etc.) ")
        UML_diagram = self._generate_UML(model_code, objectives, input_variables)

        mmd_path = os.path.join(path, "UML.mmd")
        png_path = os.path.join(path, "UML.png")
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(UML_diagram)
        render_mermaid_to_png(mmd_path, png_path)

        #Generate MOO code
        MOO_code = self._generate_code(model_code, objectives, input_variables, UML_diagram)
        clean_initial_MOO_code = remove_code_wrappers(MOO_code)
        save_model(clean_initial_MOO_code, path, "MOO_initial_code.py")

        #Combine the MOO code with the existing code
        combined_code = self._combiner(model_code, MOO_code)
        clean_initial_combined_model = remove_code_wrappers(combined_code)
        save_model(clean_initial_combined_model, path, "initial_combined_code.py")

        combined_code = os.path.join("results", "initial_combined_code.py")
        # 2. Open and read the file
        with open(combined_code, "r") as file:
            combined_code = file.read()

        checked_combined_code = self._inspector(combined_code,objectives, input_variables)
        clean_checked_combined_code = remove_code_wrappers(checked_combined_code)
        save_model(clean_checked_combined_code, path, "checked_initial_combined_code.py")

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
        suggestions = self._suggest_improvements(model_code, user_input, MOO_pareto_csv)
        return suggestions

    def _suggest_improvements(self, model_code, user_input, MOO_pareto_csv,
        model = "gpt-5.1",
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
                "Each instruction should come with a brief explanation of why you chose that datapoint and how it will improve the system. ")
        resp = self.client.chat.completions.create(
            model=model,response_format = response_format, messages=[{"role": "user", "content": prompt}])
        try:
            operator_output = resp.choices[0].message.content.strip()
            instructions_json = json.loads(operator_output)
            return instructions_json
        except Exception as e:
            raise
    
    def _generate_UML(self, model_code: str, objectives, input_variables,
        model: str ="gpt-5.1") -> str:
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
            model=model, messages=[{"role": "user", "content": prompt}])
        try:
            return resp.choices[0].message.content
        except Exception as e:
            raise

    def _generate_code(self, model_code, objectives, input_variables, UMLmmd,
             model ="gpt-5.1"):
            prompt = (
                 "You are an AI assistant that generates code for a multi-objective optimization algorithm"
                 "Your task is to generate an MOO algorithm that optimizes the following Python code. "
                 f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                 f"Objectives: {objectives}\n"
                 f"Input Variables: {input_variables}\n"
                 "The MOO algortihm you be written in python using the pymoo library. "
                 f"use the UML mmd {UMLmmd} file as guidence to how to implement you MOO algorithm"
                 "only output the code, no explanations, no markdown fences"
                 "make sure that the MOO code is compatible with the existing simulation code, and that it can be easily integrated with the existing code"
            )
            resp = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}])
            try:                
                return resp.choices[0].message.content
            except Exception as e:                
                raise e     
            
    
    def _combiner(self, model_code, MOO_code,
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
            "the final combined python code should return a csv file with all the different solutions found by the MOO algorithm, with their corresponding KPI values."
            "Name the csv file 'moo_simulation_results.csv'")
        resp= self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    def _inspector(self, combined_code, objectives, input_variables,
        model = "gpt-5.1"):
        prompt = (
            "Please evaluate if the following Python code is correct. "
            "If it is correct, do nothing. If it is incorrect, please adapt it so that it runs correctly. Only answer with the code.\n\n"
            f"```python\n{combined_code}\n```"
            f"Make sure that the MOO algorithm optimizes these objectives: {objectives}, by changing these input variables: {input_variables}. ")
        resp = self.client.chat.completions.create(
            model = model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    
        

