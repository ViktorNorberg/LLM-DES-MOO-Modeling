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

class BottleneckOptimizer:
    def __init__(self, client):
        self.client = client
    
    def optimize(self, model_code, SCORE_blueprint):

        print("\nBottleneck agent activated:")
        path = Path("results")

        WARMUP_SECONDS = 100
        SIM_TIME = 3600
        top_n_fronts = 5

        MOO_code = self._generate_code(model_code, SCORE_blueprint)
        clean_initial_MOO_code = remove_code_wrappers(MOO_code)
        save_model(clean_initial_MOO_code, path, "SCORE_initial_code.py")

        #Combine the MOO code with the simulation code
        print("\nCombining the SCORE code with the simulation code...")
        combined_code = self._combiner(model_code, MOO_code, WARMUP_SECONDS, SIM_TIME)
        clean_initial_combined_model = remove_code_wrappers(combined_code)
        save_model(clean_initial_combined_model, path, "SCORE_initial_combined_code.py")

        self.repair_and_run_code(clean_initial_combined_model, path, WARMUP_SECONDS, SIM_TIME, self.client)

        _ = self._find_pareto_front(top_n_fronts)

        frequency_analysis = self.get_flag_frequencies()

        print("\nSee the results of the SCORE analysis in 'SCORE_results.csv'")
        print(f"And the top rank {top_n_fronts} solutions in: 'SCORE_pareto_solutions.csv'")

        suggestions = self._suggest_improvements(clean_initial_combined_model, frequency_analysis)

        print("\nSuggested changes from the bottleneck analysis: ")
        print("")
        print(suggestions)
        print("\n")

        explanations = self._explain_suggestions(suggestions, model_code)
        print("\n")
        print(explanations)
        print("")

        return suggestions
    

    def _explain_suggestions(self, suggestions, model_code,
        model = "gpt-5-mini"):
        prompt = (
            "You are an AI assistant that explains suggestions for improving a production system. "
            "The suggestions are based on the results of a bottleneck analysis, and are provided in a json format. "
            "Please explain the reasoning behind these suggestions in a way that is understandable for a human production manager. "
            f"Here is my simulation code of the production system:\n\n```python\n {model_code}\n```\n\n"
            f"Here are the suggestions in a json format:\n\n {suggestions}\n"
            "Please provide a clear, concise and short explanation for each suggestion, focusing on how it will improve the production system based on the bottleneck results"
            "Do not end you answer with a question. "
        )
        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]) 7
        return resp.choices[0].message.content


    def _suggest_improvements(self, model_code, frequency_analysis,
        model = "gpt-5-mini",
        response_format={"type": "json_object"}):
        prompt = (
                "You are an AI assistant that suggests improvements to a production system based on the results of a bottleneck analysis. "
                "The improvements should be based on a frequency analysis of the system bottlenecks"
                " Please name 5 specific implementable instructions to improve the system. "
                "These should be short and concise statements, e.g. 'Increase the availability of the quality station by 10%' "
                "They should be easily implementable with the existing model. "
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"Here are the results of the bottleneck analysis:\n\n {frequency_analysis}\n"
                "Only answer with the instructions in a json format."
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
            raise e
    


    def _generate_code(self, model_code, SCORE_blueprint,
        model ="gpt-5.1"):
        prompt = (
                "You are an AI assistant that generates code for a multi-objective optimization (MOO) algorithm"
                "Your task is to generate an MOO algorithm that for a line simulation model in Python."
                f"This is the Python simulation model:\n\n```python\n {model_code}\n```\n\n"
                f"If a point violates the constraint, dont evaluate its objective values, and the datapoint should not be in the results csv file"
                "Please modify the following blueprint MOO to fit the simulation model"
                f"```python\n{SCORE_blueprint}\n```\n\n"
                "IMPORTANT: make sure that only the flag variables and the objectives are in the results csv file"
                "The objectives names in the csv file should be 'active_flags' and 'throughput'"
                "Make sure the variable names are self explanatory like in the blueprint, e.g. QualityStation_Avail_Inc_10per "
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
        
    def _combiner(self, model_code, MOO_code, WARMUP_SECONDS, SIM_TIME,
        model ="gpt-5.1"):
        prompt = (
                "You are an AI assistant that combines two pieces of Python code into one coherent script."
                "The first piece of code is a simulation model, and the second piece of code is a multi-objective optimization (MOO) algorithm."
                "Your task is to combine these two pieces of code into one script that can be executed without errors."
                f"Here is the simulation model:\n\n```python\n {model_code}\n```\n\n"
                f"Here is the MOO code:\n\n```python\n{MOO_code}\n```\n\n"
                f"Set the WARMPUP_SECONDS to: \n\n {WARMUP_SECONDS}\n"
                f"Set the SIM_TIME to: \n\n {SIM_TIME}\n"
                "Make sure the file name of the csv file is exactly: 'SCORE_results.csv' "
                "Only output the combined code, no explanations, no markdown fences."
        )
        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.2)
        try:                
            return resp.choices[0].message.content
        except Exception as e:                
            raise e

    
    def repair_and_run_code(self, code, path, WARMUP_SECONDS, SIM_TIME, client):
        # Iterative Debugging Loop
        max_attempts = 6
        attempt = 0
        error_message = None
        inspector = Modelinspector(client)

        while attempt < max_attempts:
            
            
            print(f"\nInspecting SCORE and simulation code (Attempt {attempt + 1})...")

            # pass the error_message if it exists
            code = inspector._inspect_SCORE(code, WARMUP_SECONDS, SIM_TIME, error_message)
            code = remove_code_wrappers(code)
            save_model(code, path, "SCORE_checked_initial_combined_code.py")

            input("\nPlease review the code manually: 'results/SCORE_checked_initial_combined_code.py'. Make changes if necessary. Press enter to run the SCORE analysis.")

            print("\nRunning the SCORE analysis...")
            #read the code again after manual review
            with open(os.path.join(path, "SCORE_checked_initial_combined_code.py"), "r", encoding='utf-8') as f:
                code = f.read()
            
            try:
                # Assuming run_python_code raises an Exception on failure
                # or returns a result indicating failure.
                _ = run_python_code(code)
                print("\nRun successful!")
                break 
            except Exception as e:
                error_message = str(e)
                print(f"\nAttempt {attempt + 1} failed, this is the error message:\n\n {error_message}\n\n, repairing code...")
                attempt += 1
                if attempt == max_attempts:
                    print("\nMaximum fix attempts reached. Please fix the code manually.")
                    return None
                
    

    def _find_pareto_front(self,top_n_fronts):
        
        selected_objectives = ["active_flags", "throughput"]
        directions = ["min", "max"]  
        df = pd.read_csv("SCORE_results.csv")
        
        # Create a temporary column for internal tracking
        df['temp_pareto_rank'] = pd.NA
        
        remaining_indices = df.index
        current_rank = 1
        
        # Iteratively find the Pareto front and remove it from the pool
        while current_rank <= top_n_fronts and len(remaining_indices) > 0:
            df_subset = df.loc[remaining_indices]
            
            mask = paretoset(
                df_subset[[selected_objectives[0], selected_objectives[1]]], 
                sense=[directions[0], directions[1]]
            )
            
            current_front_indices = df_subset[mask].index
            df.loc[current_front_indices, 'temp_pareto_rank'] = current_rank
            
            remaining_indices = remaining_indices.difference(current_front_indices)
            current_rank += 1
            
        # Filter to keep only the top N ranked solutions
        pareto_solutions = df[df['temp_pareto_rank'].notna()].copy()
        
        # Sort by the temporary rank first, then by the first objective
        pareto_solutions.sort_values(by=['temp_pareto_rank', selected_objectives[0]], inplace=True)

        # Drop the temporary rank column so it doesn't appear in the final output
        pareto_solutions.drop(columns=['temp_pareto_rank'], inplace=True)

        MOO_pareto_csv_path = Path("SCORE_pareto_solutions.csv")
        pareto_solutions.to_csv(MOO_pareto_csv_path, index=False)

        with open(MOO_pareto_csv_path, "r") as file:
            pareto_solutions_string = file.read()

        return pareto_solutions_string


    def get_flag_frequencies(self):
        # 1. Read the data into a Pandas DataFrame
        df = pd.read_csv("SCORE_pareto_solutions.csv")
        
        # 2. Define the columns to exclude (metadata and objectives)
        exclude_columns = ['active_flags', 'throughput']
        
        # 3. Filter the dataframe to only include the flag (variable) columns
        flag_columns = [col for col in df.columns if col not in exclude_columns]
        df_flags = df[flag_columns]
        
        # 4. Sum the columns. Since values are 0 or 1, the sum is the frequency.
        # Convert this directly to a dictionary.
        frequency_dict = df_flags.sum().to_dict()
        
        # 5. Sort the dictionary by value (frequency) in descending order
        sorted_frequency = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_frequency
