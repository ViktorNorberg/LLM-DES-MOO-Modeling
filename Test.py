
import json
import os
from pathlib import Path

from helpers.runner import run_MOO_code, run_python_code
from agents.optimizer import Modeloptimizer
from openai import OpenAI
from dotenv import load_dotenv

"""

load_dotenv()

api_key= os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def _suggest_improvements(client, model_code, user_input, pareto_solutions,
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
        
        resp = client.chat.completions.create(
            model=model,
            response_format = response_format, 
            messages=[{"role": "user", "content": prompt}])
        try:
            operator_output = resp.choices[0].message.content.strip()
            instructions_json = json.loads(operator_output)
            return instructions_json
        except Exception as e:
            raise

code_path = os.path.join("results", "initial_model.py")
# 2. Open and read the file
with open(code_path, "r") as file:
    model_code = file.read()

MOO_pareto_csv_path = Path("moo_pareto_solutions.csv")
with open(MOO_pareto_csv_path, "r") as file:
    pareto_solution = file.read()

suggestions = _suggest_improvements(client, model_code, user_input="prioritize high throughput", pareto_solutions = pareto_solution)
print(suggestions)


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
path = Path("results")

with open(os.path.join(path, "checked_initial_combined_code.py"), "r", encoding='utf-8') as f:
    code = f.read()
    
run_MOO_code(code)

#added comment
#flsjdf