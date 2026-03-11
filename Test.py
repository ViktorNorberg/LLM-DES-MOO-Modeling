
import os
from pathlib import Path
from helpers.runner import run_python_code
from agents.optimizer import Modeloptimizer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)

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