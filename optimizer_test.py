import json
import os
from pathlib import Path
from numpy.compat import Path

from openai import OpenAI
from helpers.mermaid_renderer import render_mermaid_to_png
from helpers.other_helpers import remove_code_wrappers, save_model
from helpers.runner import run_python_code
from dotenv import load_dotenv




# Define your test inputs
code_path = os.path.join("results", "initial_model.py")
# 2. Open and read the file
with open(code_path, "r") as file:
    code_input = file.read()
"""
model_name = "gpt-4o"

# Execute
def retrieve_KPIs(code, modelinfo: str):
    original_output = run_python_code(code).splitlines()
    kpi_section = [f"----Results from model: {modelinfo} {original_output[0]}"]
    return kpi_section


kpis = retrieve_KPIs(code_input, model_name)

# Print results so you can see them in the terminal
print(kpis)
"""


load_dotenv()
    
api_key= os.getenv("OPENAI_KEY")

client = OpenAI(api_key=api_key)

class Modeloptimizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def _generate_UML(self, model_code: str, objectives, input_variables, output_variables,
        model: str ="gpt-4o-mini") -> str:
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
                f"Output Variables: {output_variables}\n"
                 "HARD REQUIREMENTS:\n"
                "1) Output ONLY Mermaid code: no markdown fences, no ```python blocks, no explanations.\n"
                "2) The first non-empty line MUST be exactly: flowchart TD\n"
                "Only answer with the UML diagram in a mermaid code format."
                "Label format:\n"
                "- Use valid identifiers (letters, digits, underscore) for node IDs.\n"
                "- ALL node labels must use real line breaks inside the brackets.\n"
                "- Never output '\\n' or '\\\\n' anywhere in any label\n"
                "- The first line of every label must be the node name written in **bold**, using Markdown syntax.\n")

        resp = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        try:
            return resp.choices[0].message.content
        except Exception as e:
            raise

    def _generate_code(self, model_code, objectives, input_variables, output_variables, UMLmmd,
            model = "gpt-4o"):
            prompt = (
                 "You are an AI assistant that generates code for a multi-objective optimization algorithm"
                 "Your task is to generate an MOO algorithm that optimizes the following Python code. "
                 f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                 f"Objectives: {objectives}\n"
                 f"Input Variables: {input_variables}\n"
                 f"Output Variables: {output_variables}\n"
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
            return

    def _combiner(self,model_code, objectives, input_variables, output_variables, MOO_code,
        model = "gpt-4o"):
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
            "the final combined python code should return a csv file with all the different solutions found by the MOO algorithm, with their corresponding KPI values.")
        resp= self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

optimizer = Modeloptimizer(client)
path = Path("results")
"""
UMLmmd = optimizer._generate_UML(code_input, "Maximize throughput, minimize WIP", "Buffer sizes on the range [1,10]", "Throughput, Energy Consumption")

path = Path("results")
mmd_path = os.path.join(path, "UML.mmd")
png_path = os.path.join(path, "UML.png")
with open(mmd_path, "w", encoding="utf-8") as f:
    f.write(UMLmmd)

render_mermaid_to_png(mmd_path, png_path)


# Define your test inputs
mmd_path = os.path.join("results", "UML.mmd")
# 2. Open and read the file
with open(mmd_path, "r") as file:
    UMLmmd = file.read()


MOO_initial_code = optimizer._generate_code(code_input, "Maximize throughput, minimize WIP", "Buffer sizes on the range [1,10]", "Throughput, Energy Consumption", UMLmmd)
clean_initial_model = remove_code_wrappers(MOO_initial_code)
save_model(clean_initial_model,path, "MOO_initial_code2.py")
"""
# Define your test inputs
MOO_path = os.path.join("results", "MOO_initial_code2.py")
# 2. Open and read the file
with open(MOO_path, "r") as file:
    MOO_initial_code = file.read()

combined_code = optimizer._combiner(code_input, "Maximize throughput, minimize WIP", "Buffer sizes on the range [1,10]", "Throughput, Energy Consumption", MOO_initial_code)
clean_initial_model = remove_code_wrappers(combined_code)
save_model(clean_initial_model,path, "initial_combined_code.py")