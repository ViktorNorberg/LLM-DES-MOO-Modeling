
import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.optimizer import Modeloptimizer

load_dotenv()

api_key= os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api_key)


code_path = os.path.join("results", "initial_model.py")
with open(code_path, "r") as file:
    code_input = file.read()


optimizer = Modeloptimizer(client)

suggestions = optimizer.optimize(model_code = code_input)

print(suggestions)