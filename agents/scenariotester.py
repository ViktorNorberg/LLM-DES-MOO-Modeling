import json

from openai import OpenAI
from helpers.other_helpers import visualize_results
from agents.adapter import Modeladaptor
import matplotlib.pyplot as plt
import json

class Scenarioagent:
    def __init__(self, client: OpenAI):
        self.client = client

    def find_scenarios(self, bottleneck_suggestions, model_code, stations_table_md, sequence_text, final_path, results_initial_model):
        print("\nDefining bottleneck scenarios: ")

        bottleneck_machine = self.find_bottleneck_machine(bottleneck_suggestions)
        print(f"Identified bottleneck machine: {bottleneck_machine}")

        bottleneck_scenarios = self.scenarios(bottleneck_machine, model_code, stations_table_md, sequence_text)
        print(f"Generated scenarios for evaluationbased on bottleneck {bottleneck_machine}: {bottleneck_scenarios}")
            
        if isinstance(bottleneck_scenarios, dict):
            step_list = bottleneck_scenarios.get("instructions", [])
        elif isinstance(bottleneck_scenarios, list):
            step_list = bottleneck_scenarios
        else:
            raise ValueError("Unexpected format from Scenarioagent scenarios")
        
        for idx, step in enumerate(step_list, start=1):
            adaptor = Modeladaptor(self.client)
            kpi_adapted_model = adaptor.adapter(original_code = model_code, instruction=step, final_path=final_path,  name="Bottleneck_scenario", multi_agent_setting= False, index_model= idx)
            print(kpi_adapted_model) # Append each adapted model's KPIs to results
            results_initial_model.append(kpi_adapted_model)

        visualize_results(results_initial_model, file_name="bottleneck_scenarios_evaluations.png", save_path=final_path)
        plt.show()

        print("\nFinished evaluating bottleneck scenarios. Returning to main program.")
        print("Look at the generated figure to compare the KPIs of the different scenarios with the original model in: 'results/bottleneck_scenarios_evaluations.png'")

        return 

    def scenarios(self, bottleneck_machine, model_code, stations_table_md, sequence_text,
        model = "gpt-5-mini",
        response_format={"type": "json_object"}): 
        prompt = (
            "Your task is to generate instructions for testing a production line"
            f"This is the simulation model: \n {model_code}"
            f"Stations table: \n {stations_table_md}"
            f"Direct-follow relationsships: {sequence_text}"
            f"You should choose instructions of interest depending on the bottleneck of the system which is: \n {bottleneck_machine} "
            "Specifically, you should answer with five separate scenarios: "
            "Improving the MTTR, availability and process time of the bottlneck, and increasing the buffer size of the buffers before and after the bottlneck machine"
            "All insructions should be easy to understand and short"
            "Example: Answer in JSON format like this: { 'instructions': [ {'Press cell 2': reduce MTTR by 10%}, {'Press cell 2': reduce process time by 10%}, {'Press cell 2': increase availability by 10%}, {'PrePress2Buffer': double buffer size}, {'PostPress2Buffer': double buffer size} ] }  "
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

    def find_bottleneck_machine(self, bottleneck_suggestions,
        model = "gpt-5-mini"):
        prompt = (
            f"Given thes JSON file of suggested improvements to bottlenecks of a production line, pick out the bottleneck that appear most frequently in the JSON file"
            f"This is the JSON file: \n {bottleneck_suggestions}"
            "Only answer with the name of the machine, nothing else"
            "There can only be 1 bottleneck machine in the answer"
        )
        resp = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content