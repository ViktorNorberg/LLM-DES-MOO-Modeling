from openai import OpenAI
from helpers import 

class Scenarioagent:
    def __init__(self, client: OpenAI):
        self.client = client

    def find_scenarios(self, bottleneck_suggestions):
        print("\nDefining bottleneck scenarios: ")

        bottleneck_machine = self.find_bottleneck_machine(bottleneck_suggestions)

        bottleneck_scenarios = self.scenarios(bottleneck_machine, model_code, stations_table_md, sequence_text)
            
        if isinstance(bottleneck_scenarios, dict):
            step_list = bottleneck_scenarios.get("scenarios", [])
        elif isinstance(bottleneck_scenarios, list):
            step_list = bottleneck_scenarios
        else:
            raise ValueError("Unexpected format from Scenarioagent scenarios")
        
        for idx, step in enumerate(step_list, start=1):
            adaptor = Modeladaptor(client)
            kpi_adapted_model = adaptor.adapter(original_code = clean_initial_model, instruction=step, final_path=final_path, multi_agent_setting= False, index_model= idx)
            print(kpi_adapted_model) # Append each adapted model's KPIs to results
            results.append(kpi_adapted_model)

        visualize_results(results, save_path=final_path)
        plt.show()

        return 

    def scenarios(self, bottleneck_machine, model_code, stations_table_md, sequence_text
        model = "gpt-5-mini",
        response_format={"type": "json_object"}): 
        prompt = (
            "Your task is to generate scenarios for testing a production line"
            f"This is the simulation model: \n {model_code}"
            f"Stations table: \n {stations_table_md}"
            f"Direct-follow relationsships: {sequence_text}"
            f"You should choose scenarios of interest depending on the bottleneck of the system which is: \n {bottleneck_machine} "
            "Specifically, you should answer with five separate scenarios: "
            "Improving the MTTR, availability and process time of the bottlneck, and increasing the buffer size of the buffers before and after the bottlneck machine"
            "Therefore, fomulate this scenarios as a JSON file based on the information of the production line provided"
            "All scenarios are to be easily describe and implemented"
            "Answer in JSON format like this: { 'scenarios': [ {'}, {'}, {'}, {'}, {'} ] }  ")


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
        )
        resp = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content