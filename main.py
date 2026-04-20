
import sys
from matplotlib import path, pyplot as plt
from openai import OpenAI
from processmining import eventlog, metrics
from agents.builder import ModelBuilder
from agents.optimizer import Blueprintoptimizer
from agents.adapter import Modeladaptor
from agents.evaluator import Evaluater
from agents.cpdagent import CPD
from agents.visualizer import Modelvisualizer
from agents.bottleneck import BottleneckOptimizer
from helpers.other_helpers import save_model, remove_code_wrappers, retrieve_KPIs, visualize_results, inspect_code
from helpers.mermaid_renderer import render_mermaid_to_png
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

api_key= os.getenv("OPENAI_KEY")

client = OpenAI(api_key=api_key)
file_path_eventlog = Path("data/workingtest.csv")
file_path_blueprintmodel_util = Path("blueprint/blueprint_util.py")
file_path_blueprintmodel_MOO_buffer = Path("blueprint/blueprint_MOO_buffer.py")
file_path_blueprintmodel_MOO_availability = Path("blueprint/blueprint_MOO_availability.py")
file_path_blueprintmodel_SCORE = Path("blueprint/blueprint_SCORE.py")
final_path = Path("results")
buffers_info_specific = "PostLoadingBuffer(Capacity = 2, processtime = 10), PostConveyorBuffer(Capacity = 2, processtime = 10), PostWashingBuffer(Capacity = 2, processtime = 10), PrePress1Buffer(Capacity = 3, processtime = 32), PrePress2Buffer(Capacity = 3, processtime = 32), " \
"PostPress1&Press2Buffer(Capacity = 3, processtime = 32)"
defect_info = "Defect rate = 0.089, defect sink = defect,initiated at Qualitystation"
cpd_info = "1. The presses need to have a processtime of at least 60s. " # All buffer capacities must be kept at the same original level.

def main() -> None:
    df_raw = eventlog.load(file_path_eventlog)
    df_clean = eventlog.preprocess(df_raw)
    stations_md = metrics.compute(df_clean).to_string(index=False)
    sequence_text = eventlog.to_sequence_text(df_clean)
    print(stations_md)
    print("")
    print(sequence_text)

    # read-in the blueprints
    blueprint_code = open(file_path_blueprintmodel_util, "r", encoding="utf-8").read()
    MOO_blueprint_buffer = open(file_path_blueprintmodel_MOO_buffer, "r", encoding="utf-8").read()
    MOO_blueprint_availability = open(file_path_blueprintmodel_MOO_availability, "r", encoding="utf-8").read()
    SCORE_blueprint = open(file_path_blueprintmodel_SCORE, "r", encoding="utf-8").read()

    # Build initial model
    builder = ModelBuilder(client)
    model_code = builder.build(
        blueprint_code=blueprint_code,
        stations_table_md=stations_md,
        sequence_text=sequence_text,
        buffers=buffers_info_specific,  
        defects=defect_info,  
        manual_note="Production stops from friday 17.00 till saturday 07:00 and from saturday 17:00 till sunday 07:00.")

    clean_initial_model = remove_code_wrappers(model_code)
    save_model(clean_initial_model,final_path, "initial_model.py")
    init_model_path = os.path.join(final_path,"initial_model.py")

    print("Inspecting initial model for errors...")

    inspected_initial_model = inspect_code(clean_initial_model, client, "initial_model.py")
    clean_inspected_initial_model = remove_code_wrappers(inspected_initial_model)

    # Manual adaptation
    manual = input("Do you want to manually edit the initial model before proceeding? (y/n): ").strip().lower()
    if manual == 'y':
        print(f" Please open and edit:\n  {init_model_path}\n When you're done, save it and press Enter.")
        input()  # wait for user confirmation
        # reload their edits
        with open(init_model_path, 'r', encoding='utf-8') as f:
            clean_inspected_initial_model = f.read()
        print("Loaded your manually adapted model.")

    # Visualize initial model into flow chart
    visualizer = Modelvisualizer(client)
    mermaid_code = visualizer.visualize_agent(clean_inspected_initial_model)
    mmd_path = os.path.join(final_path, "model_visualization.mmd")
    png_path = os.path.join(final_path, "model_visualization.png")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    render_mermaid_to_png(mmd_path, png_path, client)
    print(f"Flow chart saved to: {png_path}")
    

    print("\nEvaluating initial model...")

    kpi_original = retrieve_KPIs(clean_inspected_initial_model, "Original model")

    results = []
    results.append(kpi_original)
    print(kpi_original)


    print("\n" + "="*33)
    print("---OPTIONAL OPTIMIZATION STAGE---")
    print("="*33)
    
    optimize_choice = input("Press 1 to find system bottlenecks (by the SCORE method). \nPress 2 to run specific MOO optimizations (e.g. buffer configurations etc.). \nPress 3 to exit. \nAnswer: ").strip()
        
    while True:
        if optimize_choice == "1":
            optimizer = BottleneckOptimizer(client)
            suggestions = optimizer.optimize(model_code = clean_inspected_initial_model, SCORE_blueprint=SCORE_blueprint)
            _ = 
            break
        elif optimize_choice == "2":
            optimizer = Blueprintoptimizer(client)
            suggestions = optimizer.optimize(
                model_code = clean_inspected_initial_model,
                MOO_blueprint_buffer = MOO_blueprint_buffer,
                MOO_blueprint_availability = MOO_blueprint_availability
            )
            break
        elif optimize_choice == "3":
            print("Exiting program.")
            sys.exit()  
        else:
            print("Invalid choice. Choose between 1, 2, and 3: ")
    
    
    print("back in main.py")

    if isinstance(suggestions, dict):
        step_list = suggestions.get("instructions", [])
    elif isinstance(suggestions, list):
        step_list = suggestions
    else:
        raise ValueError("Unexpected format from optimizer suggestions")
    
    human_input = input("Do you want to manually add one change to the model? If yes please answer with the change (leave blank to skip): ").strip().lower()
    if human_input:
        print("Added human input.")
        step_list.append(human_input)
    
    cpdexpert = CPD(client)
    
    print(cpdexpert.evaluatecpd(step_list, cpd_info))
    
    for idx, step in enumerate(step_list, start=1):
        adaptor = Modeladaptor(client)
        kpi_adapted_model = adaptor.adapter(original_code = clean_initial_model, instruction=step, final_path=final_path, multi_agent_setting= False, index_model= idx)
        print(kpi_adapted_model) # Append each adapted model's KPIs to results
        results.append(kpi_adapted_model)

    # Evaluate all results
    evaluator = Evaluater(client)
    print(evaluator.evaluate(results))

    visualize_results(results, save_path=final_path)
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")
