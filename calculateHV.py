import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV


# 1. Load the CSV file
df_framework = pd.read_csv("moo_simulation_results.csv")

# 2. Extract objectives (assuming last two columns)
# obj1: Minimize (WIP)
# obj2: Maximize (Throughput)
points_framework = df_framework.iloc[:, -2:].values.astype(float)

# 3. Transform Maximization to Minimization
# We multiply the throughput column (index 1) by -1
points_framework[:, 1] = points_framework[:, 1] * -1
#print(points_framework[:5])


#facts point
df_facts = pd.read_csv('Factsresults_fixed.csv')
#grab second and third column without column names
points_facts = df_facts.iloc[:, 1:3].apply(pd.to_numeric, errors='coerce').dropna().values.astype(float)
#change places of the two colums
points_facts = points_facts[:, [1, 0]]
# Transform Maximization to Minimization for facts as well
points_facts[:, 1] = points_facts[:, 1] * -1
#print(points_facts[:5])







def calculate_hypervolume(points_framework, points_facts):

    
    combined_points = np.vstack((points_framework, points_facts))
    
    ref1 = combined_points[:, 0].max() * 1.1
    ref2 = combined_points[:, 1].max() * 0.9

    reference_point = np.array([ref1, ref2])
    
    print(f"Transformed Reference Point (Minimization Space): {reference_point}")
    
    # 5. Calculate Hypervolume
    hv_indicator = HV(ref_point=reference_point)
    hv_value_framework = hv_indicator(points_framework)
    hv_value_facts = hv_indicator(points_facts)

    print(f"Hypervolume Framework: {hv_value_framework:.4f}")
    print(f"Hypervolume Facts: {hv_value_facts:.4f}")

# Example usage
# Pass the actual arrays, not the filenames!
calculate_hypervolume(points_framework, points_facts)
