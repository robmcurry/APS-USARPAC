# run_batch_optimization.py

import pandas as pd
import os
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage_commodity

# Step 1: Build network and load locations
csv_path = "disaster_logistics_model/data/pacific_cities.csv"
G, locations = build_geospatial_network(csv_path)

# Step 2: Load pre-generated scenarios from JSON file
import json

json_path = os.path.join(os.path.dirname(__file__), "..", "monte_carlo", "simulation_output.json")
with open(json_path, "r") as f:
    scenarios = json.load(f)

import ast

def restore_dict(d):
    if isinstance(d, dict):
        return {tuple(ast.literal_eval(str(k))): v for k, v in d.items()}
    elif isinstance(d, list):
        return {tuple(k): v for k, v in d}
    else:
        raise ValueError("Unsupported format for restoration")

# Helper to convert all dict keys to strings for JSON serialization
def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    else:
        return obj

for scenario in scenarios:
    scenario['demand'] = restore_dict(scenario['demand'])
    scenario['supply'] = restore_dict(scenario['supply'])
    scenario['capacity'] = restore_dict(scenario['capacity'])
    scenario['params']['safety_stock'] = restore_dict(scenario['params']['safety_stock'])
    scenario['params']['node_capacity'] = restore_dict(scenario['params']['node_capacity'])

# Step 3: Run deterministic optimization for each scenario
import time

results = []
for scenario in scenarios:
    start_time = time.time()
    print(f"Solving scenario {scenario['scenario_id']}...")
    res = solve_deterministic_vrp_with_aps_single_stage(scenario)
    results.append(res)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

# Step 4: Save summary
df = pd.DataFrame(results)
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False)
print("Batch summary saved as batch_summary.csv")

# Step 5: Save full results (including tree data) to JSON
json_output_path = os.path.join(output_dir, "batch_detailed_results.json")
results_serializable = convert_keys_to_str(results)
with open(json_output_path, "w") as f:
    json.dump(results_serializable, f, indent=2)
print("Detailed results saved as batch_detailed_results.json")
