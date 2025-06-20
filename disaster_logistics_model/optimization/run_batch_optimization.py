# run_batch_optimization.py

import pandas as pd
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model import solve_deterministic_vrp_with_aps
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage

# Step 1: Build network and load locations
csv_path = "disaster_logistics_model/data/pacific_cities.csv"
G, locations = build_geospatial_network(csv_path)

# Step 2: Generate scenarios
scenarios = generate_scenarios(G, locations)

# Step 3: Run deterministic optimization for each scenario
results = []
for scenario in scenarios:
    print(f"Solving scenario {scenario['scenario_id']}...")
    # res = solve_deterministic_vrp_with_aps(scenario)
    res = solve_deterministic_vrp_with_aps_single_stage(scenario)
    results.append(res)

# Step 4: Save summary
df = pd.DataFrame(results)
df.to_csv("batch_summary.csv", index=False)
print("Batch summary saved as batch_summary.csv")