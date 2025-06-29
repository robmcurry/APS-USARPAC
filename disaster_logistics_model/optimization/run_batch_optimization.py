# run_batch_optimization.py

# run_batch_optimization.py

import pandas as pd
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage_commodity

# Step 1: Build network and load locations
csv_path = "disaster_logistics_model/data/pacific_cities.csv"
G, locations = build_geospatial_network(csv_path)

# Step 2: Generate scenarios
scenarios = generate_scenarios(G, locations)

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
df.to_csv("batch_summary.csv", index=False)
print("Batch summary saved as batch_summary.csv")
