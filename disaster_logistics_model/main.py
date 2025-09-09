from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model_single_stage import (
    solve_deterministic_vrp_with_aps_single_stage,
)
from disaster_logistics_model.visualization.generate_summary_charts import (
    generate_aps_frequency_chart,
    generate_objective_chart,
    generate_aps_selection_map
)
import pandas as pd
import os
import time

def main():
    # File paths and constants
    csv_path = "data/pacific_cities.csv"
    output_batch_summary = "output/batch_summary.csv"
    num_scenarios = 2  # Adjust as needed

    # Step 1: Build the network
    print("[1/4] Building the geospatial network...")
    G, locations = build_geospatial_network(csv_path)
    print(f"Network and locations loaded. {len(locations)} locations identified.")

    # Step 2: Generate scenarios
    print("[2/4] Generating disaster scenarios...")
    scenarios = generate_scenarios(G, locations, num_scenarios=num_scenarios)
    print(f"{len(scenarios)} scenarios generated.")

    # Step 3: Deterministic solution for the first scenario (optional)
    # print("[3/4] Optimizing first scenario (demo run)...")
    # start_time = time.time()
    # single_result = solve_deterministic_vrp_with_aps_single_stage(scenarios[0])
    # end_time = time.time()
    # print(f"Single scenario solved in {end_time - start_time:.2f} seconds.")

    # Step 4: Batch process all scenarios
    print("[4/4] Running batch optimization...")
    results = []
    for scenario in scenarios:
        print(f"Solving scenario {scenario['scenario_id']}...")
        res = solve_deterministic_vrp_with_aps_single_stage(scenario)
        results.append(res)

    # Save batch results to CSV
    os.makedirs(os.path.dirname(output_batch_summary), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_batch_summary, index=False)
    print(f"Batch summary saved at: {output_batch_summary}")


    # Set paths for input data and output directory
    batch_summary_path = "output/batch_summary.csv"
    cities_data_path = "data/pacific_cities.csv"
    output_dir = "output"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate APS Frequency Chart
    print("Generating APS Frequency Chart...")
    generate_aps_frequency_chart(batch_summary_path, output_dir)
    print("APS Frequency Chart saved!")

    # Step 2: Generate Objective Value Chart
    print("Generating Objective Value Chart...")
    generate_objective_chart(batch_summary_path, output_dir)
    print("Objective Value Chart saved!")

    print("Generating APS Selection Map...")
    generate_aps_selection_map(batch_summary_path, cities_data_path, output_dir)
    print("APS Selection Map saved!")

    # Step 3: Generate APS Selection Map
    # print("Generating APS Selection Map...")
    # generate_aps_selection_map(batch_summary_path, cities_data_path, output_dir)
    # print("APS Selection Map saved!")

    print("\nAll visuals generated successfully.")

    # Step 5: Generate summary charts
    # Step 5: Generate summary charts
    # if os.path.exists(output_batch_summary):
    #     print("[5/5] Generating summary charts...")
    #     generate_all_charts(output_batch_summary)
    #     print("Charts generated successfully.")
    # else:
    #     print("[5/5] Skipping chart generation - batch summary file not found at:", output_batch_summary)

print("\nProcess complete. All steps executed successfully.")

if __name__ == "__main__":
    main()