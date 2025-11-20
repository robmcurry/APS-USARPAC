from network.network_builder import build_geospatial_network
from monte_carlo.simulator import generate_scenarios
from optimization.deterministic_model_single_stage import (
    solve_deterministic_vrp_with_aps_single_stage,
)
from visualization.generate_summary_charts import (
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
    num_scenarios = 100   # Adjust as needed

    # Step 1: Build the network
    print("[1/4] Building the geospatial network...")
    max_days = 4  # maximum allowed travel days
    G, locations, arc_days = build_geospatial_network(csv_path, max_days=max_days)
    print(f"Network and locations loaded. {len(locations)} locations identified.")

    # Step 2: Generate scenarios
    print("[2/4] Generating disaster scenarios...")
    scenarios = generate_scenarios(G, locations, num_scenarios=num_scenarios)
    print(f"{len(scenarios)} scenarios generated.")

    # Step 3: Deterministic solution for the first scenario (optional)
    # print("[3/4] Optimizing first scenario (demo run)...")
    # start_time = time.time()
    # single_result = solve_deterministic_vrp_with_aps_single_stage(scenarios[0], locations)
    # end_time = time.time()
    # print(f"Single scenario solved in {end_time - start_time:.2f} seconds.")


    node_list = sorted(locations.keys())
    base_num_vehicles = 300
    base_safety_stock_value = 13500000

    global commodity_list
    for scenario in scenarios:
        commodity_list = sorted(set(c for (_, c) in scenario['demand']))

    # base_redundancy = {int(i): 3 for i in node_list}  # Defines redundancy requirements for each node
    base_L = {c: 3 for c in commodity_list}
    # base_num_APS - 5
    num_aps=5

    # Step 4: Batch process all scenarios
    print("[4/4] Running batch optimization...")

    for k in [1,2,3,4]:
        base_redundancy = {int(i): k for i in node_list}
        results = []
        for scenario in scenarios:
            print(f"Solving scenario {scenario['scenario_id']}...")
            res = solve_deterministic_vrp_with_aps_single_stage(scenario, locations, num_aps, base_redundancy, base_L)
            results.append(res)

    # for num_aps in [3,4,5,6]:
    #     results = []
    #     for scenario in scenarios:
    #         print(f"Solving scenario {scenario['scenario_id']}...")
    #         res = solve_deterministic_vrp_with_aps_single_stage(scenario, locations, num_aps, base_redundancy, base_L)
    #         results.append(res)

        output_dir = f"output/Redun/Redun_{k}"
        os.makedirs(output_dir, exist_ok=True)
        output_batch_summary_Redun = os.path.join(output_dir, f"batch_summary_Redun_{k}.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_batch_summary_Redun, index=False)
        print(f"Batch summary saved at: {output_batch_summary_Redun}")


    cities_data_path = locations


    for k in [1, 2, 3, 4]:
        base_redundancy = {int(i): k for i in node_list}
        batch_summary_path = os.path.join(f"output/Redun/Redun_{k}", f"batch_summary_Redun_{k}.csv")
        output_dir = f"output/Redun/Redun_{k}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating visuals for Redun = {k}...")

        generate_aps_frequency_chart(batch_summary_path, output_dir)
        print("APS Frequency Chart saved!")

        generate_objective_chart(batch_summary_path, output_dir)
        print("Objective Value Chart saved!")

        generate_aps_selection_map(batch_summary_path, cities_data_path, os.path.join(output_dir, "aps_selection_map.html"))
        print("APS Selection Map saved!")

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