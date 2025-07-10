# control_tower.py

import os
import pandas as pd
from disaster_logistics_model.config.loader import load_parameters
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model_single_stage import solve_deterministic_vrp_with_aps_single_stage
from disaster_logistics_model.visualization.generate_summary_charts import generate_aps_frequency_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_objective_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_severity_vs_objective_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_3d_severity_objective_connectivity_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_interactive_3d_severity_objective_connectivity_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_aps_selection_map
from disaster_logistics_model.visualization.generate_summary_charts import generate_epicenter_objective_map
from disaster_logistics_model.visualization.generate_summary_charts import generate_composite_score_map
from disaster_logistics_model.visualization.generate_summary_charts import generate_scenario_tree_map


def run_full_pipeline():
    output_dir = "disaster_logistics_model/output/maps"

    print("Loading parameters...")
    params = load_parameters()

    # 1. Build network
    print("Building network...")
    network_params = params["network"]
    csv_path = network_params["location_file"]
    G, locations = build_geospatial_network(csv_path)

    # 2. Generate scenarios
    print("Generating disaster scenarios...")
    sim_params = params["simulation"]
    scenarios = generate_scenarios(G, locations, sim_params["num_scenarios"])

    # 3. Run optimization for each scenario
    print("Solving scenarios...")
    opt_params = params["optimization"]
    results = []
    for scenario in scenarios:
        print(f"Solving scenario {scenario['scenario_id']} (Type {scenario['scenario_type']}) with attributes {scenario['type_attributes']}")
        print(f"Solving scenario {scenario['scenario_id']}...")
        result = solve_deterministic_vrp_with_aps_single_stage(
            scenario,
            P_max=opt_params["P_max"],
            M=opt_params["M"]
        )
        results.append(result)

    # 4. Save output
    print("Saving batch summary...")
    output_file = params["output"]["batch_summary_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Batch summary saved to {output_file}")

    # 5. Generate visualizations
    print("Generating APS frequency chart...")
    generate_aps_frequency_chart(
        batch_summary_path=output_file,
        output_dir=output_dir
    )
    print("Generating objective value chart...")
    generate_objective_chart(
        batch_summary_path=output_file,
        output_dir=output_dir
    )
    print("Generating severity vs. objective chart...")
    generate_severity_vs_objective_chart(
        batch_summary_path=output_file,
        output_dir=output_dir
    )
    print("Generating 3D severity-objective-connectivity chart...")
    generate_3d_severity_objective_connectivity_chart(
        batch_summary_path=output_file,
        output_dir=output_dir
    )
    print("Generating interactive 3D severity-objective-connectivity chart...")
    generate_interactive_3d_severity_objective_connectivity_chart(
        batch_summary_path=output_file,
        output_dir=output_dir
    )
    print("Generating APS selection map...")
    generate_aps_selection_map(
        batch_summary_path=output_file,
        location_data=locations,
        output_path=os.path.join(output_dir, "aps_selection_map.html")
    )
    print("Generating epicenter objective map...")
    generate_epicenter_objective_map(
        batch_summary_path=output_file,
        location_data=locations,
        output_path=os.path.join(output_dir, "epicenter_objective_map.html")
    )
    #print("Generating composite strategy score map...")
    #generate_composite_score_map(
        #batch_summary_path=output_file,
        #location_data=locations,
        #output_path=os.path.join(output_dir, "composite_score_map.html")
    #)

    import json

    scenario_tree_path = os.path.join(output_dir, "tree_data_scenario_0.json")
    if os.path.exists(scenario_tree_path):
        with open(scenario_tree_path, "r") as f:
            tree_data = json.load(f)
        generate_scenario_tree_map(
            tree_data=tree_data,
            city_data_path=csv_path,
            output_path=os.path.join(output_dir, "scenario_tree_map.html")
        )
    else:
        print(f"Scenario tree data not found at {scenario_tree_path}")


    print("Generating disaster type impact charts...")
    generate_disaster_type_impact_charts(
        output_file,
        output_dir
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()