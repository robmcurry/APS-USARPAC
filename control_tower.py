# control_tower.py

import os
import pandas as pd
from disaster_logistics_model.config.loader import load_parameters
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model import solve_deterministic_vrp_with_aps
from disaster_logistics_model.visualization.generate_summary_charts import generate_aps_frequency_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_objective_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_severity_vs_objective_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_3d_severity_objective_connectivity_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_interactive_3d_severity_objective_connectivity_chart
from disaster_logistics_model.visualization.generate_summary_charts import generate_aps_selection_map
from disaster_logistics_model.visualization.generate_summary_charts import generate_epicenter_objective_map
from disaster_logistics_model.visualization.generate_summary_charts import generate_composite_score_map


def run_full_pipeline():
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
        result = solve_deterministic_vrp_with_aps(
            scenario,
            time_periods=range(1, opt_params["time_periods"] + 1),
            vehicle_list=opt_params["vehicle_list"],
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
        output_dir=os.path.dirname(output_file)
    )
    print("Generating objective value chart...")
    generate_objective_chart(
        batch_summary_path=output_file,
        output_dir=os.path.dirname(output_file)
    )
    print("Generating severity vs. objective chart...")
    generate_severity_vs_objective_chart(
        batch_summary_path=output_file,
        output_dir=os.path.dirname(output_file)
    )
    print("Generating 3D severity-objective-connectivity chart...")
    generate_3d_severity_objective_connectivity_chart(
        batch_summary_path=output_file,
        output_dir=os.path.dirname(output_file)
    )
    print("Generating interactive 3D severity-objective-connectivity chart...")
    generate_interactive_3d_severity_objective_connectivity_chart(
        batch_summary_path=output_file,
        output_dir=os.path.dirname(output_file)
    )
    print("Generating APS selection map...")
    generate_aps_selection_map(
        batch_summary_path=output_file,
        location_data=locations,
        output_path=os.path.join(os.path.dirname(output_file), "aps_selection_map.html")
    )
    print("Generating epicenter objective map...")
    generate_epicenter_objective_map(
        batch_summary_path=output_file,
        location_data=locations,
        output_path=os.path.join(os.path.dirname(output_file), "epicenter_objective_map.html")
    )
    print("Generating composite strategy score map...")
    generate_composite_score_map(
        batch_summary_path=output_file,
        location_data=locations,
        output_path=os.path.join(os.path.dirname(output_file), "composite_score_map.html")
    )

    from disaster_logistics_model.visualization.generate_summary_charts import generate_disaster_type_impact_charts

    print("Generating disaster type impact charts...")
    generate_disaster_type_impact_charts(
        output_file,
        os.path.dirname(output_file)
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()