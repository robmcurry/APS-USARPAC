# run_batch_optimization.py

import pandas as pd
from disaster_logistics_model.network.network_builder import build_geospatial_network
from disaster_logistics_model.monte_carlo.simulator import generate_scenarios
from disaster_logistics_model.optimization.deterministic_model import solve_deterministic_vrp_with_aps

def run_batch_summary():
    # Step 1: Build network and load locations
    csv_path = "disaster_logistics_model/data/pacific_cities.csv"
    G, locations = build_geospatial_network(csv_path)

    # Step 2: Generate scenarios
    scenarios = generate_scenarios(G, locations)

    # Step 3: Run deterministic optimization for each scenario
    results = []
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        print(f"Solving scenario {scenario_id}...")
        result = solve_deterministic_vrp_with_aps(scenario)

        record = {
            "scenario_id": scenario_id,
            "scenario_type": scenario.get("scenario_type"),
            "epicenter": scenario.get("epicenter"),
            "impact_radius_km": scenario.get("impact_radius_km"),
            "infrastructure_loss_rate": scenario.get("infrastructure_loss_rate"),
            "geographic_tag": scenario.get("geographic_tag"),
            **result
        }

        results.append(record)

    # Step 4: Save results to CSV
    df = pd.DataFrame(results)
    output_path = "disaster_logistics_model/output/batch_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"Batch summary saved to {output_path}")

if __name__ == "__main__":
    run_batch_summary()