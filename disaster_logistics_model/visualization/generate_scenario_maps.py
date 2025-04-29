# disaster_logistics_model/visualization/generate_scenario_maps.py

import os
import pandas as pd
import json
from disaster_logistics_model.visualization.flow_map import plot_full_scenario_map


def main():
    batch_summary = pd.read_csv('disaster_logistics_model/output/batch_summary.csv')
    scenario_folder = 'disaster_logistics_model/output/scenarios'
    maps_folder = 'disaster_logistics_model/output/maps'

    os.makedirs(maps_folder, exist_ok=True)

    for _, row in batch_summary.iterrows():
        scenario_id = row['scenario_id']
        scenario_file = os.path.join(scenario_folder, f"scenario_{scenario_id}.json")

        if os.path.exists(scenario_file):
            with open(scenario_file, 'r') as f:
                scenario = json.load(f)

            # Dummy aps_result just for testing, if you want to load real APS results we can adjust later
            aps_result = {
                "aps_locations": eval(row['aps_locations']) if isinstance(row['aps_locations'], str) else [],
                "flow_summary": {},  # No full flow info yet unless you separately save it
            }

            output_file = os.path.join(maps_folder, f"scenario_{scenario_id}_map.html")
            plot_full_scenario_map(scenario, aps_result, output_file)
            print(f"Saved {output_file}")


if __name__ == "__main__":
    main()