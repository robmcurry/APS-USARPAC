import json
import os
import random
from typing import Dict, List

import networkx as nx
import pandas as pd
from geopy.distance import geodesic

from stoch_loader import load_parameters


def _get_coords(location: Dict) -> tuple[float, float]:
    """
    Return (lat, lon) coordinates from a location record.
    """
    if "coords" in location and location["coords"] is not None:
        coords = tuple(location["coords"])
        return float(coords[0]), float(coords[1])

    lat = location.get("lat", location.get("Latitude"))
    lon = location.get("lon", location.get("Longitude"))
    return float(lat), float(lon)


def _normalize_dict_keys(d: Dict) -> Dict[str, float]:
    """
    Convert tuple or scalar keys to strings so nested dictionaries can be exported to CSV.
    """
    normalized = {}
    for k, v in d.items():
        normalized[str(k)] = v
    return normalized


def generate_scenarios(
    G: nx.Graph,
    locations: Dict[int, Dict],
    num_scenarios: int = None,
    seed: int = None,
    save_path: str = None,
) -> List[Dict]:
    """
    Generate disaster scenarios for the stochastic course project.

    This simulator now produces only the exogenous scenario realization needed
    to model arc-capacity uncertainty later in the input builder.

    Scenario structure returned:
        - scenario_id
        - epicenter
        - severity
        - affected_nodes
        - node_severity
        - affected_radius_km
        - probability

    Notes:
        - Demand is intentionally NOT generated here.
        - Storage capacity is intentionally NOT generated here.
        - Residual arc capacity is intentionally NOT generated here.
        - Those fixed and model-dependent quantities will be built later in
          stochastic_input_builder.py so that arc-capacity uncertainty can be
          isolated cleanly through gamma.
    """
    del G  # graph topology is not needed at the scenario-generation stage

    params = load_parameters()

    if seed is None:
        seed = params.get("seed", 42)
    random.seed(seed)
    print(f"[Stoch Simulator] Using seed: {seed}")

    if num_scenarios is None:
        num_scenarios = params.get("num_scenarios", 100)

    node_ids = list(locations.keys())
    scenarios: List[Dict] = []

    for s_id in range(num_scenarios):
        epicenter = random.choice(node_ids)
        severity = random.uniform(*params["default_disaster"]["severity_range"])

        base_radius_km = params["default_disaster"]["affected_radius_km"]["base"]
        multiplier_km = params["default_disaster"]["affected_radius_km"]["multiplier"]
        affected_radius_km = base_radius_km + severity * multiplier_km

        epicenter_coords = _get_coords(locations[epicenter])

        affected_nodes = []
        node_severity = {}

        for i in node_ids:
            node_coords = _get_coords(locations[i])
            dist_km = geodesic(epicenter_coords, node_coords).kilometers

            if dist_km <= affected_radius_km:
                local_severity = max(0.0, severity * (1.0 - dist_km / affected_radius_km))
                affected_nodes.append(i)
            else:
                local_severity = 0.0

            node_severity[i] = local_severity

        scenario = {
            "scenario_id": s_id,
            "epicenter": epicenter,
            "severity": severity,
            "affected_nodes": affected_nodes,
            "node_severity": node_severity,
            "affected_radius_km": affected_radius_km,
            "probability": 1.0 / num_scenarios,
        }
        scenarios.append(scenario)

    output_file = save_path if save_path is not None else os.path.join("output", "stoch_scenarios.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    scenario_rows = []
    for scenario in scenarios:
        epicenter_id = scenario["epicenter"]
        epicenter_name = locations.get(epicenter_id, {}).get(
            "name",
            locations.get(epicenter_id, {}).get("Name", "Unknown"),
        )
        row = {
            "scenario_id": scenario["scenario_id"],
            "epicenter": epicenter_id,
            "epicenter_name": epicenter_name,
            "severity": scenario["severity"],
            "affected_radius_km": scenario["affected_radius_km"],
            "affected_nodes": json.dumps(scenario["affected_nodes"]),
            "node_severity": json.dumps(_normalize_dict_keys(scenario["node_severity"])),
            "probability": scenario["probability"],
        }
        scenario_rows.append(row)

    df = pd.DataFrame(scenario_rows)
    df.to_csv(output_file, index=False)
    print(f"[Stoch Simulator] Scenarios saved to {output_file}")

    return scenarios
