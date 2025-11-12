# simulator.py
import random
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import math
from geopy.distance import geodesic
import json  # For serialization of complex data to JSON strings
import os

from config.loader import load_parameters

def generate_scenarios(G: nx.Graph, locations: Dict[int, Dict], num_scenarios: int = None, seed: int = None, save_path: str = None) -> List[Dict]:
    """
    Generate a list of disaster scenarios based on a graph of locations and their attributes.

    Each scenario simulates a disaster epicenter, severity, affected nodes, and resulting demand, supply,
    and capacity constraints on arcs and nodes.

    Returns a list of scenario dictionaries, each containing:
        - scenario_id: unique identifier
        - epicenter: node where disaster originates
        - severity: disaster severity scalar
        - affected_nodes: nodes within the affected radius
        - demand: demand per node and commodity
        - supply: baseline supply adjusted by severity
        - capacity: arc capacities adjusted by severity
        - aps_capacity: available pre-positioned stock capacity per node and commodity
        - node_capacity: steady-state node storage capacity (days of supply × population × per-capita need)
        - available_node_capacity: degraded node storage capacity post-disaster
        - affected_radius_km: radius in kilometers affected by the disaster
    """
    # --- Parameter setup and seeding ---
    # Load parameters from configuration
    params = load_parameters()

    # Use provided seed or fallback to default in parameters or 42
    if seed is None:
        seed = params.get("seed", 42)
    random.seed(seed)
    print(f"[Simulator] Using seed: {seed}")

    # Determine number of scenarios to generate
    if num_scenarios is None:
        num_scenarios = params.get("num_scenarios", 100) # default 50 scenarios if none given in params

    scenarios = []
    print(locations)
    node_ids = list(locations.keys())

    for s_id in range(num_scenarios):
        # --- Disaster epicenter and severity ---
        # Randomly select epicenter node for disaster
        epicenter = random.choice(node_ids)
        # Randomly sample severity within configured range
        severity = random.uniform(*params["default_disaster"]["severity_range"])

        # Calculate affected radius based on severity and configured NumAPS and multiplier (in kilometers)
        base_radius_km = params["default_disaster"]["affected_radius_km"]["base"]
        multiplier_km = params["default_disaster"]["affected_radius_km"]["multiplier"]
        affected_radius_km = base_radius_km + severity * multiplier_km

        # --- Determine affected nodes and node-specific severity based on geographical distance ---
        epicenter_coords = None
        if "coords" in locations[epicenter]:
            epicenter_coords = tuple(locations[epicenter]["coords"])
        else:
            epicenter_lat = locations[epicenter].get("lat", locations[epicenter].get("Latitude"))
            epicenter_lon = locations[epicenter].get("lon", locations[epicenter].get("Longitude"))
            epicenter_coords = (epicenter_lat, epicenter_lon)
        epicenter_pop = locations[epicenter].get("pop", locations[epicenter].get("Population"))
        affected_nodes = []
        node_severity = {}
        for i in node_ids:
            if "coords" in locations[i]:
                node_coords = tuple(locations[i]["coords"])
            else:
                node_lat = locations[i].get("lat", locations[i].get("Latitude"))
                node_lon = locations[i].get("lon", locations[i].get("Longitude"))
                node_coords = (node_lat, node_lon)
            node_pop = locations[i].get("pop", locations[i].get("Population"))
            dist_km = geodesic(epicenter_coords, node_coords).kilometers
            if dist_km <= affected_radius_km:
                affected_nodes.append(i)
                node_severity[i] = max(0.0, severity * (1.0 - dist_km / affected_radius_km))
            else:
                node_severity[i] = 0.0

        # --- Demand and supply generation ---
        demand = {}
        supply = {}
        for i in affected_nodes:
            pop = locations[i].get("pop", locations[i].get("Population"))
            for c in params["commodities"]:
                # Demand proportional to population and severity at node
                demand[(i, c)] = pop * node_severity[i]
                # Baseline supply degraded by severity (less supply available in more severely affected nodes)
                baseline_supply = locations[i].get("baseline_supply", {}).get(c, 0.0)
                supply[(i, c)] = max(0.0, baseline_supply * (1 - node_severity[i]))

        # --- Arc capacity (permissive; arcs act as connectivity only) ---
        capacity = {}
        cap_bigM = params.get("arc_capacity", {}).get("big_M", 10**9)  # configurable in YAML
        for (i, j) in G.edges():
            for c in params["commodities"]:
                capacity[(i, j, c)] = cap_bigM
                capacity[(j, i, c)] = cap_bigM  # symmetric

        # --- APS capacity (uniform per node/commodity) ---
        # APS = Available Pre-positioned Stock capacity, uniform across nodes and commodities unless overridden
        aps_capacity = {}
        for i in node_ids:
            for c in params["commodities"]:
                aps_capacity[(i, c)] = params.get("aps_capacity", {}).get(c, 10000)

        # --- Steady-state node storage capacity ---
        # Storage capacity based on days of supply, population, and per-capita daily need (nu)
        nu_water = params.get("nu_water", 3.0)  # liters/person/day for water
        nu_food  = params.get("nu_food", 1.0)   # rations/person/day for food
        dos_water = params.get("node_capacity_days", {}).get("water", 3)  # days of supply for water
        dos_food  = params.get("node_capacity_days", {}).get("food", 7)   # days of supply for food
        nu = {"water": nu_water, "food": nu_food}
        dos = {"water": dos_water, "food": dos_food}
        node_capacity = {}
        for i in node_ids:
            pop_i = locations[i].get("pop", locations[i].get("Population"))
            for c in params["commodities"]:
                # Capacity is population × per-capita need × days of supply
                node_capacity[(i, c)] = pop_i * nu.get(c, 1.0) * dos.get(c, 7)

        # --- Degraded node storage capacity (post-disaster) ---
        # Node capacity degraded by node-specific severity and configurable degradation factors per commodity
        deg = {
            "water": params.get("node_capacity_degradation", {}).get("water", 0.3),
            "food":  params.get("node_capacity_degradation", {}).get("food", 0.3),
        }
        available_node_capacity = {}
        for i in node_ids:
            S_i = node_severity.get(i, 0.0)
            for c in params["commodities"]:
                # Factor reduces capacity based on severity and degradation factor
                factor = max(0.0, 1.0 - deg.get(c, 0.3) * S_i)
                available_node_capacity[(i, c)] = node_capacity[(i, c)] * factor

        # --- Build scenario artifact ---
        scenarios.append({
            "scenario_id": s_id,
            "epicenter": epicenter,
            "severity": severity,
            "affected_nodes": affected_nodes,
            "demand": demand,
            "supply": supply,
            "capacity": capacity,
            "aps_capacity": aps_capacity,
            "node_capacity": node_capacity,
            "available_node_capacity": available_node_capacity,
            "affected_radius_km": affected_radius_km,
        })

    # --- Always export scenarios to output/scenarios.csv ---
    def normalize_dict_keys(d):
        normalized = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                if len(k) == 2:
                    key_str = f"node:{k[0]},commodity:{k[1]}"
                elif len(k) == 3:
                    key_str = f"node1:{k[0]},node2:{k[1]},commodity:{k[2]}"
                else:
                    key_str = ",".join(str(x) for x in k)
            else:
                key_str = str(k)
            normalized[key_str] = v
        return normalized

    output_file = os.path.join("output", "scenarios.csv")
    scenario_rows = []
    for scenario in scenarios:
        epicenter_id = scenario["epicenter"]
        epicenter_name = locations.get(epicenter_id, {}).get("name", locations.get(epicenter_id, {}).get("Name", "Unknown"))
        row = {
            "scenario_id": scenario["scenario_id"],
            "epicenter": epicenter_id,
            "epicenter_name": epicenter_name,
            "severity": scenario["severity"],
            "affected_radius_km": scenario["affected_radius_km"],
            "affected_nodes": json.dumps(scenario["affected_nodes"]),
            "demand": json.dumps(normalize_dict_keys(scenario["demand"])),
            "supply": json.dumps(normalize_dict_keys(scenario["supply"])),
            "capacity": json.dumps(normalize_dict_keys(scenario["capacity"])),
            "aps_capacity": json.dumps(normalize_dict_keys(scenario["aps_capacity"])),
            "node_capacity": json.dumps(normalize_dict_keys(scenario["node_capacity"])),
            "available_node_capacity": json.dumps(normalize_dict_keys(scenario["available_node_capacity"]))
        }
        scenario_rows.append(row)
    df = pd.DataFrame(scenario_rows)
    df.to_csv(output_file, index=False)
    print(f"[Simulator] Scenarios saved to {output_file}")

    return scenarios
