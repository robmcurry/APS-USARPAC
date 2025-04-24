# simulator.py
import random
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import math

from disaster_logistics_model.config.loader import load_parameters

def generate_scenarios(G: nx.Graph, locations: Dict[int, Dict], num_scenarios: int = None, seed: int = None) -> List[Dict]:
    params = load_parameters()

    if seed is None:
        seed = params.get("seed", None)
    if num_scenarios is None:
        num_scenarios = params.get("num_scenarios", 5)

    if seed is not None:
        random.seed(seed)

    scenarios = []
    node_ids = list(locations.keys())
    for s_id in range(num_scenarios):
        epicenter = random.choice(node_ids)
        severity = random.uniform(*params["default_disaster"]["severity_range"])

        base_radius = params["default_disaster"]["affected_radius"]["base"]
        multiplier = params["default_disaster"]["affected_radius"]["multiplier"]
        affected_radius = base_radius + int(severity * multiplier)

        affected_nodes = list(nx.single_source_shortest_path_length(G, epicenter, cutoff=affected_radius).keys())

        demand = {}
        supply = {}
        for i in affected_nodes:
            pop = 500 + int(1000 * severity)
            for c in params["commodities"]:
                demand[(i, c)] = pop * (0.2 if c == "food" else 0.6) * severity
                supply[(i, c)] = 0

        capacity = {}
        for (i, j) in G.edges():
            for c in params["commodities"]:
                base_cap = params["arc_capacity"]["base"] + params["arc_capacity"]["multiplier"] * severity
                capacity[(i, j, c)] = base_cap
                capacity[(j, i, c)] = base_cap

        scenarios.append({
            "scenario_id": s_id,
            "epicenter": epicenter,
            "severity": severity,
            "affected_nodes": affected_nodes,
            "demand": demand,
            "supply": supply,
            "capacity": capacity
        })

    return scenarios