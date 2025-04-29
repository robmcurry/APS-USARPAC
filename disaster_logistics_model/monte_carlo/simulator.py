import random
import math
import networkx as nx
from disaster_logistics_model.config.loader import load_parameters


def generate_scenarios(G, locations, num_scenarios=None):
    params = load_parameters()
    sim_params = params["simulation"]

    if num_scenarios is None:
        num_scenarios = sim_params["num_scenarios"]

    scenarios = []
    node_list = list(G.nodes())

    for sid in range(num_scenarios):
        epicenter = random.choice(node_list)
        severity = max(0.1, random.gauss(
            sim_params["severity_distribution"]["mean"],
            sim_params["severity_distribution"]["std_dev"]
        ))

        affected_nodes = [
            node for node in node_list
            if haversine(locations[epicenter]["coords"], locations[node]["coords"]) <= sim_params["disaster_radius_km"]
        ]

        demand = {}
        supply = {}
        for node in affected_nodes:
            pop = G.nodes[node]["population"]
            food_demand = pop * sim_params["base_demand_food_per_capita"] * severity
            water_demand = pop * sim_params["base_demand_water_per_capita"] * severity
            demand[(node, "food")] = food_demand
            demand[(node, "water")] = water_demand

            food_supply = pop * sim_params["base_demand_food_per_capita"] * sim_params["supply_fraction"]
            water_supply = pop * sim_params["base_demand_water_per_capita"] * sim_params["supply_fraction"]
            supply[(node, "food")] = food_supply
            supply[(node, "water")] = water_supply

        capacity = {}
        for (i, j) in G.edges():
            capacity[(i, j, "food")] = 1e6
            capacity[(i, j, "water")] = 1e6

        scenario = {
            "scenario_id": sid,
            "epicenter": epicenter,
            "severity": severity,
            "affected_nodes": affected_nodes,
            "demand": demand,
            "supply": supply,
            "capacity": capacity
        }
        scenarios.append(scenario)

    return scenarios


def haversine(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))