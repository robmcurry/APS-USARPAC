import random
import math
import networkx as nx
from disaster_logistics_model.config.loader import load_parameters


def generate_scenarios(G, locations, num_scenarios=None):
    """
    Generate disaster scenarios based on the input graph and location data.

    Args:
        G (networkx.Graph): Graph representing locations and their connections.
        locations (dict): Dictionary with location coordinates and other info.
        num_scenarios (int, optional): Number of scenarios to generate. If None, uses default from parameters.

    Returns:
        list: A list of scenario dictionaries containing details such as epicenter, affected nodes, demand, supply, and parameters.
    """
    # Load parameters for simulation, commodities, vehicles, and APS
    params = load_parameters()
    sim_params = params["simulation"]
    commodity_size = params["commodities"]["size"]
    vehicle_capacity = params["vehicles"]["capacity"]
    min_APS_per_commodity = params["aps"]["min_per_commodity"]
    default_safety_stock = params["simulation"]["safety_stock"]
    default_node_capacity = params["simulation"]["node_capacity"]

    # Define profiles for different scenario types with their characteristic distributions and tags
    scenario_profiles = {
        0: {
            "impact_radius_km": lambda: random.gauss(500, 100),
            "infrastructure_loss_rate": lambda: random.uniform(0.2, 0.5),
            "geographic_tag": lambda: "coastal"
        },
        1: {
            "impact_radius_km": lambda: random.gauss(800, 200),
            "infrastructure_loss_rate": lambda: random.uniform(0.1, 0.3),
            "geographic_tag": lambda: "island_chain"
        },
        2: {
            "impact_radius_km": lambda: random.gauss(300, 50),
            "infrastructure_loss_rate": lambda: random.uniform(0.4, 0.7),
            "geographic_tag": lambda: "urban_cluster"
        }
    }

    # Determine the number of scenarios to generate
    if num_scenarios is None:
        num_scenarios = sim_params["num_scenarios"]

    scenarios = []
    node_list = list(G.nodes())

    for sid in range(num_scenarios):
        # Select an epicenter node and scenario type randomly
        epicenter = random.choice(node_list)
        scenario_type = random.choice(list(scenario_profiles.keys()))
        profile = scenario_profiles[scenario_type]

        # Sample scenario-specific attributes
        impact_radius_km = profile["impact_radius_km"]()
        loss_rate = profile["infrastructure_loss_rate"]()
        geo_tag = profile["geographic_tag"]()

        # Calculate number of neighbors within impact radius
        num_neighbors = sum(
            1 for neighbor in G.neighbors(epicenter)
            if haversine(locations[epicenter]["coords"], locations[neighbor]["coords"]) <= impact_radius_km
        )

        # Sample severity of the scenario
        severity = max(0.1, random.gauss(
            sim_params["severity_distribution"]["mean"],
            sim_params["severity_distribution"]["std_dev"]
        ))

        # Identify affected nodes within the impact radius
        affected_nodes = [
            node for node in node_list
            if haversine(locations[epicenter]["coords"], locations[node]["coords"]) <= impact_radius_km
        ]

        # Calculate demand and supply for affected nodes
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

        # Set capacities for edges in the graph
        capacity = {}
        for (i, j) in G.edges():
            capacity[(i, j, "food")] = 1e6
            capacity[(i, j, "water")] = 1e6

        # Define safety stock and node capacity for affected nodes
        safety_stock = {(node, c): default_safety_stock for node in affected_nodes for c in ["food", "water"]}
        node_capacity = {(node, t): default_node_capacity for node in affected_nodes for t in range(1, 6)}

        # Assemble the scenario dictionary
        scenario = {
            "scenario_id": sid,
            "epicenter": epicenter,
            "epicenter_neighbors": num_neighbors,
            "severity": severity,
            "affected_nodes": affected_nodes,
            "demand": demand,
            "supply": supply,
            "capacity": capacity,
            "scenario_type": scenario_type,
            "type_attributes": {
                "impact_radius_km": impact_radius_km,
                "infrastructure_loss_rate": loss_rate,
                "geographic_tag": geo_tag
            },
            "params": {
                "commodity_size": commodity_size,
                "vehicle_capacity": vehicle_capacity,
                "safety_stock": safety_stock,
                "min_APS_per_commodity": min_APS_per_commodity,
                "node_capacity": node_capacity
            }
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