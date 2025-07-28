import random
import math
import networkx as nx
import json
import os


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
    params = {
        "simulation": {
            "num_scenarios": 5,
            "base_demand_food_per_capita": 0.3,
            "base_demand_water_per_capita": 0.5,
            "supply_fraction": {"value": 1.0},
            "initial_supply": {"surplus_factor": 1.2},
            "safety_days": 3,
            "capacity_per_capita": 1.5,
            "node_capacity": {"base": 1.0, "surplus_factor": 0.002},
            "severity_distribution": {"mean": 1.0, "std_dev": 0.5},
            "scenario_profiles": {
                "coastal": {
                    "impact_radius_mean_km": 500,
                    "impact_radius_std_km": 100,
                    "infrastructure_loss_min": 0.2,
                    "infrastructure_loss_max": 0.6
                },
                "island_chain": {
                    "impact_radius_mean_km": 700,
                    "impact_radius_std_km": 150,
                    "infrastructure_loss_min": 0.3,
                    "infrastructure_loss_max": 0.7
                },
                "urban_cluster": {
                    "impact_radius_mean_km": 300,
                    "impact_radius_std_km": 80,
                    "infrastructure_loss_min": 0.1,
                    "infrastructure_loss_max": 0.5
                }
            }
        },
        "commodities": {
            "size": {"food": 1.0, "water": 1.0}
        },
        "vehicles": {
            "capacity": 1000
        },
        "aps": {
            "min_per_commodity": 1
        }
    }
    sim_params = params["simulation"]
    commodity_size = params["commodities"]["size"]
    vehicle_capacity = params["vehicles"]["capacity"]
    min_APS_per_commodity = params["aps"]["min_per_commodity"]


    # Define profiles for different scenario types with their characteristic distributions and tags
    scenario_profiles = {
        0: {
            "impact_radius_km": lambda: random.gauss(
                sim_params["scenario_profiles"]["coastal"]["impact_radius_mean_km"],
                sim_params["scenario_profiles"]["coastal"]["impact_radius_std_km"]
            ),
            "infrastructure_loss_rate": lambda: random.uniform(
                sim_params["scenario_profiles"]["coastal"]["infrastructure_loss_min"],
                sim_params["scenario_profiles"]["coastal"]["infrastructure_loss_max"]
            ),
            "geographic_tag": lambda: "coastal"
        },
        1: {
            "impact_radius_km": lambda: random.gauss(
                sim_params["scenario_profiles"]["island_chain"]["impact_radius_mean_km"],
                sim_params["scenario_profiles"]["island_chain"]["impact_radius_std_km"]
            ),
            "infrastructure_loss_rate": lambda: random.uniform(
                sim_params["scenario_profiles"]["island_chain"]["infrastructure_loss_min"],
                sim_params["scenario_profiles"]["island_chain"]["infrastructure_loss_max"]
            ),
            "geographic_tag": lambda: "island_chain"
        },
        2: {
            "impact_radius_km": lambda: random.gauss(
                sim_params["scenario_profiles"]["urban_cluster"]["impact_radius_mean_km"],
                sim_params["scenario_profiles"]["urban_cluster"]["impact_radius_std_km"]
            ),
            "infrastructure_loss_rate": lambda: random.uniform(
                sim_params["scenario_profiles"]["urban_cluster"]["infrastructure_loss_min"],
                sim_params["scenario_profiles"]["urban_cluster"]["infrastructure_loss_max"]
            ),
            "geographic_tag": lambda: "urban_cluster"
        }
    }

    # Determine the number of scenarios to generate
    if num_scenarios is None:
        num_scenarios = sim_params["num_scenarios"]

    scenarios = []
    node_list = list(G.nodes())
    aps_eligible_nodes = node_list

    for sid in range(num_scenarios):
        # Select an epicenter node and scenario type randomly
        epicenter = random.choice(node_list)
        scenario_type = random.choice(list(scenario_profiles.keys()))
        profile = scenario_profiles[scenario_type]

        # Severity is bounded between 0.1 and 3.0 to prevent unrealistic scaling
        severity = min(
            max(0.1, random.gauss(
                sim_params["severity_distribution"]["mean"],
                sim_params["severity_distribution"]["std_dev"]
            )),
            3.0  # Upper bound for severity
        )

        # Adjust impact radius based on severity: higher severity = larger impact area
        # Formula: adjusted_radius = base_radius * (1 + alpha * (severity - 1))
        base_radius = profile["impact_radius_km"]()
        # Scale impact radius based on severity, with adjustable alpha
        alpha = 0.3  # Tuning parameter for severity influence
        impact_radius_km = base_radius * (1 + alpha * (severity - 1))

        loss_rate = profile["infrastructure_loss_rate"]()
        geo_tag = profile["geographic_tag"]()

        # Calculate number of neighbors within impact radius
        num_neighbors = sum(
            1 for neighbor in G.neighbors(epicenter)
            if haversine(locations[epicenter]["coords"], locations[neighbor]["coords"]) <= impact_radius_km
        )

        # Identify affected nodes within the impact radius
        affected_nodes = [
            node for node in node_list
            if haversine(locations[epicenter]["coords"], locations[node]["coords"]) <= impact_radius_km
        ]

        # Calculate demand and supply for affected nodes
        demand = {}
        supply = {}

        supply_fraction = sim_params["supply_fraction"]["value"]
        surplus_factor = sim_params.get("initial_supply", {}).get("surplus_factor", 1.0)

        for node in G.nodes():
            pop = G.nodes[node]["population"]

            # Only generate demand for affected nodes
            if node in affected_nodes:
                food_demand = pop * sim_params["base_demand_food_per_capita"] * severity
                water_demand = pop * sim_params["base_demand_water_per_capita"] * severity
                demand[(node, "food")] = food_demand
                demand[(node, "water")] = water_demand

            # Always generate supply
            food_supply = pop * sim_params["base_demand_food_per_capita"] * supply_fraction
            water_supply = pop * sim_params["base_demand_water_per_capita"] * supply_fraction
            food_supply *= surplus_factor
            water_supply *= surplus_factor
            supply[(node, "food")] = food_supply
            supply[(node, "water")] = water_supply

        # Set capacities for edges in the graph
        capacity = {}
        for (i, j) in G.edges():
            capacity[(i, j, "food")] = 1e6
            capacity[(i, j, "water")] = 1e6

        # Define safety stock and node capacity for affected nodes
        safety_days = sim_params["safety_days"]
        capacity_per_capita = sim_params["capacity_per_capita"]
        base_demand_food = sim_params["base_demand_food_per_capita"]
        base_demand_water = sim_params["base_demand_water_per_capita"]

        safety_stock = {}
        node_capacity = {}

        for node in G.nodes():
            pop = G.nodes[node]["population"]
            capacity_base = sim_params.get("node_capacity", {}).get("base", 1.0)
            capacity_surplus = sim_params.get("node_capacity", {}).get("surplus_factor", 0.0)
            scaled_capacity = capacity_base * (1 + capacity_surplus * pop)

            for c in ["food", "water"]:
                if node not in affected_nodes:
                    base_demand = base_demand_food if c == "food" else base_demand_water
                    safety_stock[(node, c)] = safety_days * base_demand * pop
                else:
                    safety_stock[(node, c)] = 0.0

            for t in range(1, 6):  # Time periods 1 to 5
                node_capacity[(node, t)] = scaled_capacity

        # Assemble the scenario dictionary
        scenario = {
            "scenario_id": sid,
            "epicenter": epicenter,
            "epicenter_neighbors": num_neighbors,
            "region_mapping": {node: locations[node].get("region", None) for node in node_list},
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
                "aps_eligible_nodes": aps_eligible_nodes,
                "node_capacity": node_capacity
            }
        }
        scenarios.append(scenario)

    def convert_to_nested_list_format(d):
        return [[list(k), v] for k, v in d.items()]

    export_scenarios = []
    for s in scenarios:
        export_scenario = s.copy()
        export_scenario["demand"] = convert_to_nested_list_format(s["demand"])
        export_scenario["supply"] = convert_to_nested_list_format(s["supply"])
        export_scenario["capacity"] = convert_to_nested_list_format(s["capacity"])
        export_scenario["params"]["safety_stock"] = convert_to_nested_list_format(s["params"]["safety_stock"])
        export_scenario["params"]["node_capacity"] = convert_to_nested_list_format(s["params"]["node_capacity"])
        export_scenarios.append(export_scenario)

    output_path = os.path.join(os.path.dirname(__file__), "simulation_output.json")
    with open(output_path, "w") as f:
        json.dump(export_scenarios, f, indent=2)
    print(f"Simulation results saved to {output_path}")
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


if __name__ == "__main__":
    from disaster_logistics_model.network.network_builder import build_geospatial_network

    csv_path = "disaster_logistics_model/data/pacific_cities.csv"
    G, locations = build_geospatial_network(csv_path)
    generate_scenarios(G, locations)