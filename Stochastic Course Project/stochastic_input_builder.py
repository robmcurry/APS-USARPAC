"""
stochastic_input_builder.py

Builds a stochastic optimization instance for the course project.

This file assumes:
- scenarios contain only exogenous disaster realization data
- demand is fixed across scenarios
- inventory is fixed across scenarios
- nominal arc capacity is fixed across scenarios
- residual arc capacity varies by scenario through gamma and node severity

Unit convention:
- 1 unit of any commodity = 1 person-day of support
"""

from typing import Dict, List, Tuple

from geopy.distance import geodesic



def build_directed_arcs(
    locations: Dict[int, Dict],
    undirected_edges: List[Tuple[int, int]],
    params: Dict,
) -> List[Tuple[int, int]]:
    """
    Convert undirected edges into directed arcs, pruning any arc whose
    great-circle distance exceeds the maximum feasible transport range.

    Maximum feasible range is computed as:
        ship_km_per_day * max_transport_days

    where max_transport_days is taken as the minimum value across the
    commodity-specific max_voyage_days settings so that the retained arc set
    is feasible for all commodities.
    """
    ship_km_per_day = float(params.get("ship_km_per_day", 1500))
    max_voyage_days = params.get("max_voyage_days", {})

    if max_voyage_days:
        max_transport_days = float(min(max_voyage_days.values()))
    else:
        max_transport_days = 3.0

    max_distance_km = ship_km_per_day * max_transport_days

    arcs = []
    for i, j in undirected_edges:
        coords_i = (float(locations[i]["lat"]), float(locations[i]["lon"]))
        coords_j = (float(locations[j]["lat"]), float(locations[j]["lon"]))
        distance_km = geodesic(coords_i, coords_j).kilometers

        if distance_km <= max_distance_km:
            arcs.append((i, j))
            arcs.append((j, i))

    return arcs



def build_fixed_demand(
    locations: Dict[int, Dict],
    commodities: List[str],
) -> Dict[Tuple[int, str], float]:
    """
    Build fixed demand in person-day units.

    Demand is held constant across scenarios to isolate arc-capacity uncertainty.
    One unit = one person-day of support.
    """
    demand = {}
    for i, data in locations.items():
        pop_i = float(data.get("pop", 0.0))
        for r in commodities:
            demand[(i, r)] = pop_i
    return demand



def build_inventory(
    locations: Dict[int, Dict],
    commodities: List[str],
    params: Dict,
) -> Dict[Tuple[int, str], float]:
    """
    Build fixed exogenous inventory q_bar[i,r] in person-day units.

    q_bar[i,r] = population_i * coverage_days_r
    """
    inventory = {}
    coverage_days = params.get("inventory_coverage_days", {})

    for i, data in locations.items():
        pop_i = float(data.get("pop", 0.0))
        for r in commodities:
            days = float(coverage_days.get(r, 3))
            inventory[(i, r)] = pop_i * days

    return inventory


def build_inventory_availability(
    locations: Dict[int, Dict],
    commodities: List[str],
    scenarios: List[Dict],
    cutoff_severity: float = 4.0,
) -> Dict[Tuple[int, int, str], float]:
    """
    Build scenario-dependent inventory availability a[w,i,r].

    Rule:
        a[w,i,r] = 0 if node severity at i in scenario w is >= cutoff_severity
        a[w,i,r] = 1 otherwise

    This prevents severely affected nodes from acting as root supply origins,
    while still allowing them to serve as transshipment nodes in the model.
    """
    availability = {}

    for scenario in scenarios:
        w = int(scenario["scenario_id"])
        node_severity = scenario["node_severity"]

        for i in locations:
            s_i = float(node_severity.get(i, 0.0))
            factor = 0.0 if s_i >= cutoff_severity else 1.0

            for r in commodities:
                availability[(w, i, r)] = factor

    return availability



def build_nominal_arc_capacity(
    locations: Dict[int, Dict],
    arcs: List[Tuple[int, int]],
    params: Dict,
) -> Dict[Tuple[int, int], float]:
    """
    Build nominal arc capacity U[i,j] in person-day units.

    U[i,j] = throughput_days * max(pop_i, pop_j)
    """
    nominal_capacity = {}
    throughput_days = float(
        params.get("nominal_arc_capacity", {}).get("throughput_days", 3)
    )

    for i, j in arcs:
        pop_i = float(locations[i].get("pop", 0.0))
        pop_j = float(locations[j].get("pop", 0.0))
        pop_max = max(pop_i, pop_j)
        nominal_capacity[(i, j)] = throughput_days * pop_max

    return nominal_capacity


def build_residual_arc_capacity(
    arcs: List[Tuple[int, int]],
    scenarios: List[Dict],
    nominal_capacity: Dict[Tuple[int, int], float],
    gamma: float,
) -> Dict[Tuple[int, int, int], float]:
    """
    Build residual arc capacity u[w,i,j] from scenario node severity and gamma.

    u[w,i,j] = U[i,j] * max(0, 1 - gamma * max(s_i, s_j) / 5)

    Severity is assumed to be on a 0 to 5 scale.
    """
    residual_capacity = {}

    for scenario in scenarios:
        w = int(scenario["scenario_id"])
        node_severity = scenario["node_severity"]

        for i, j in arcs:
            s_i = float(node_severity.get(i, 0.0))
            s_j = float(node_severity.get(j, 0.0))
            s_bar = max(s_i, s_j)
            residual_factor = max(0.0, 1.0 - gamma * (s_bar / 5.0))
            U_ij = nominal_capacity[(i, j)]
            residual_capacity[(w, i, j)] = U_ij * residual_factor

    return residual_capacity



def build_penalty(
    locations: Dict[int, Dict],
    commodities: List[str],
    base_penalty: float = 1.0,
) -> Dict[Tuple[int, str], float]:
    """
    Build unmet-demand penalty delta[i,r].
    """
    penalty = {}
    for i in locations:
        for r in commodities:
            penalty[(i, r)] = float(base_penalty)
    return penalty



def build_probability(scenarios: List[Dict]) -> Dict[int, float]:
    """
    Build scenario probability dictionary.
    """
    prob = {}
    for scenario in scenarios:
        w = int(scenario["scenario_id"])
        prob[w] = float(scenario.get("probability", 0.0))
    return prob



def build_stochastic_instance(
    locations: Dict[int, Dict],
    undirected_edges: List[Tuple[int, int]],
    scenarios: List[Dict],
    params: Dict,
    gamma: float,
) -> Dict:
    """
    Build the full stochastic model instance dictionary.

    The directed arc set is pruned to retain only arcs that are feasible within
    the allowable transport-time limit implied by the parameter settings.
    """
    nodes = list(locations.keys())
    arcs = build_directed_arcs(locations, undirected_edges, params)
    commodities = list(params["commodities"])
    omega = [int(scenario["scenario_id"]) for scenario in scenarios]

    demand = build_fixed_demand(locations, commodities)
    inventory_if_open = build_inventory(locations, commodities, params)
    inventory_cutoff_severity = float(
        params.get("inventory_disruption", {}).get("cutoff_severity", 4.0)
    )
    safety_stock_fraction = float(
        params.get("inventory_disruption", {}).get("safety_stock", {}).get("fraction", 0.0)
    )
    inventory_availability = build_inventory_availability(
        locations=locations,
        commodities=commodities,
        scenarios=scenarios,
        cutoff_severity=inventory_cutoff_severity,
    )
    nominal_arc_capacity = build_nominal_arc_capacity(locations, arcs, params)
    residual_arc_capacity = build_residual_arc_capacity(
        arcs=arcs,
        scenarios=scenarios,
        nominal_capacity=nominal_arc_capacity,
        gamma=gamma,
    )
    penalty = build_penalty(locations, commodities)
    probability = build_probability(scenarios)

    instance = {
        "nodes": nodes,
        "arcs": arcs,
        "commodities": commodities,
        "scenarios": omega,
        "probability": probability,
        "demand": demand,
        "inventory_if_open": inventory_if_open,
        "inventory_availability": inventory_availability,
        "nominal_arc_capacity": nominal_arc_capacity,
        "residual_arc_capacity": residual_arc_capacity,
        "penalty": penalty,
        "gamma": float(gamma),
        "safety_stock_fraction": safety_stock_fraction,
        "inventory_cutoff_severity": inventory_cutoff_severity,
        "P_max": int(params.get("P_max", 5)),
        "beta": float(params.get("beta", 0.9)),
    }

    return instance