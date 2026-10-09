"""
input_builder.py

Second stage of the pipeline (scenarios -> stochastic instance -> gurobi
model). Takes the raw scenario draws from scenarios/scenario_generator.py plus
the node table and yaml parameters, and turns them into every set/parameter the
gurobi model in model/model.py needs (arcs, demand, inventory,
costs, capacities, probabilities).

This file assumes:
- scenarios contain only exogenous disaster realization data
- demand is fixed across scenarios
- inventory is fixed across scenarios
- nominal arc capacity is fixed across scenarios
- residual arc capacity varies by scenario through the degradation matrix
  scaled by alpha (uniform) or mode_alphas (per-mode override)

Unit convention:
- 1 unit of any commodity = 1 person-day of support
  (e.g. demand of 100 units = enough food/water for 100 people for 1 day)

Step 4 changes:
- Removed: build_directed_arcs, build_arc_cost, build_nominal_arc_capacity,
  build_residual_arc_capacity (all deprecated since Step 3)
- Removed: deprecated backward-compat instance keys (arcs, arc_cost,
  nominal_arc_capacity, residual_arc_capacity, gamma scalar)
- Added: alpha and mode_alphas parameters to build_modal_residual_capacity
- Added: alpha, mode_alphas, forced_type parameters to build_stochastic_instance
"""

import warnings
from typing import Dict, List, Optional, Tuple

from network.network_builder import load_sea_arcs, load_air_arcs, load_land_arcs, load_transfer_capacities


def build_modal_arcs() -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load the three modal arc layers (sea, air, land) from the network CSVs
    produced by network/build_modal_arcs.py.

    Returns a 4-tuple:
        modal_arcs: dict with keys "sea", "air", "land" (each a list of
            directed (i,j) pairs) plus "all" (sorted union across all modes)
        modal_capacity: nested dict modal_capacity[mode][(i,j)][commodity]
            giving the undisrupted throughput in person-days from U_food /
            U_water columns of the arc CSVs
        modal_arc_cost: nested dict modal_arc_cost[mode][(i,j)] = distance_km
            / 1000 (SME placeholder for per-mode transport cost coefficient)
        modal_arc_distance: nested dict modal_arc_distance[mode][(i,j)] =
            distance_km (raw great-circle km, used by the vehicle distance
            budget constraint)
    """
    loaders = {"sea": load_sea_arcs, "air": load_air_arcs, "land": load_land_arcs}
    modal_arcs_raw: Dict[str, List] = {"sea": [], "air": [], "land": []}
    modal_capacity: Dict[str, Dict] = {"sea": {}, "air": {}, "land": {}}
    modal_arc_cost: Dict[str, Dict] = {"sea": {}, "air": {}, "land": {}}
    modal_arc_distance: Dict[str, Dict] = {"sea": {}, "air": {}, "land": {}}

    for mode, loader in loaders.items():
        for row in loader():
            i = int(row["from_node"])
            j = int(row["to_node"])
            dist_km = float(row["distance_km"])
            modal_arcs_raw[mode].append((i, j))
            modal_capacity[mode][(i, j)] = {
                "food": float(row["U_food"]),
                "water": float(row["U_water"]),
            }
            modal_arc_cost[mode][(i, j)] = dist_km * 0.0001
            modal_arc_distance[mode][(i, j)] = dist_km

    all_arcs = sorted(
        set(modal_arcs_raw["sea"])
        | set(modal_arcs_raw["air"])
        | set(modal_arcs_raw["land"])
    )
    modal_arcs_raw["all"] = all_arcs

    return modal_arcs_raw, modal_capacity, modal_arc_cost, modal_arc_distance


def build_modal_residual_capacity(
    modal_arcs: Dict,
    modal_capacity: Dict,
    scenario: Dict,
    params: Dict,
    alpha: float = 1.0,
    mode_alphas: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Build residual arc capacity u[mode][(i,j)][commodity] for a single scenario.

    Uses the scenario's disaster_type field and the degradation_matrix from
    params to compute a per-mode, per-commodity degradation factor:

        alpha_m     = mode_alphas[m] if mode_alphas else alpha
        gamma_m     = degradation_matrix[mode][disaster_type] * alpha_m
        s_endpoint  = max(node_severity[i], node_severity[j])
        factor      = max(0, 1 - gamma_m * s_endpoint / 5)
        u[m][(i,j)][commodity] = modal_capacity[m][(i,j)][commodity] * factor

    alpha: uniform scaling factor applied to all gamma[mode][type] values.
        alpha=0.0 means arcs are never degraded; alpha=1.0 means full matrix
        values apply. Used for the Primary (Option A) sweep.

    mode_alphas: per-mode scaling overrides. If provided, overrides alpha for
        specific modes. e.g. {"sea": 0.5, "air": 1.0, "land": 1.0} scales
        sea degradation by 0.5 while air/land use their full degradation_matrix
        values. mode_alphas[m] takes precedence over alpha for mode m.
        Used for the Secondary B (mode-isolation) sweep.

    Missing node_severity entries default to 0 (node unaffected).
    Missing disaster_type or unknown type defaults to no degradation (factor=1).
    """
    degradation_matrix = params.get("degradation_matrix", {})
    node_severity = scenario.get("node_severity", {})
    disaster_type = scenario.get("disaster_type")

    u_residual: Dict[str, Dict] = {}
    for mode in ("sea", "air", "land"):
        u_residual[mode] = {}

        if mode_alphas is not None:
            alpha_m = float(mode_alphas.get(mode, alpha))
        else:
            alpha_m = float(alpha)

        if disaster_type:
            gamma_m = float(degradation_matrix.get(mode, {}).get(disaster_type, 0.0)) * alpha_m
        else:
            gamma_m = 0.0

        for (i, j), cap_by_commodity in modal_capacity[mode].items():
            s_i = float(node_severity.get(i, 0.0))
            s_j = float(node_severity.get(j, 0.0))
            severity_endpoint = max(s_i, s_j)
            degradation_factor = max(0.0, 1.0 - gamma_m * severity_endpoint / 5.0)
            u_residual[mode][(i, j)] = {
                commodity: cap * degradation_factor
                for commodity, cap in cap_by_commodity.items()
            }

    return u_residual


def build_transfer_capacity(params: Dict) -> Dict[int, Dict]:
    """
    Load intermodal transfer capacity from network/transfer_capacities.csv.

    transfer_cap[node_id][(m1, m2)] = capacity in person-days per window.
    Only mode pairs with capacity > 0 are included; inactive pairs are omitted.

    params: loaded yaml parameters (unused; accepted for signature consistency)

    returns transfer_cap dict keyed by node_id then (m1, m2) mode-pair tuple
    """
    raw = load_transfer_capacities()

    mode_pair_cols = {
        ("sea", "air"):  "T_sea_to_air",
        ("sea", "land"): "T_sea_to_land",
        ("air", "land"): "T_air_to_land",
        ("land", "sea"): "T_land_to_sea",
        ("land", "air"): "T_land_to_air",
        ("air", "sea"):  "T_air_to_sea",
    }

    transfer_cap: Dict[int, Dict] = {}
    for node_id, row in raw.items():
        node_cap = {}
        for (m1, m2), col in mode_pair_cols.items():
            val = row.get(col, 0)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                fval = 0.0
            if fval > 0.0:
                node_cap[(m1, m2)] = fval
        if node_cap:
            transfer_cap[node_id] = node_cap

    return transfer_cap


def build_transfer_cost(params: Dict) -> Dict[Tuple[str, str], float]:
    """
    Build intermodal transfer cost multipliers.

    Values are dimensionless multipliers applied as:
        cost_contribution = transfer_cost[(m1, m2)] * tau[w, i, m1, m2, r]
    in the loss function (consistent with modal_arc_cost = distance_km/1000).

    Read from the transfer_costs block of config/model_parameters.yaml
    (keys like sea_air, land_sea); the previous hardcoded values are kept
    as per-pair fallback defaults for configs without the block.

    params: loaded yaml parameters (transfer_costs)

    returns dict mapping (m1, m2) mode-pair tuples to cost multipliers
    """
    defaults = {
        ("sea",  "air"):  15.0,
        ("sea",  "land"): 2.5,
        ("air",  "land"): 4.0,
        ("land", "sea"):  3.5,
        ("land", "air"):  6.0,
        ("air",  "sea"):  10.0,
    }
    configured = params.get("transfer_costs", {}) or {}
    return {
        (m1, m2): float(configured.get(f"{m1}_{m2}", default))
        for (m1, m2), default in defaults.items()
    }


def build_fixed_demand(
    locations: Dict[int, Dict],
    commodities: List[str],
    scenarios: List[Dict],
    params: Dict,
) -> Dict[Tuple, float]:
    """
    Build scenario-dependent demand d[w,i,r] = alpha_r * severity_w_i * pop_i.

    Each scenario's realized node severity drives its own demand vector, so
    the flow balance constraints see different RHS values per scenario.
    One unit = one person-day of support.

    locations: node attribute dict, used for population
    commodities: list of commodity names (food, water)
    scenarios: scenario list from generate_scenarios
    params: loaded yaml parameters (demand.alpha)

    returns dict keyed by (scenario_id, node_id, commodity) -> demand in person-days
    """
    active_bin = params.get("active_bin", "bin_1")
    bin_params = params.get("time_bins", {}).get(active_bin, {})
    alpha = bin_params.get("demand_alpha", params.get("demand", {}).get("alpha", {}))

    from network.network_builder import load_sea_arcs, load_air_arcs, load_land_arcs
    reachable_nodes = set()
    for arc in load_sea_arcs():
        reachable_nodes.add(int(arc['to_node']))
    for arc in load_air_arcs():
        reachable_nodes.add(int(arc['to_node']))
    for arc in load_land_arcs():
        reachable_nodes.add(int(arc['to_node']))

    demand = {}
    for scenario in scenarios:
        w = int(scenario["scenario_id"])
        node_severity = scenario["node_severity"]
        for i, data in locations.items():
            pop_i = float(data.get("pop", 0.0))
            s_i = float(node_severity.get(i, 0.0))
            for r in commodities:
                alpha_r = float(alpha.get(r, 0.15))
                if i in reachable_nodes:
                    demand[(w, i, r)] = alpha_r * s_i * pop_i
                else:
                    demand[(w, i, r)] = 0.0
    return demand


def build_inventory(
    locations: Dict[int, Dict],
    commodities: List[str],
    params: Dict,
) -> Dict[Tuple[int, str], float]:
    """
    Build fixed inventory q_bar[i,r] based on PPL tier.

    Inventory is determined by the site's tier classification, not by
    local population. Non-PPL nodes carry zero inventory regardless of
    whether they are selected, since only PPL-eligible nodes can be
    activated as prepositioning sites.
    One unit = one person-day of support.

    locations: node attribute dict, used for tier classification
    commodities: list of commodity names (food, water)
    params: loaded yaml parameters (inventory.tier_capacity)

    returns dict keyed by (node_id, commodity) -> inventory capacity in
    person-days, same value for both commodities at a given node
    """
    tier_capacity = params.get("inventory", {}).get("tier_capacity", {})

    inventory = {}
    for i, data in locations.items():
        tier = str(data.get("tier", "None")).strip()
        capacity = float(tier_capacity.get(tier, 0.0))
        for r in commodities:
            inventory[(i, r)] = capacity

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

    locations: node attribute dict
    commodities: list of commodity names (food, water)
    scenarios: scenario list from generate_scenarios, used for node_severity
    cutoff_severity: severity threshold at/above which a site's stock is
    considered unusable for that scenario

    returns dict keyed by (scenario_id, node_id, commodity) -> 0.0 or 1.0
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


def build_site_cost(
    locations: Dict[int, Dict],
    params: Dict,
) -> Dict[int, float]:
    """
    Build node-level fixed site selection cost f[i].

    Supported parameter structures:
    1. prepositioning.site_cost keyed by hub type, e.g.
       major/regional/local
    2. prepositioning.site_cost_by_node keyed by node id
    3. prepositioning.default_site_cost keyed by hub type with optional
       prepositioning.site_cost_overrides keyed by node id

    Node-specific overrides take precedence over hub-type defaults.

    locations: node attribute dict, used for hub_type
    params: loaded yaml parameters (prepositioning.*)

    returns dict keyed by node_id -> selection cost (budget units)
    """
    prepositioning = params.get("prepositioning", {})

    site_cost_by_type = prepositioning.get("site_cost", {})
    site_cost_by_node = prepositioning.get("site_cost_by_node", {})
    default_site_cost = prepositioning.get("default_site_cost", {})
    site_cost_overrides = prepositioning.get("site_cost_overrides", {})

    node_cost_lookup = {
        int(node_id): float(cost)
        for node_id, cost in site_cost_by_node.items()
    }
    override_lookup = {
        int(node_id): float(cost)
        for node_id, cost in site_cost_overrides.items()
    }

    type_cost_lookup = site_cost_by_type if site_cost_by_type else default_site_cost
    type_cost_lookup = {
        str(hub_type).strip().lower(): float(cost)
        for hub_type, cost in type_cost_lookup.items()
    }

    site_cost = {}
    for i, data in locations.items():
        if i in override_lookup:
            site_cost[i] = override_lookup[i]
            continue

        if i in node_cost_lookup:
            site_cost[i] = node_cost_lookup[i]
            continue

        hub_type = str(data.get("hub_type", "local")).strip().lower()
        if hub_type not in type_cost_lookup:
            raise KeyError(
                f"Missing site selection cost for hub type '{hub_type}' at node {i}."
            )

        site_cost[i] = type_cost_lookup[hub_type]

    return site_cost


def build_selection_budget(params: Dict) -> float:
    """
    Build scalar preposition site selection budget B.

    params: loaded yaml parameters (prepositioning.selection_budget)

    returns the total selection budget as a float, used to cap the
    cost-weighted sum of selected sites
    """
    prepositioning = params.get("prepositioning", {})
    if "selection_budget" not in prepositioning:
        raise KeyError(
            "Missing required parameter 'prepositioning.selection_budget'."
        )
    return float(prepositioning["selection_budget"])


def build_penalty(
    locations: Dict[int, Dict],
    commodities: List[str],
    base_penalty: float = 1.0,
) -> Dict[Tuple[int, str], float]:
    """
    Build unmet-demand penalty delta[i,r].

    locations: node attribute dict (iterated to build one entry per node)
    commodities: list of commodity names (food, water)
    base_penalty: penalty cost charged per unit of unmet demand

    returns dict keyed by (node_id, commodity) -> penalty per unmet unit
    """
    penalty = {}
    for i in locations:
        for r in commodities:
            penalty[(i, r)] = float(base_penalty)
    return penalty


def build_probability(scenarios: List[Dict]) -> Dict[int, float]:
    """
    Build scenario probability dictionary.

    scenarios: scenario list from generate_scenarios

    returns dict keyed by scenario_id -> probability, used as prob[w] in
    the cvar objective
    """
    prob = {}
    for scenario in scenarios:
        w = int(scenario["scenario_id"])
        prob[w] = float(scenario.get("probability", 0.0))
    return prob


def build_vehicle_params(
    locations: Dict[int, Dict],
    params: Dict,
) -> Dict:
    """
    Build vehicle type parameters for the vehicle-heterogeneity extension.

    Computes basing eligibility J_k, initial basing b_{k,j} (tier-weighted
    3:2:1 with floor-divide and remainder to highest tier), distance budget
    D_k = 3*v_k, and turnaround penalty pi_k = (turnaround_hours/24)*v_k.

    Returns a dict with keys "vehicle_types" and "K_m", or empty dict if
    no vehicles are configured.
    """
    vehicle_config = params.get("vehicles", {})
    if not vehicle_config:
        return {}

    tier_weights = {"PPL-1": 3, "PPL-2": 2, "PPL-3": 1}

    ppl_nodes = [
        i for i in locations
        if locations[i].get("ppl_eligible", "No") in ("Y", "Y(C)")
    ]

    vehicle_types: Dict[str, Dict] = {}
    K_m: Dict[str, List[str]] = {"sea": [], "air": [], "land": []}

    for vtype_name, vconfig in vehicle_config.items():
        mode = str(vconfig["mode"])
        fleet_size = int(vconfig["fleet_size"])
        rating_field = str(vconfig["min_rating_field"])
        rating_value = int(vconfig["min_rating_value"])
        cruise_speed = float(vconfig["cruise_speed_km_day"])
        turnaround_hours = float(vconfig["turnaround_hours"])
        capacity = {r: float(v) for r, v in vconfig["capacity"].items()}

        J_k = [
            j for j in ppl_nodes
            if int(locations[j].get(rating_field, 0)) >= rating_value
        ]

        basing_tiers = vconfig.get("basing_tiers")
        if basing_tiers is not None:
            allowed = set(basing_tiers)
            J_k = [j for j in J_k if str(locations[j].get("tier", "None")).strip() in allowed]

        b_kj: Dict[int, int] = {}
        total_weight = sum(
            tier_weights.get(str(locations[j].get("tier", "None")).strip(), 0)
            for j in J_k
        )
        if total_weight > 0 and fleet_size > 0:
            for j in J_k:
                w = tier_weights.get(str(locations[j].get("tier", "None")).strip(), 0)
                b_kj[j] = int(fleet_size * w // total_weight)
            remainder = fleet_size - sum(b_kj.values())
            if remainder > 0:
                best = max(J_k, key=lambda j: (
                    tier_weights.get(str(locations[j].get("tier", "None")).strip(), 0),
                    -j,
                ))
                b_kj[best] += remainder

        vehicle_types[vtype_name] = {
            "mode": mode,
            "fleet_size": fleet_size,
            "capacity": capacity,
            "J_k": J_k,
            "b_kj": b_kj,
            "D_k": 3.0 * cruise_speed,
            "pi_k": (turnaround_hours / 24.0) * cruise_speed,
            "cruise_speed_km_day": cruise_speed,
        }
        K_m[mode].append(vtype_name)

    return {
        "vehicle_types": vehicle_types,
        "K_m": K_m,
    }


def build_stochastic_instance(
    locations: Dict[int, Dict],
    scenarios: List[Dict],
    params: Dict,
    alpha: float = 1.0,
    mode_alphas: Optional[Dict[str, float]] = None,
    forced_type: Optional[str] = None,
    **_kwargs,
) -> Dict:
    """
    Build the full stochastic model instance dictionary.

    This function is the single assembly point — it calls every build_*
    helper above and packages the results into one dict that
    solve_stochastic_cvar consumes directly.

    locations: node attribute dict from nodes.csv
    scenarios: scenario list from generate_scenarios
    params: loaded yaml parameters
    alpha: uniform scaling factor applied to all gamma[mode][type] entries in
        the degradation_matrix. alpha=0.0 disables all arc degradation;
        alpha=1.0 applies the full matrix (default). Used for the Primary
        (Option A) sensitivity sweep.
    mode_alphas: per-mode scaling overrides for the Secondary B (mode-
        isolation) sweep. If provided, overrides alpha for the specified modes.
        e.g. {"sea": 0.5, "air": 1.0, "land": 1.0} degrades sea arcs at half
        strength while air and land use their full degradation_matrix values.
    forced_type: the disaster type used to generate scenarios (or None for
        mixed-type draws). Stored in the instance dict for traceability; the
        actual forcing is applied at scenario generation time in
        scenarios/scenario_generator.py.
    **_kwargs: absorbs legacy positional arguments (e.g. undirected_edges,
        gamma) so callers using old signatures do not raise TypeError.

    returns the instance dict (keys documented in-line below)
    """
    # D2 audit fix: legacy arguments are absorbed for signature compatibility,
    # but silently ignoring a non-empty one (e.g. gamma=0.5) lets a caller
    # believe it parameterized the instance when it did nothing. Warn loudly.
    _swallowed = sorted(k for k, v in _kwargs.items() if v is not None)
    if _swallowed:
        warnings.warn(
            f"build_stochastic_instance ignoring legacy argument(s) {_swallowed}: "
            "these have no effect since the Step 4 signature change "
            "(gamma was replaced by alpha, which uniformly scales the "
            "degradation_matrix; undirected_edges is unused).",
            stacklevel=2,
        )

    nodes = list(locations.keys())
    ppl_nodes = [
        i for i in nodes
        if locations[i].get("ppl_eligible", "No") in ("Y", "Y(C)")
    ]
    commodities = list(params["commodities"])
    omega = [int(scenario["scenario_id"]) for scenario in scenarios]

    # --- Modal arc layers ---
    modal_arcs, modal_capacity, modal_arc_cost, modal_arc_distance = build_modal_arcs()

    # Per-scenario residual capacity for each modal arc and commodity,
    # degraded by disaster_type-specific gamma_m from the degradation_matrix
    # scaled by alpha (or per-mode by mode_alphas for the Secondary B sweep).
    modal_residual = [
        build_modal_residual_capacity(
            modal_arcs, modal_capacity, scenario, params,
            alpha=alpha, mode_alphas=mode_alphas,
        )
        for scenario in scenarios
    ]

    transfer_cap = build_transfer_capacity(params)
    transfer_cost = build_transfer_cost(params)
    modes = ["sea", "air", "land"]

    # --- Vehicle heterogeneity ---
    vehicle_data = build_vehicle_params(locations, params)

    # --- Demand / inventory / cost parameters ---
    demand = build_fixed_demand(locations, commodities, scenarios, params)
    inventory_if_open = build_inventory(locations, commodities, params)
    inventory_cutoff_severity = float(
        params.get("inventory_disruption", {}).get("cutoff_severity", 4.0)
    )
    safety_stock_fraction = float(
        params.get("inventory_disruption", {}).get("safety_stock", {}).get("fraction", 0.25)
    )
    inventory_availability = build_inventory_availability(
        locations=locations,
        commodities=commodities,
        scenarios=scenarios,
        cutoff_severity=inventory_cutoff_severity,
    )
    site_cost = build_site_cost(locations, params)
    selection_budget = build_selection_budget(params)
    active_bin = params.get("active_bin", "bin_1")
    bin_params = params.get("time_bins", {}).get(active_bin, {})
    bin_penalty = float(bin_params.get("unmet_penalty", 1.0))
    penalty = build_penalty(locations, commodities, base_penalty=bin_penalty)
    probability = build_probability(scenarios)

    return {
        # --- Sets ---
        "nodes": nodes,
        "ppl_nodes": ppl_nodes,
        "commodities": commodities,
        "scenarios": omega,
        "modes": modes,
        # --- Modal network ---
        "modal_arcs": modal_arcs,
        "modal_capacity": modal_capacity,
        "modal_arc_cost": modal_arc_cost,
        "modal_arc_distance": modal_arc_distance,
        "modal_residual": modal_residual,
        "transfer_cap": transfer_cap,
        "transfer_cost": transfer_cost,
        "vehicle_types": vehicle_data.get("vehicle_types", {}),
        "K_m": vehicle_data.get("K_m", {"sea": [], "air": [], "land": []}),
        # --- Stochastic parameters ---
        "probability": probability,
        "demand": demand,
        "inventory_if_open": inventory_if_open,
        "inventory_availability": inventory_availability,
        "penalty": penalty,
        "site_cost": site_cost,
        "selection_budget": selection_budget,
        "B": selection_budget,
        "safety_stock_fraction": safety_stock_fraction,
        "rho": safety_stock_fraction,
        "inventory_cutoff_severity": inventory_cutoff_severity,
        "tau_cutoff": inventory_cutoff_severity,
        "P_max": int(params.get("P_max", 5)),
        "beta": float(params.get("beta", 0.9)),
        # --- Experiment metadata ---
        "alpha": float(alpha),
        "mode_alphas": mode_alphas,
        "forced_type": forced_type,
    }
