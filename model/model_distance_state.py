"""Type-indexed vehicle flow on a cumulative-distance-expanded network.

This formulation replaces interchangeable individual vehicle labels with
integer vehicle counts by type.  A movement variable carries a cumulative
distance state, so every unit of flow can be decomposed into a base-rooted
vehicle path whose total distance is within the vehicle type's budget.

The input data contains one planning horizon rather than explicit periods.
Consequently the state expansion is (location, cumulative distance), not
(location, time, cumulative distance).  The same construction can accept a
time dimension later when travel-time and planning-period data are available.
"""

from collections import defaultdict, deque
import math
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


def _status_to_string(status_code: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status_code, f"STATUS_{status_code}")


def _degradation_factor(gamma: float, severity_term: float) -> float:
    return max(0.0, 1.0 - gamma * severity_term / 5.0)


def _distance_network(
    vehicle_types: Dict[str, Dict],
    modal_arcs: Dict[str, List[Tuple[int, int]]],
    modal_arc_distance: Dict[str, Dict[Tuple[int, int], float]],
    distance_step_km,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build sparse reachable distance states and transitions by type.

    Arc consumption is rounded upward and the budget downward.  The resulting
    approximation is conservative: every admitted path satisfies the physical
    distance budget, although a path very close to the limit may be excluded.
    """
    states_by_type: Dict[str, set] = {}
    transitions_by_type: Dict[str, List[Tuple[int, int, int, int]]] = {}
    transitions_from: Dict[Tuple[str, int, int], List] = defaultdict(list)
    transitions_to: Dict[Tuple[str, int, int], List] = defaultdict(list)

    for k, data in vehicle_types.items():
        mode = data["mode"]
        step_km = float(
            distance_step_km[k]
            if isinstance(distance_step_km, dict)
            else distance_step_km
        )
        max_units = int(math.floor(float(data["D_k"]) / step_km + 1e-9))
        if max_units <= 0:
            raise ValueError(
                f"vehicle type {k!r} has distance budget {data['D_k']!r}, "
                f"smaller than distance_step_km={step_km}"
            )

        arc_units = {}
        outgoing = defaultdict(list)
        for i, j in modal_arcs[mode]:
            physical = float(modal_arc_distance.get(mode, {}).get((i, j), 0.0))
            physical += float(data["pi_k"])
            if physical <= 0.0:
                raise ValueError(
                    f"distance-state arcs must consume positive distance; "
                    f"type={k!r}, arc={(i, j)!r}, distance+turnaround={physical}"
                )
            units = max(1, int(math.ceil(physical / step_km - 1e-12)))
            arc_units[i, j] = units
            outgoing[i].append((j, units))

        # All eligible bases are possible sources because basing is optimized.
        reached = {(int(j), 0) for j in data["J_k"]}
        queue = deque(sorted(reached))
        transitions = set()
        while queue:
            i, d = queue.popleft()
            for j, units in outgoing.get(i, ()):
                next_d = d + units
                if next_d > max_units:
                    continue
                transitions.add((i, j, d, next_d))
                state = (j, next_d)
                if state not in reached:
                    reached.add(state)
                    queue.append(state)

        states_by_type[k] = reached
        transitions_by_type[k] = sorted(transitions)
        for i, j, d, next_d in transitions_by_type[k]:
            transitions_from[k, i, d].append((i, j, d, next_d))
            transitions_to[k, j, next_d].append((i, j, d, next_d))

    return states_by_type, transitions_by_type, transitions_from, transitions_to


def solve_distance_state(
    *,
    model: gp.Model,
    instance: Dict[str, Any],
    p: Dict,
    N: List[int],
    PPL: List[int],
    PPL_set: set,
    R: List[str],
    Omega: List[int],
    modes: List[str],
    modal_arcs: Dict[str, List[Tuple[int, int]]],
    modal_arc_cost: Dict[str, Dict[Tuple[int, int], float]],
    vehicle_types: Dict[str, Dict],
    demand: Dict,
    penalty: Dict,
    inventory_if_open: Dict,
    releasable_fraction: float,
    site_cost: Dict,
    selection_budget: float,
    P_max: int,
    beta: float,
    prob: Dict,
    build_only: bool,
    verbose: bool,
    solve_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build and optionally solve the type/distance-state formulation."""
    solve_config = solve_config or {}
    if not vehicle_types:
        raise ValueError("distance_state requires non-empty vehicle_types")
    if "resource_weight" not in instance or "modal_arc_distance" not in instance:
        raise ValueError("distance_state requires resource_weight and modal_arc_distance")
    for required in (
        "node_severity", "disaster_type", "degradation_matrix",
        "nominal_throughput", "node_handling_capacity", "node_handling_bonus",
    ):
        if required not in instance:
            raise ValueError(f"distance_state requires instance[{required!r}]")

    K = sorted(vehicle_types)
    resource_weight = instance["resource_weight"]
    modal_arc_distance = instance["modal_arc_distance"]
    node_severity = instance["node_severity"]
    disaster_type = instance["disaster_type"]
    degradation_matrix = instance["degradation_matrix"]
    nominal_throughput = instance["nominal_throughput"]
    node_handling_capacity = instance["node_handling_capacity"]
    node_handling_bonus = instance["node_handling_bonus"]
    alpha = float(instance.get("alpha", 1.0))

    for r in R:
        if float(resource_weight[r]) <= 0:
            raise ValueError(f"resource_weight[{r!r}] must be positive")
    for k, data in vehicle_types.items():
        for required in ("mode", "fleet_size", "J_k", "cap_tons", "D_k", "pi_k"):
            if required not in data:
                raise ValueError(f"vehicle_types[{k!r}] is missing {required!r}")

    explicit_step = solve_config.get("distance_step_km")
    distance_buckets = int(solve_config.get("distance_buckets", 16))
    if distance_buckets <= 0:
        raise ValueError("distance_buckets must be positive")
    if explicit_step is not None:
        explicit_step = float(explicit_step)
        if explicit_step <= 0:
            raise ValueError("distance_step_km must be positive")
        distance_steps = {k: explicit_step for k in K}
    else:
        distance_steps = {
            k: float(vehicle_types[k]["D_k"]) / distance_buckets for k in K
        }

    a = {
        (w, i): (0.0 if node_severity.get((w, i), 0.0) >= 1.0 else 1.0)
        for w in Omega for i in N
    }
    gamma = {
        (w, m): degradation_matrix.get(m, {}).get(disaster_type[w], 0.0) * alpha
        for w in Omega for m in modes
    }
    states, transitions, transitions_from, transitions_to = _distance_network(
        vehicle_types, modal_arcs, modal_arc_distance, distance_steps
    )

    # First-stage integer vehicle counts based at each eligible location.
    b_keys = [(k, j) for k in K for j in vehicle_types[k]["J_k"]]
    b = model.addVars(b_keys, lb=0, vtype=GRB.INTEGER, name="ds_b")
    for k, j in b_keys:
        b[k, j].UB = int(vehicle_types[k]["fleet_size"])

    # n[w,k,i,j,d] is the number of type-k vehicles traversing (i,j)
    # after having consumed d distance buckets.  The destination state is
    # determined by the transition table.
    n_keys = [
        (w, k, i, j, d)
        for w in Omega for k in K
        for i, j, d, _next_d in transitions[k]
    ]
    n = model.addVars(n_keys, lb=0, vtype=GRB.INTEGER, name="ds_n")
    for w, k, i, j, d in n_keys:
        n[w, k, i, j, d].UB = int(vehicle_types[k]["fleet_size"])

    nbar_keys = [
        (w, k, i, d)
        for w in Omega for k in K for i, d in sorted(states[k])
    ]
    nbar = model.addVars(nbar_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_nbar")

    reachable_arc_states = defaultdict(list)
    for k in K:
        for i, j, d, _next_d in transitions[k]:
            reachable_arc_states[k, i, j].append(d)
    reachable_arcs = {
        k: sorted((i, j) for kk, i, j in reachable_arc_states if kk == k)
        for k in K
    }

    x_keys = [
        (w, k, i, j, r)
        for w in Omega for k in K for i, j in reachable_arcs[k] for r in R
    ]
    x = model.addVars(x_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_x")
    node_r_keys = [(w, i, r) for w in Omega for i in N for r in R]
    y = model.addVars(node_r_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_y")
    z = model.addVars(node_r_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_z")
    for key in node_r_keys:
        z[key].UB = demand[key]

    model.addConstr(gp.quicksum(p[i] for i in PPL) <= P_max, name="DS_SiteBudget")
    model.addConstr(
        gp.quicksum(site_cost[i] * p[i] for i in PPL) <= selection_budget,
        name="DS_SelectionBudget",
    )
    for k in K:
        fleet = int(vehicle_types[k]["fleet_size"])
        model.addConstr(
            gp.quicksum(b[k, j] for j in vehicle_types[k]["J_k"]) == fleet,
            name=f"DSBaseAssign_k{k}",
        )
        for j in vehicle_types[k]["J_k"]:
            model.addConstr(b[k, j] <= fleet * p[j], name=f"DSBaseLink_k{k}_j{j}")

    # Vehicle conservation on the expanded acyclic graph.  Every vehicle
    # starts at a selected, available base in state zero and terminates once.
    eligible = {k: set(vehicle_types[k]["J_k"]) for k in K}
    for w in Omega:
        for k in K:
            for i, d in states[k]:
                incoming = gp.quicksum(
                    n[w, k, ip, jp, dp]
                    for ip, jp, dp, _dn in transitions_to.get((k, i, d), ())
                )
                outgoing = gp.quicksum(
                    n[w, k, ip, jp, dp]
                    for ip, jp, dp, _dn in transitions_from.get((k, i, d), ())
                )
                source = a[w, i] * b[k, i] if d == 0 and i in eligible[k] else 0
                model.addConstr(
                    outgoing + nbar[w, k, i, d] == incoming + source,
                    name=f"DSVehicleConservation_w{w}_k{k}_i{i}_d{d}",
                )

    # Residual capacity per vehicle and arc.
    residual = {}
    for w in Omega:
        for k in K:
            mode = vehicle_types[k]["mode"]
            for i, j in reachable_arcs[k]:
                sigma = max(
                    node_severity.get((w, i), 0.0),
                    node_severity.get((w, j), 0.0),
                )
                throughput = nominal_throughput.get(mode, {}).get((i, j), 0.0)
                residual[w, k, i, j] = min(
                    throughput * _degradation_factor(gamma[w, mode], sigma),
                    float(vehicle_types[k]["cap_tons"]),
                )
                count = gp.quicksum(
                    n[w, k, i, j, d] for d in reachable_arc_states[k, i, j]
                )
                model.addConstr(
                    gp.quicksum(resource_weight[r] * x[w, k, i, j, r] for r in R)
                    <= residual[w, k, i, j] * count,
                    name=f"DSVehicleCapacity_w{w}_k{k}_i{i}_j{j}",
                )
                for r in R:
                    x[w, k, i, j, r].UB = (
                        residual[w, k, i, j]
                        * int(vehicle_types[k]["fleet_size"])
                        / resource_weight[r]
                    )

    # Commodity conservation, aggregated over interchangeable vehicles.
    incoming_x = defaultdict(list)
    outgoing_x = defaultdict(list)
    for w, k, i, j, r in x_keys:
        outgoing_x[w, i, r].append(x[w, k, i, j, r])
        incoming_x[w, j, r].append(x[w, k, i, j, r])
    for w, i, r in node_r_keys:
        release = (
            releasable_fraction * inventory_if_open[i, r] * a[w, i] * p[i]
            if i in PPL_set else 0
        )
        model.addConstr(
            gp.quicksum(incoming_x[w, i, r]) + release + z[w, i, r]
            == demand[w, i, r] + y[w, i, r] + gp.quicksum(outgoing_x[w, i, r]),
            name=f"DSResourceBalance_w{w}_i{i}_r{r}",
        )

    # Node handling counts all arrivals, independent of distance state.
    for w in Omega:
        for mode in modes:
            mode_types = [k for k in K if vehicle_types[k]["mode"] == mode]
            for i in N:
                arrivals = gp.quicksum(
                    n[w, k, ip, jp, d]
                    for k in mode_types
                    for ip, jp in reachable_arcs[k] if jp == i
                    for d in reachable_arc_states[k, ip, jp]
                )
                baseline = node_handling_capacity.get((i, mode), 0.0)
                bonus = (
                    node_handling_bonus.get((i, mode), 0.0) * p[i]
                    if i in PPL_set else 0
                )
                theta = _degradation_factor(
                    gamma[w, mode], node_severity.get((w, i), 0.0)
                ) * (baseline + bonus)
                model.addConstr(arrivals <= theta, name=f"DSHandling_w{w}_m{mode}_i{i}")

    loss = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_loss")
    epsilon = 0.04
    for w in Omega:
        model.addConstr(
            loss[w]
            == gp.quicksum(penalty[i, r] * z[w, i, r] for i in N for r in R)
            + gp.quicksum(
                modal_arc_cost.get(vehicle_types[k]["mode"], {}).get((i, j), 0.0)
                * x[w, k, i, j, r]
                for ww, k, i, j, r in x_keys if ww == w
            )
            + epsilon * gp.quicksum(
                n[ww, k, i, j, d]
                for ww, k, i, j, d in n_keys if ww == w
            ),
            name=f"DSLossDefinition_w{w}",
        )
    eta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ds_eta")
    xi = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="ds_xi")
    for w in Omega:
        model.addConstr(xi[w] >= loss[w] - eta, name=f"DSCVaRExcess_w{w}")
    model.setObjective(
        eta + (1.0 / (1.0 - beta)) * gp.quicksum(prob[w] * xi[w] for w in Omega),
        GRB.MINIMIZE,
    )

    variables = {
        "p": p, "b": b, "n": n, "nbar": nbar, "x": x, "y": y, "z": z,
        "loss": loss, "eta": eta, "xi": xi,
    }

    def fix_variable(var, value):
        var.UB = float(value)
        var.LB = float(value)

    for family_name, default_value in solve_config.get("fix_defaults", {}).items():
        if family_name not in variables:
            raise ValueError(f"unknown distance-state variable family: {family_name!r}")
        for var in variables[family_name].values():
            fix_variable(var, default_value)
    for family_name, values in solve_config.get("fix", {}).items():
        if family_name not in variables:
            raise ValueError(f"unknown distance-state variable family: {family_name!r}")
        for key, value in values.items():
            if key not in variables[family_name]:
                raise KeyError(f"unknown {family_name} key in fix: {key!r}")
            fix_variable(variables[family_name][key], value)
    for family_name in solve_config.get("relax_fixed_families", []):
        if family_name not in variables:
            raise ValueError(f"unknown distance-state variable family: {family_name!r}")
        for var in variables[family_name].values():
            var.VType = GRB.CONTINUOUS
    for family_name, default_value in solve_config.get("start_defaults", {}).items():
        for var in variables[family_name].values():
            var.Start = float(default_value)
    for family_name, values in solve_config.get("start", {}).items():
        for key, value in values.items():
            if key in variables[family_name]:
                variables[family_name][key].Start = float(value)

    strategic_p = solve_config.get("strategic_p")
    p_radius = solve_config.get("p_neighborhood")
    if strategic_p is not None and p_radius is not None:
        model.addConstr(
            gp.quicksum(
                1 - p[i] if strategic_p.get(i, 0) > 0.5 else p[i] for i in PPL
            ) <= int(p_radius),
            name="DSStagedSiteNeighborhood",
        )
    strategic_b = solve_config.get("strategic_b")
    b_radius = solve_config.get("b_neighborhood")
    if strategic_b is not None and b_radius is not None:
        moved_in = model.addVars(b_keys, lb=0.0, name="ds_base_move_in")
        for key in b_keys:
            model.addConstr(moved_in[key] >= b[key] - strategic_b.get(key, 0))
        model.addConstr(
            gp.quicksum(moved_in[key] for key in b_keys) <= int(b_radius),
            name="DSStagedBaseNeighborhood",
        )

    for parameter_name, value in solve_config.get("params", {}).items():
        model.setParam(parameter_name, value)

    metadata = {
        "distance_buckets": distance_buckets,
        "distance_step_km_by_type": distance_steps,
        "distance_states_by_type": {k: len(states[k]) for k in K},
        "distance_transitions_by_type": {k: len(transitions[k]) for k in K},
        "conservative_rounding": True,
    }
    if verbose:
        print(
            "Distance-state network: "
            f"buckets={distance_buckets}, "
            f"states={sum(metadata['distance_states_by_type'].values())}, "
            f"transitions={sum(metadata['distance_transitions_by_type'].values())}"
        )

    if build_only:
        model.update()
        return {"model": model, "variables": variables, "distance_state": metadata}

    model.optimize()
    results: Dict[str, Any] = {
        "status_code": model.Status,
        "status": _status_to_string(model.Status),
        "objective_value": None,
        "eta": None,
        "scenario_losses": {},
        "xi": {},
        "node_availability": dict(a),
        "subtour_callback_stats": {
            "invocations": 0, "cuts_added": 0, "candidates_rejected": 0,
            "callback_seconds": 0.0, "errors": 0, "last_error": None,
        },
        "selected_sites": [],
        "basing": {},
        "vehicle_arcs": {},
        "vehicle_terminations": {},
        "flows": {},
        "retained": {},
        "unmet_demand": {},
        "model": model,
        "variables": variables,
        "distance_state": metadata,
    }
    if model.SolCount > 0:
        results["objective_value"] = model.ObjVal
        results["eta"] = eta.X
        results["scenario_losses"] = {w: loss[w].X for w in Omega}
        results["xi"] = {w: xi[w].X for w in Omega}
        results["selected_sites"] = [i for i in PPL if p[i].X > 0.5]
        results["basing"] = {key: int(round(var.X)) for key, var in b.items() if var.X > 0.5}
        results["vehicle_arcs"] = {
            key: int(round(var.X)) for key, var in n.items() if var.X > 0.5
        }
        results["vehicle_terminations"] = {
            key: var.X for key, var in nbar.items() if var.X > 1e-6
        }
        results["flows"] = {key: var.X for key, var in x.items() if var.X > 1e-6}
        results["retained"] = {key: var.X for key, var in y.items() if var.X > 1e-6}
        results["unmet_demand"] = {key: var.X for key, var in z.items() if var.X > 1e-6}
    return results
