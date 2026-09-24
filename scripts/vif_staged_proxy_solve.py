"""Staged VIF solve with a compact CVaR strategic proxy in Stage 1.

Stage 1 contains no arc-by-arc or distance-state vehicle-flow variables.  It
chooses binary sites and integer vehicle base counts over all scenarios using
continuous direct-service trip and delivery proxies.  Scenario-specific
shortest paths precompute reachability, distance, transport cost, and bottleneck
capacity.  Stages 2 and 3 then use the detailed distance-state formulation:

1. all-scenario strategic proxy: choose p and b under proxy CVaR;
2. one detailed routing MIP per scenario with p and b fixed; and
3. one full-scenario CVaR allocation LP with p, b, and routes fixed.

The final solution is feasible for the detailed distance-state formulation,
but the strategic choices are heuristic because Stage 1 values them with a
direct-service surrogate rather than detailed routing.

Example:

    python scripts/vif_staged_proxy_solve.py \
        --scenarios 100 --proxy-time 1200 --scenario-time 60 \
        --final-time 1800 --distance-buckets 16 --mip-gap 0.10
"""

import argparse
import copy
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios


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


def _degradation_factor(gamma: float, severity: float) -> float:
    return max(0.0, 1.0 - gamma * severity / 5.0)


def _subset_instance(
    instance: Dict[str, Any], scenario_ids: Sequence[int]
) -> Dict[str, Any]:
    """Return a scenario subset, renormalizing its probability distribution."""
    selected = list(scenario_ids)
    selected_set = set(selected)
    original = list(instance["scenarios"])
    probabilities = {w: float(instance["probability"][w]) for w in selected}
    total = sum(probabilities.values())
    if total <= 0:
        probabilities = {w: 1.0 / len(selected) for w in selected}
    else:
        probabilities = {w: value / total for w, value in probabilities.items()}

    result = dict(instance)
    result["scenarios"] = selected
    result["probability"] = probabilities
    for name in ("demand", "node_severity", "inventory_availability"):
        values = instance.get(name)
        if isinstance(values, dict):
            result[name] = {
                key: value
                for key, value in values.items()
                if isinstance(key, tuple) and key and key[0] in selected_set
            }
    if isinstance(instance.get("disaster_type"), dict):
        result["disaster_type"] = {
            w: instance["disaster_type"][w] for w in selected
        }
    residual = instance.get("modal_residual")
    if isinstance(residual, list) and len(residual) == len(original):
        by_scenario = dict(zip(original, residual))
        result["modal_residual"] = [by_scenario[w] for w in selected]
    return result


def _solver_params(
    time_limit: float, mip_gap: float, mip_focus: int = None
) -> Dict[str, Any]:
    params = {
        "TimeLimit": float(time_limit),
        "MIPGap": float(mip_gap),
        "CutPasses": 1,
        "DegenMoves": 0,
    }
    if mip_focus is not None:
        params["MIPFocus"] = int(mip_focus)
    return params


def _dijkstra_labels(
    source: int,
    adjacency: Dict[int, List[Tuple[int, float, float, float]]],
) -> Dict[int, Tuple[float, float, float]]:
    """Return node -> (distance, bottleneck capacity, transport cost)."""
    labels: Dict[int, Tuple[float, float, float]] = {
        source: (0.0, math.inf, 0.0)
    }
    queue = [(0.0, 0.0, source)]
    while queue:
        distance, transport_cost, i = heapq.heappop(queue)
        current_distance, current_bottleneck, current_cost = labels[i]
        if distance > current_distance + 1e-9:
            continue
        if (
            abs(distance - current_distance) <= 1e-9
            and transport_cost > current_cost + 1e-9
        ):
            continue
        for j, arc_distance, arc_capacity, arc_cost in adjacency.get(i, ()):
            next_distance = current_distance + arc_distance
            next_cost = current_cost + arc_cost
            next_bottleneck = min(current_bottleneck, arc_capacity)
            previous = labels.get(j)
            improves = previous is None or next_distance < previous[0] - 1e-9
            tie_improves = (
                previous is not None
                and abs(next_distance - previous[0]) <= 1e-9
                and (
                    next_cost < previous[2] - 1e-9
                    or (
                        abs(next_cost - previous[2]) <= 1e-9
                        and next_bottleneck > previous[1] + 1e-9
                    )
                )
            )
            if improves or tie_improves:
                labels[j] = (next_distance, next_bottleneck, next_cost)
                heapq.heappush(queue, (next_distance, next_cost, j))
    return labels


def build_service_options(
    instance: Dict[str, Any], return_to_base: bool = False
) -> Dict[Tuple[int, str, int, int], Dict[str, float]]:
    """Precompute feasible direct-service proxies for every scenario and type.

    The underlying path can contain several physical network arcs, but Stage 1
    represents the entire base-to-demand mission with one continuous trip
    variable.  Arc distances include the same per-movement turnaround-distance
    charge used by the detailed distance-state model.
    """
    scenarios = instance["scenarios"]
    nodes = instance["nodes"]
    vehicle_types = instance["vehicle_types"]
    modal_arcs = instance["modal_arcs"]
    arc_distance = instance["modal_arc_distance"]
    arc_cost = instance["modal_arc_cost"]
    nominal = instance["nominal_throughput"]
    severity = instance["node_severity"]
    disaster_type = instance["disaster_type"]
    degradation = instance["degradation_matrix"]
    alpha = float(instance.get("alpha", 1.0))
    options: Dict[Tuple[int, str, int, int], Dict[str, float]] = {}

    for w in scenarios:
        for k, data in vehicle_types.items():
            mode = data["mode"]
            payload = float(data["cap_tons"])
            turnaround_distance = float(data["pi_k"])
            distance_budget = float(data["D_k"])
            gamma = (
                float(degradation.get(mode, {}).get(disaster_type[w], 0.0))
                * alpha
            )
            forward: Dict[int, List[Tuple[int, float, float, float]]] = defaultdict(list)
            reverse: Dict[int, List[Tuple[int, float, float, float]]] = defaultdict(list)
            for i, j in modal_arcs[mode]:
                degradation_factor = _degradation_factor(
                    gamma,
                    max(severity.get((w, i), 0.0), severity.get((w, j), 0.0)),
                )
                usable_capacity = min(
                    payload,
                    float(nominal.get(mode, {}).get((i, j), 0.0))
                    * degradation_factor,
                )
                if usable_capacity <= 1e-12:
                    continue
                distance = (
                    float(arc_distance.get(mode, {}).get((i, j), 0.0))
                    + turnaround_distance
                )
                if distance <= 0:
                    continue
                cost = float(arc_cost.get(mode, {}).get((i, j), 0.0))
                forward[i].append((j, distance, usable_capacity, cost))
                reverse[j].append((i, distance, usable_capacity, cost))

            for base in data["J_k"]:
                outward = _dijkstra_labels(base, forward)
                returning = (
                    _dijkstra_labels(base, reverse) if return_to_base else {}
                )
                for destination in nodes:
                    if destination == base or destination not in outward:
                        continue
                    out_distance, out_capacity, out_cost = outward[destination]
                    mission_distance = out_distance
                    mission_capacity = out_capacity
                    if return_to_base:
                        if destination not in returning:
                            continue
                        back_distance, back_capacity, _back_cost = returning[destination]
                        mission_distance += back_distance
                        mission_capacity = min(mission_capacity, back_capacity)
                    if mission_distance <= distance_budget + 1e-9:
                        options[w, k, base, destination] = {
                            "distance": mission_distance,
                            "capacity": mission_capacity,
                            # Cargo travels outbound; the empty return does not
                            # receive a per-unit resource-flow cost.
                            "transport_cost": out_cost,
                        }
    return options


def solve_strategic_proxy(
    instance: Dict[str, Any],
    time_limit: float = 1200.0,
    mip_gap: float = 0.10,
    return_to_base: bool = False,
    verbose: bool = True,
    build_only: bool = False,
) -> Dict[str, Any]:
    """Build and solve the compact all-scenario CVaR strategic surrogate."""
    N = list(instance["nodes"])
    PPL = list(instance["ppl_nodes"])
    PPL_set = set(PPL)
    R = list(instance["commodities"])
    Omega = list(instance["scenarios"])
    modes = list(instance["modes"])
    vehicle_types = instance["vehicle_types"]
    K = sorted(vehicle_types)
    demand = instance["demand"]
    probability = instance["probability"]
    beta = float(instance["beta"])
    resource_weight = instance["resource_weight"]
    severity = instance["node_severity"]
    disaster_type = instance["disaster_type"]
    degradation = instance["degradation_matrix"]
    alpha = float(instance.get("alpha", 1.0))
    releasable_fraction = 1.0 - float(instance.get("safety_stock_fraction", 0.0))

    for r in R:
        if float(resource_weight[r]) <= 0:
            raise ValueError(f"resource_weight[{r!r}] must be positive")

    print("[stage 1/3] building all-scenario strategic proxy", flush=True)
    options = build_service_options(instance, return_to_base=return_to_base)
    model = gp.Model("vif_strategic_proxy_cvar")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.Method = 2
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.MIPFocus = 1

    p = model.addVars(PPL, vtype=GRB.BINARY, name="proxy_p")
    b_keys = [(k, j) for k in K for j in vehicle_types[k]["J_k"]]
    b = model.addVars(b_keys, lb=0, vtype=GRB.INTEGER, name="proxy_b")
    for k, j in b_keys:
        b[k, j].UB = int(vehicle_types[k]["fleet_size"])

    trip_keys = sorted(options)
    trips = model.addVars(trip_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_trips")
    delivery_keys = [key + (r,) for key in trip_keys for r in R]
    delivery = model.addVars(
        delivery_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_delivery"
    )
    local_keys = [(w, j, r) for w in Omega for j in PPL for r in R]
    local = model.addVars(local_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_local")
    unmet_keys = [(w, i, r) for w in Omega for i in N for r in R]
    unmet = model.addVars(unmet_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_unmet")
    for key in unmet_keys:
        unmet[key].UB = float(demand[key])

    model.addConstr(
        gp.quicksum(p[j] for j in PPL) <= int(instance["P_max"]),
        name="ProxySiteLimit",
    )
    model.addConstr(
        gp.quicksum(float(instance["site_cost"][j]) * p[j] for j in PPL)
        <= float(instance["selection_budget"]),
        name="ProxySiteBudget",
    )
    for k in K:
        fleet = int(vehicle_types[k]["fleet_size"])
        model.addConstr(
            gp.quicksum(b[k, j] for j in vehicle_types[k]["J_k"]) == fleet,
            name=f"ProxyBaseAssign_k{k}",
        )
        for j in vehicle_types[k]["J_k"]:
            model.addConstr(
                b[k, j] <= fleet * p[j], name=f"ProxyBaseLink_k{k}_j{j}"
            )

    # Payload supplied by a fractional direct-service trip.
    for key in trip_keys:
        w, k, j, i = key
        model.addConstr(
            gp.quicksum(
                float(resource_weight[r]) * delivery[w, k, j, i, r] for r in R
            ) <= float(options[key]["capacity"]) * trips[key],
            name=f"ProxyTripCapacity_w{w}_k{k}_j{j}_i{i}",
        )

    # Aggregate distance available to the fleet of each type at each base.
    option_destinations: Dict[Tuple[int, str, int], List[int]] = defaultdict(list)
    for w, k, j, i in trip_keys:
        option_destinations[w, k, j].append(i)
    for w in Omega:
        for k in K:
            distance_budget = float(vehicle_types[k]["D_k"])
            for j in vehicle_types[k]["J_k"]:
                destinations = option_destinations.get((w, k, j), [])
                base_available = 0.0 if severity.get((w, j), 0.0) >= 1.0 else 1.0
                model.addConstr(
                    gp.quicksum(
                        float(options[w, k, j, i]["distance"])
                        * trips[w, k, j, i]
                        for i in destinations
                    ) <= distance_budget * base_available * b[k, j],
                    name=f"ProxyFleetDistance_w{w}_k{k}_j{j}",
                )
                if not return_to_base:
                    # A one-way direct-service proxy consumes one vehicle for
                    # the horizon; without a return leg it cannot reappear at
                    # the base to perform another independent mission.
                    model.addConstr(
                        gp.quicksum(trips[w, k, j, i] for i in destinations)
                        <= base_available * b[k, j],
                        name=f"ProxyOneWayVehicleCount_w{w}_k{k}_j{j}",
                    )

    # Inventory can satisfy local demand or outbound proxy deliveries.
    outbound_delivery: Dict[Tuple[int, int, str], List] = defaultdict(list)
    inbound_delivery: Dict[Tuple[int, int, str], List] = defaultdict(list)
    arrivals: Dict[Tuple[int, str, int], List] = defaultdict(list)
    for w, k, j, i in trip_keys:
        arrivals[w, vehicle_types[k]["mode"], i].append(trips[w, k, j, i])
        for r in R:
            outbound_delivery[w, j, r].append(delivery[w, k, j, i, r])
            inbound_delivery[w, i, r].append(delivery[w, k, j, i, r])

    for w in Omega:
        for j in PPL:
            source_available = 0.0 if severity.get((w, j), 0.0) >= 1.0 else 1.0
            for r in R:
                model.addConstr(
                    local[w, j, r]
                    + gp.quicksum(outbound_delivery[w, j, r])
                    <= releasable_fraction
                    * float(instance["inventory_if_open"][j, r])
                    * source_available
                    * p[j],
                    name=f"ProxyInventory_w{w}_j{j}_r{r}",
                )

    for w in Omega:
        for i in N:
            for r in R:
                local_service = local[w, i, r] if i in PPL_set else 0
                model.addConstr(
                    local_service
                    + gp.quicksum(inbound_delivery[w, i, r])
                    + unmet[w, i, r]
                    == float(demand[w, i, r]),
                    name=f"ProxyDemand_w{w}_i{i}_r{r}",
                )

    # Retain scenario-degraded node handling as an aggregate arrival limit.
    for w in Omega:
        for mode in modes:
            gamma = (
                float(degradation.get(mode, {}).get(disaster_type[w], 0.0))
                * alpha
            )
            for i in N:
                baseline = float(instance["node_handling_capacity"].get((i, mode), 0.0))
                bonus = (
                    float(instance["node_handling_bonus"].get((i, mode), 0.0)) * p[i]
                    if i in PPL_set else 0
                )
                residual_handling = _degradation_factor(
                    gamma, float(severity.get((w, i), 0.0))
                ) * (baseline + bonus)
                model.addConstr(
                    gp.quicksum(arrivals[w, mode, i]) <= residual_handling,
                    name=f"ProxyHandling_w{w}_m{mode}_i{i}",
                )

    loss = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_loss")
    epsilon = 0.04
    delivery_keys_by_scenario: Dict[int, List[Tuple]] = defaultdict(list)
    trip_keys_by_scenario: Dict[int, List[Tuple]] = defaultdict(list)
    for key in delivery_keys:
        delivery_keys_by_scenario[key[0]].append(key)
    for key in trip_keys:
        trip_keys_by_scenario[key[0]].append(key)
    for w in Omega:
        model.addConstr(
            loss[w]
            == gp.quicksum(
                float(instance["penalty"][i, r]) * unmet[w, i, r]
                for i in N for r in R
            )
            + gp.quicksum(
                float(options[ww, k, j, i]["transport_cost"])
                * delivery[ww, k, j, i, r]
                for ww, k, j, i, r in delivery_keys_by_scenario[w]
            )
            + epsilon * gp.quicksum(
                trips[ww, k, j, i]
                for ww, k, j, i in trip_keys_by_scenario[w]
            ),
            name=f"ProxyLoss_w{w}",
        )
    eta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_eta")
    xi = model.addVars(Omega, lb=0.0, vtype=GRB.CONTINUOUS, name="proxy_xi")
    for w in Omega:
        model.addConstr(xi[w] >= loss[w] - eta, name=f"ProxyCVaR_w{w}")
    model.setObjective(
        eta
        + (1.0 / (1.0 - beta))
        * gp.quicksum(float(probability[w]) * xi[w] for w in Omega),
        GRB.MINIMIZE,
    )

    variables = {
        "p": p,
        "b": b,
        "trips": trips,
        "delivery": delivery,
        "local": local,
        "unmet": unmet,
        "loss": loss,
        "eta": eta,
        "xi": xi,
    }
    metadata = {
        "service_options": len(options),
        "trip_variables": len(trip_keys),
        "delivery_variables": len(delivery_keys),
        "return_to_base": bool(return_to_base),
    }
    if verbose:
        print(
            "Strategic proxy: "
            f"service_options={len(options)}, trips={len(trip_keys)}, "
            f"deliveries={len(delivery_keys)}",
            flush=True,
        )
    if build_only:
        model.update()
        return {"model": model, "variables": variables, "metadata": metadata}

    model.optimize()
    results: Dict[str, Any] = {
        "model": model,
        "variables": variables,
        "metadata": metadata,
        "status": _status_to_string(model.Status),
        "objective_value": None,
        "eta": None,
        "scenario_losses": {},
        "selected_sites": [],
        "basing": {},
    }
    if model.SolCount > 0:
        results["objective_value"] = float(model.ObjVal)
        results["eta"] = float(eta.X)
        results["scenario_losses"] = {w: float(loss[w].X) for w in Omega}
        results["selected_sites"] = [j for j in PPL if p[j].X > 0.5]
        results["basing"] = {
            key: int(round(var.X)) for key, var in b.items() if var.X > 0.5
        }
    return results


def _require_solution(results: Dict[str, Any], label: str) -> None:
    if results["model"].SolCount <= 0:
        raise RuntimeError(
            f"{label} produced no feasible solution (status={results.get('status')})"
        )


def _dispose(results: Dict[str, Any]) -> None:
    if results.get("model") is not None:
        results["model"].dispose()


def solve_staged_proxy_vif(
    instance: Dict[str, Any],
    proxy_time: float = 1200.0,
    scenario_time: float = 60.0,
    final_time: float = 1800.0,
    mip_gap: float = 0.10,
    distance_buckets: int = 16,
    return_to_base: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run compact strategic proxy, detailed routing, and final CVaR LP."""
    started = time.time()
    proxy_results = solve_strategic_proxy(
        instance,
        time_limit=proxy_time,
        mip_gap=mip_gap,
        return_to_base=return_to_base,
        verbose=verbose,
    )
    _require_solution(proxy_results, "strategic proxy")
    p_values = {
        i: int(var.X > 0.5) for i, var in proxy_results["variables"]["p"].items()
    }
    b_values = {
        key: int(round(var.X))
        for key, var in proxy_results["variables"]["b"].items()
    }
    proxy_summary = {
        "status": proxy_results["status"],
        "objective": proxy_results["objective_value"],
        "eta": proxy_results["eta"],
        "gap": proxy_results["model"].MIPGap,
        "runtime": proxy_results["model"].Runtime,
        "selected_sites": proxy_results["selected_sites"],
        "basing": [
            [k, j, value] for (k, j), value in sorted(b_values.items()) if value
        ],
        "metadata": proxy_results["metadata"],
    }
    _dispose(proxy_results)

    print("[stage 2/3] detailed fixed-strategy scenario routing", flush=True)
    fixed_strategy = {"p": p_values, "b": b_values}
    routing_values: Dict[Tuple, int] = {}
    scenario_summaries = []
    for position, w in enumerate(instance["scenarios"], start=1):
        print(f"  scenario {w} ({position}/{len(instance['scenarios'])})", flush=True)
        results = solve_stochastic_cvar(
            _subset_instance(instance, [w]),
            vehicle_formulation="distance_state",
            verbose=verbose,
            vif_solve_config={
                "distance_buckets": int(distance_buckets),
                "fix": fixed_strategy,
                "params": _solver_params(scenario_time, mip_gap, mip_focus=1),
            },
        )
        _require_solution(results, f"scenario {w} routing")
        routing_values.update({
            key: int(round(var.X))
            for key, var in results["variables"]["n"].items()
            if var.X > 0.5
        })
        model = results["model"]
        scenario_summaries.append({
            "scenario": w,
            "status": results["status"],
            "objective": results["objective_value"],
            "gap": model.MIPGap if model.IsMIP and model.SolCount else 0.0,
            "runtime": model.Runtime,
            "selected_distance_state_arcs": len(results["vehicle_arcs"]),
        })
        _dispose(results)

    print(
        "[stage 3/3] full CVaR allocation LP with strategy and routes fixed",
        flush=True,
    )
    final_results = solve_stochastic_cvar(
        instance,
        vehicle_formulation="distance_state",
        verbose=verbose,
        vif_solve_config={
            "distance_buckets": int(distance_buckets),
            "fix_defaults": {"n": 0.0},
            "fix": {"p": p_values, "b": b_values, "n": routing_values},
            "relax_fixed_families": ["p", "b", "n"],
            "params": _solver_params(final_time, mip_gap),
        },
    )
    _require_solution(final_results, "final CVaR allocation")
    final_model = final_results["model"]
    summary = {
        "elapsed_seconds": time.time() - started,
        "method": "all_scenario_strategic_proxy_then_detailed_routing",
        "scenario_count": len(instance["scenarios"]),
        "distance_buckets": int(distance_buckets),
        "proxy_return_to_base": bool(return_to_base),
        "strategic_proxy": proxy_summary,
        "scenario_routing": scenario_summaries,
        "fixed_route_state_arcs": len(routing_values),
        "final_status": final_results["status"],
        "final_objective": final_results["objective_value"],
        "final_eta": final_results["eta"],
        "final_scenario_losses": final_results["scenario_losses"],
        "final_gap": (
            final_model.MIPGap
            if final_model.IsMIP and final_model.SolCount
            else 0.0
        ),
        "final_runtime": final_model.Runtime,
        "final_selected_sites": final_results["selected_sites"],
    }
    return final_results, summary


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--proxy-time", type=float, default=1200.0)
    parser.add_argument("--scenario-time", type=float, default=60.0)
    parser.add_argument("--final-time", type=float, default=1800.0)
    parser.add_argument("--mip-gap", type=float, default=0.10)
    parser.add_argument("--distance-buckets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--proxy-return-to-base",
        action="store_true",
        help="Require each Stage 1 direct-service proxy mission to return to base.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir", default=os.path.join("output", "vif_staged_proxy")
    )
    args = parser.parse_args()
    if args.scenarios is not None and args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if args.distance_buckets <= 0:
        parser.error("--distance-buckets must be positive")

    params = copy.deepcopy(load_parameters())
    scenario_count = args.scenarios or int(params["num_scenarios"])
    seed = args.seed if args.seed is not None else int(params["seed"])
    locations = load_locations()
    graph = build_graph(locations)
    scenarios = generate_scenarios(
        graph, locations, num_scenarios=scenario_count, seed=seed
    )
    instance = build_stochastic_instance(
        locations=locations,
        scenarios=scenarios,
        params=params,
        alpha=args.alpha,
    )
    final_results, summary = solve_staged_proxy_vif(
        instance,
        proxy_time=args.proxy_time,
        scenario_time=args.scenario_time,
        final_time=args.final_time,
        mip_gap=args.mip_gap,
        distance_buckets=args.distance_buckets,
        return_to_base=args.proxy_return_to_base,
        verbose=not args.quiet,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"summary_{stamp}.json")
    solution_path = os.path.join(args.output_dir, f"solution_{stamp}.sol")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(_json_ready(summary), stream, indent=2, sort_keys=True)
    final_results["model"].write(solution_path)
    print(f"staged proxy summary -> {summary_path}")
    print(f"final solution -> {solution_path}")
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    _dispose(final_results)


if __name__ == "__main__":
    main()
