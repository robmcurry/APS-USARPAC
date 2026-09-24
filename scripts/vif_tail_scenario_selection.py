"""Tail-aware strategic scenario selection for the distance-state VIF model.

This script is deliberately separate from ``vif_staged_solve.py``.  It:

1. solves the continuous relaxation over every generated scenario;
2. builds operational feature vectors and separates a buffered CVaR tail;
3. selects actual scenarios with weighted k-medoids in the tail and body;
4. assigns each representative the probability mass of its cluster;
5. optionally solves the strategic integer model on those representatives;
6. evaluates the resulting site/base decision once on every scenario; and
7. reports how the relaxed and evaluated integer tails differ.

The selected representatives and their aggregate probability weights form a
distributional approximation: their weights sum to one.  This is different
from conditioning on a selected subset and renormalizing its original weights.

Example (selection only):

    python scripts/vif_tail_scenario_selection.py \
        --scenarios 100 --representatives 20 --selection-only

Example (selection, one strategic solve, and one full evaluation):

    python scripts/vif_tail_scenario_selection.py \
        --scenarios 100 --representatives 20 \
        --lp-time 1800 --strategic-time 1200 --evaluation-time 60
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios


def _subset_instance(
    instance: Dict[str, Any],
    scenario_ids: Sequence[int],
    probabilities: Dict[int, float],
) -> Dict[str, Any]:
    """Return an instance subset with caller-supplied representative weights."""
    selected = list(scenario_ids)
    selected_set = set(selected)
    original = list(instance["scenarios"])
    missing = selected_set - set(original)
    if missing:
        raise ValueError(f"unknown scenario ids: {sorted(missing)}")
    if set(probabilities) != selected_set:
        raise ValueError("probability keys must exactly match selected scenarios")
    total_probability = sum(float(probabilities[w]) for w in selected)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"representative probabilities must sum to one; got {total_probability}"
        )

    result = dict(instance)
    result["scenarios"] = selected
    result["probability"] = {w: float(probabilities[w]) for w in selected}

    for name in ("demand", "node_severity", "inventory_availability"):
        values = instance.get(name)
        if isinstance(values, dict):
            result[name] = {
                key: value
                for key, value in values.items()
                if isinstance(key, tuple) and key and key[0] in selected_set
            }

    values = instance.get("disaster_type")
    if isinstance(values, dict):
        result["disaster_type"] = {w: values[w] for w in selected}

    residual = instance.get("modal_residual")
    if isinstance(residual, list) and len(residual) == len(original):
        residual_by_scenario = dict(zip(original, residual))
        result["modal_residual"] = [residual_by_scenario[w] for w in selected]
    return result


def _single_scenario_instance(
    instance: Dict[str, Any], scenario_id: int
) -> Dict[str, Any]:
    return _subset_instance(instance, [scenario_id], {scenario_id: 1.0})


def _require_solution(results: Dict[str, Any], label: str) -> None:
    model = results["model"]
    if model.SolCount <= 0:
        raise RuntimeError(
            f"{label} produced no feasible solution (status={results.get('status')})"
        )


def _dispose(results: Dict[str, Any]) -> None:
    model = results.get("model")
    if model is not None:
        model.dispose()


def _solver_params(
    time_limit: float, mip_gap: float = None, mip_focus: int = None
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"TimeLimit": float(time_limit)}
    if mip_gap is not None:
        params["MIPGap"] = float(mip_gap)
    if mip_focus is not None:
        params["MIPFocus"] = int(mip_focus)
    return params


def _weighted_cvar(
    losses: Dict[int, float], probabilities: Dict[int, float], beta: float
) -> Tuple[float, float, List[int]]:
    """Return (CVaR, lower weighted beta-quantile, descending tail ids)."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    if set(losses) != set(probabilities):
        raise ValueError("loss and probability scenario sets differ")
    total = sum(float(value) for value in probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"probabilities must sum to one; got {total}")

    ascending = sorted(losses, key=lambda w: (float(losses[w]), w))
    cumulative = 0.0
    eta = float(losses[ascending[-1]])
    for w in ascending:
        cumulative += float(probabilities[w])
        if cumulative + 1e-12 >= beta:
            eta = float(losses[w])
            break
    cvar = eta + sum(
        float(probabilities[w]) * max(0.0, float(losses[w]) - eta)
        for w in losses
    ) / (1.0 - beta)

    tail = []
    tail_mass = 0.0
    for w in sorted(losses, key=lambda s: (-float(losses[s]), s)):
        tail.append(w)
        tail_mass += float(probabilities[w])
        if tail_mass + 1e-12 >= 1.0 - beta:
            break
    return cvar, eta, tail


def _standardize(rows: Dict[int, List[float]]) -> Dict[int, List[float]]:
    ids = sorted(rows)
    if not ids:
        return {}
    width = len(rows[ids[0]])
    means = [sum(rows[w][j] for w in ids) / len(ids) for j in range(width)]
    scales = []
    for j in range(width):
        variance = sum((rows[w][j] - means[j]) ** 2 for w in ids) / len(ids)
        scales.append(math.sqrt(variance) if variance > 1e-20 else 1.0)
    return {
        w: [(rows[w][j] - means[j]) / scales[j] for j in range(width)]
        for w in ids
    }


def _scenario_features(
    instance: Dict[str, Any],
    relaxed_losses: Dict[int, float],
    loss_feature_weight: float,
) -> Dict[int, List[float]]:
    """Build standardized loss, demand, damage, geography, and type features."""
    scenarios = sorted(instance["scenarios"])
    nodes = sorted(instance["nodes"])
    resources = sorted(instance["commodities"])
    modes = sorted(instance["modes"])
    disaster_types = sorted(set(instance["disaster_type"].values()))
    raw: Dict[int, List[float]] = {}

    for w in scenarios:
        severity = [float(instance["node_severity"].get((w, i), 0.0)) for i in nodes]
        total_demand_by_resource = [
            sum(float(instance["demand"].get((w, i, r), 0.0)) for i in nodes)
            for r in resources
        ]
        demand_by_node = [
            sum(float(instance["demand"].get((w, i, r), 0.0)) for r in resources)
            for i in nodes
        ]
        damage_by_mode = []
        for mode in modes:
            gamma = (
                float(instance["degradation_matrix"].get(mode, {}).get(
                    instance["disaster_type"][w], 0.0
                ))
                * float(instance.get("alpha", 1.0))
            )
            arcs = instance["modal_arcs"].get(mode, [])
            if arcs:
                damage_by_mode.append(
                    sum(
                        min(
                            1.0,
                            gamma
                            * max(
                                instance["node_severity"].get((w, i), 0.0),
                                instance["node_severity"].get((w, j), 0.0),
                            )
                            / 5.0,
                        )
                        for i, j in arcs
                    ) / len(arcs)
                )
            else:
                damage_by_mode.append(0.0)

        raw[w] = (
            [float(relaxed_losses[w])]
            + total_demand_by_resource
            + [
                max(severity) if severity else 0.0,
                sum(severity) / len(severity) if severity else 0.0,
                float(sum(value >= 1.0 for value in severity)),
            ]
            + damage_by_mode
            # Node-level values preserve the geographic damage/demand pattern.
            + severity
            + demand_by_node
            + [
                1.0 if instance["disaster_type"][w] == disaster_type else 0.0
                for disaster_type in disaster_types
            ]
        )

    standardized = _standardize(raw)
    for w in standardized:
        standardized[w][0] *= float(loss_feature_weight)
    return standardized


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _weighted_k_medoids(
    scenario_ids: Sequence[int],
    features: Dict[int, List[float]],
    probabilities: Dict[int, float],
    count: int,
    max_iterations: int = 50,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Deterministic weighted k-medoids with farthest-first initialization."""
    ids = sorted(scenario_ids)
    if count <= 0:
        return [], {}
    if count >= len(ids):
        return ids, {w: [w] for w in ids}

    # Start with the scenario nearest the weighted centroid.
    total_weight = sum(float(probabilities[w]) for w in ids)
    centroid = [
        sum(float(probabilities[w]) * features[w][j] for w in ids) / total_weight
        for j in range(len(features[ids[0]]))
    ]
    medoids = [min(ids, key=lambda w: (_distance(features[w], centroid), w))]
    while len(medoids) < count:
        candidates = [w for w in ids if w not in medoids]
        next_medoid = max(
            candidates,
            key=lambda w: (
                min(_distance(features[w], features[m]) for m in medoids),
                float(probabilities[w]),
                -w,
            ),
        )
        medoids.append(next_medoid)

    for _iteration in range(max_iterations):
        clusters: Dict[int, List[int]] = {m: [] for m in medoids}
        for w in ids:
            chosen = min(
                medoids,
                key=lambda m: (_distance(features[w], features[m]), m),
            )
            clusters[chosen].append(w)

        updated = []
        for medoid in medoids:
            members = clusters[medoid]
            best = min(
                members,
                key=lambda candidate: (
                    sum(
                        float(probabilities[w])
                        * _distance(features[w], features[candidate])
                        for w in members
                    ),
                    candidate,
                ),
            )
            updated.append(best)
        updated = sorted(updated)
        if set(updated) == set(medoids):
            medoids = updated
            break
        medoids = updated

    clusters = {m: [] for m in medoids}
    for w in ids:
        chosen = min(medoids, key=lambda m: (_distance(features[w], features[m]), m))
        clusters[chosen].append(w)
    return sorted(medoids), clusters


def _cluster_weights(
    clusters: Dict[int, List[int]], original_probabilities: Dict[int, float]
) -> Dict[int, float]:
    weights = {
        representative: sum(
            float(original_probabilities[w]) for w in members
        )
        for representative, members in clusters.items()
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("cluster probability is zero")
    # Only correct floating-point accumulation drift; do not invent equal weights.
    return {w: value / total for w, value in weights.items()}


def select_tail_aware_scenarios(
    instance: Dict[str, Any],
    relaxed_losses: Dict[int, float],
    representative_count: int,
    tail_buffer_factor: float = 1.25,
    tail_representative_share: float = 0.60,
    loss_feature_weight: float = 2.0,
) -> Dict[str, Any]:
    """Select weighted medoids separately from relaxed tail and body."""
    scenarios = list(instance["scenarios"])
    probabilities = instance["probability"]
    beta = float(instance["beta"])
    if not 1 <= representative_count <= len(scenarios):
        raise ValueError("representative_count must be between 1 and scenario count")
    if tail_buffer_factor < 1.0:
        raise ValueError("tail_buffer_factor must be at least one")
    if not 0.0 <= tail_representative_share <= 1.0:
        raise ValueError("tail_representative_share must lie in [0,1]")

    ranked = sorted(scenarios, key=lambda w: (-float(relaxed_losses[w]), w))
    target_mass = min(1.0, tail_buffer_factor * (1.0 - beta))
    tail_candidates = []
    mass = 0.0
    for w in ranked:
        tail_candidates.append(w)
        mass += float(probabilities[w])
        if mass + 1e-12 >= target_mass:
            break
    tail_set = set(tail_candidates)
    body_candidates = [w for w in scenarios if w not in tail_set]

    features = _scenario_features(
        instance, relaxed_losses, loss_feature_weight=loss_feature_weight
    )
    if not body_candidates or representative_count == 1:
        tail_count = representative_count
        body_count = 0
    else:
        tail_count = max(1, int(round(representative_count * tail_representative_share)))
        tail_count = min(tail_count, len(tail_candidates))
        body_count = representative_count - tail_count
        if body_count <= 0:
            body_count = 1
            tail_count = representative_count - 1
        if body_count > len(body_candidates):
            difference = body_count - len(body_candidates)
            body_count = len(body_candidates)
            tail_count += difference

    tail_medoids, tail_clusters = _weighted_k_medoids(
        tail_candidates, features, probabilities, tail_count
    )
    body_medoids, body_clusters = _weighted_k_medoids(
        body_candidates, features, probabilities, body_count
    )
    clusters = {**tail_clusters, **body_clusters}
    weights = _cluster_weights(clusters, probabilities)
    representatives = sorted(tail_medoids + body_medoids)
    return {
        "representatives": representatives,
        "weights": weights,
        "clusters": clusters,
        "tail_candidates": tail_candidates,
        "tail_representatives": tail_medoids,
        "body_representatives": body_medoids,
        "buffered_tail_probability": mass,
    }


def _solve_relaxation(
    instance: Dict[str, Any], distance_buckets: int, time_limit: float, verbose: bool
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    print("[screen 1] all-scenario continuous relaxation", flush=True)
    results = solve_stochastic_cvar(
        instance,
        vehicle_formulation="distance_state",
        verbose=verbose,
        vif_solve_config={
            "distance_buckets": int(distance_buckets),
            "relax_fixed_families": ["p", "b", "n"],
            "params": _solver_params(time_limit),
        },
    )
    _require_solution(results, "all-scenario relaxation")
    model = results["model"]
    relaxed_cvar = float(model.ObjVal)
    first_pass_runtime = float(model.Runtime)

    # CVaR does not reward improvements to scenarios already below eta.  Use
    # a lexicographic second pass so their loss values are meaningful inputs
    # to clustering: preserve the best CVaR (within numerical tolerance),
    # then minimize probability-weighted expected loss.
    cvar_expression = model.getObjective()
    cvar_tolerance = max(1e-6, 1e-7 * max(1.0, abs(relaxed_cvar)))
    model.addConstr(
        cvar_expression <= relaxed_cvar + cvar_tolerance,
        name="TailScreenPreserveRelaxedCVaR",
    )
    loss_variables = results["variables"]["loss"]
    model.setObjective(
        sum(
            float(instance["probability"][w]) * loss_variables[w]
            for w in instance["scenarios"]
        )
    )
    model.ModelSense = 1
    model.setParam("TimeLimit", max(1.0, float(time_limit) - first_pass_runtime))
    model.optimize()
    if model.SolCount <= 0:
        raise RuntimeError("expected-loss tie-break pass produced no solution")
    losses = {int(w): float(loss_variables[w].X) for w in instance["scenarios"]}
    summary = {
        "status": results["status"],
        "cvar_objective": relaxed_cvar,
        "cvar_preservation_tolerance": cvar_tolerance,
        "eta_after_expected_loss_tiebreak": float(results["variables"]["eta"].X),
        "expected_loss_tiebreak_objective": float(model.ObjVal),
        "runtime": first_pass_runtime + float(model.Runtime),
        "scenario_losses": losses,
    }
    _dispose(results)
    return losses, summary


def _solve_strategic_ip(
    instance: Dict[str, Any],
    clusters: Dict[int, List[int]],
    distance_buckets: int,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
) -> Tuple[Dict[int, int], Dict[Tuple[str, int], int], Dict[str, Any]]:
    weights = _cluster_weights(clusters, instance["probability"])
    representatives = sorted(clusters)
    strategic_instance = _subset_instance(instance, representatives, weights)
    results = solve_stochastic_cvar(
        strategic_instance,
        vehicle_formulation="distance_state",
        verbose=verbose,
        vif_solve_config={
            "distance_buckets": int(distance_buckets),
            "params": _solver_params(time_limit, mip_gap=mip_gap, mip_focus=1),
        },
    )
    _require_solution(results, "strategic integer solve")
    p_values = {
        int(key): int(var.X > 0.5)
        for key, var in results["variables"]["p"].items()
    }
    b_values = {
        key: int(round(var.X))
        for key, var in results["variables"]["b"].items()
    }
    model = results["model"]
    summary = {
        "representatives": representatives,
        "weights": weights,
        "status": results["status"],
        "objective": results["objective_value"],
        "eta": results["eta"],
        "gap": model.MIPGap if model.IsMIP and model.SolCount else 0.0,
        "runtime": model.Runtime,
        "selected_sites": results["selected_sites"],
        "basing": [
            [k, j, value] for (k, j), value in sorted(b_values.items()) if value
        ],
    }
    _dispose(results)
    return p_values, b_values, summary


def _evaluate_strategy(
    instance: Dict[str, Any],
    p_values: Dict[int, int],
    b_values: Dict[Tuple[str, int], int],
    distance_buckets: int,
    time_limit: float,
    mip_gap: float,
    verbose: bool,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    print("[evaluate] fixed strategy on every scenario", flush=True)
    losses: Dict[int, float] = {}
    summaries = []
    fixed_strategy = {"p": p_values, "b": b_values}
    for position, w in enumerate(instance["scenarios"], start=1):
        print(f"  scenario {w} ({position}/{len(instance['scenarios'])})", flush=True)
        results = solve_stochastic_cvar(
            _single_scenario_instance(instance, w),
            vehicle_formulation="distance_state",
            verbose=verbose,
            vif_solve_config={
                "distance_buckets": int(distance_buckets),
                "fix": fixed_strategy,
                "params": _solver_params(time_limit, mip_gap=mip_gap, mip_focus=1),
            },
        )
        _require_solution(results, f"scenario {w} evaluation")
        loss = float(results["scenario_losses"][w])
        losses[w] = loss
        model = results["model"]
        summaries.append({
            "scenario": w,
            "loss": loss,
            "status": results["status"],
            "gap": model.MIPGap if model.IsMIP and model.SolCount else 0.0,
            "runtime": model.Runtime,
        })
        _dispose(results)
    return losses, summaries


def run_tail_selection(
    instance: Dict[str, Any],
    representative_count: int = 20,
    distance_buckets: int = 16,
    tail_buffer_factor: float = 1.25,
    tail_representative_share: float = 0.60,
    loss_feature_weight: float = 2.0,
    lp_time: float = 1800.0,
    strategic_time: float = 1200.0,
    evaluation_time: float = 60.0,
    mip_gap: float = 0.10,
    selection_only: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run selection and optionally one strategic solve/full evaluation."""
    started = time.time()
    relaxed_losses, relaxation_summary = _solve_relaxation(
        instance, distance_buckets, lp_time, verbose
    )
    selection = select_tail_aware_scenarios(
        instance,
        relaxed_losses,
        representative_count=representative_count,
        tail_buffer_factor=tail_buffer_factor,
        tail_representative_share=tail_representative_share,
        loss_feature_weight=loss_feature_weight,
    )
    clusters = selection["clusters"]
    print(
        "[screen 2] selected representatives: "
        f"tail={selection['tail_representatives']} "
        f"body={selection['body_representatives']}",
        flush=True,
    )

    output: Dict[str, Any] = {
        "scenario_count": len(instance["scenarios"]),
        "beta": float(instance["beta"]),
        "distance_buckets": int(distance_buckets),
        "relaxation": relaxation_summary,
        "selection": {
            key: selection[key]
            for key in (
                "representatives", "weights", "clusters", "tail_candidates",
                "tail_representatives", "body_representatives",
                "buffered_tail_probability",
            )
        },
    }
    if selection_only:
        output["elapsed_seconds"] = time.time() - started
        return output

    print(
        f"[strategic] one IP solve with {len(clusters)} representatives",
        flush=True,
    )
    p_values, b_values, strategic_summary = _solve_strategic_ip(
        instance, clusters, distance_buckets, strategic_time, mip_gap, verbose
    )
    losses, scenario_summaries = _evaluate_strategy(
        instance,
        p_values,
        b_values,
        distance_buckets,
        evaluation_time,
        mip_gap,
        verbose,
    )
    cvar, eta, actual_tail = _weighted_cvar(
        losses, instance["probability"], float(instance["beta"])
    )
    representative_set = set(clusters)
    relaxed_tail_set = set(selection["tail_candidates"])
    output["strategic"] = strategic_summary
    output["full_scenario_evaluation"] = {
        "cvar": cvar,
        "eta": eta,
        "actual_tail": actual_tail,
        "actual_tail_not_selected_as_representatives": [
            w for w in actual_tail if w not in representative_set
        ],
        "actual_tail_outside_relaxed_buffered_tail": [
            w for w in actual_tail if w not in relaxed_tail_set
        ],
        "scenario_evaluations": scenario_summaries,
    }
    output["elapsed_seconds"] = time.time() - started
    return output


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--representatives", type=int, default=20)
    parser.add_argument("--distance-buckets", type=int, default=16)
    parser.add_argument("--tail-buffer-factor", type=float, default=1.25)
    parser.add_argument("--tail-representative-share", type=float, default=0.60)
    parser.add_argument("--loss-feature-weight", type=float, default=2.0)
    parser.add_argument("--lp-time", type=float, default=1800.0)
    parser.add_argument("--strategic-time", type=float, default=1200.0)
    parser.add_argument("--evaluation-time", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir", default=os.path.join("output", "vif_tail_selection")
    )
    args = parser.parse_args()

    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if not 1 <= args.representatives <= args.scenarios:
        parser.error("--representatives must lie between 1 and --scenarios")
    if args.distance_buckets <= 0:
        parser.error("--distance-buckets must be positive")
    if args.tail_buffer_factor < 1.0:
        parser.error("--tail-buffer-factor must be at least one")
    if not 0.0 <= args.tail_representative_share <= 1.0:
        parser.error("--tail-representative-share must lie in [0,1]")

    params = copy.deepcopy(load_parameters())
    seed = args.seed if args.seed is not None else int(params["seed"])
    locations = load_locations()
    graph = build_graph(locations)
    scenarios = generate_scenarios(
        graph, locations, num_scenarios=args.scenarios, seed=seed
    )
    instance = build_stochastic_instance(
        locations=locations,
        scenarios=scenarios,
        params=params,
        alpha=args.alpha,
    )
    result = run_tail_selection(
        instance,
        representative_count=args.representatives,
        distance_buckets=args.distance_buckets,
        tail_buffer_factor=args.tail_buffer_factor,
        tail_representative_share=args.tail_representative_share,
        loss_feature_weight=args.loss_feature_weight,
        lp_time=args.lp_time,
        strategic_time=args.strategic_time,
        evaluation_time=args.evaluation_time,
        mip_gap=args.mip_gap,
        selection_only=args.selection_only,
        verbose=not args.quiet,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"tail_selection_{stamp}.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(_json_ready(result), stream, indent=2, sort_keys=True)
    print(f"tail-aware scenario selection -> {output_path}")
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
