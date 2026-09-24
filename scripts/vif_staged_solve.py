"""Staged solution workflow for the full PRS-VIF formulation.

The workflow is intentionally heuristic:

1. Solve a reduced-scenario type/distance-state model to choose sites and
   vehicle base counts.
2. Fix those strategic decisions and solve one routing model per scenario.
3. Fix the combined scenario routes in the full model.
4. Solve the remaining continuous resource-allocation/CVaR problem as an LP.

Pass --reoptimize-routes to recover the optional joint final MIP, where the
combined routes are only a start and limited site/base changes are allowed.

Run from the aps_usarpac project root, for example:

    python scripts/vif_staged_solve.py \
        --strategic-scenarios 3 \
        --strategic-time 600 \
        --scenario-time 300 \
        --final-time 3600
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios


def _subset_instance(instance: Dict[str, Any], scenario_ids: Sequence[int]) -> Dict[str, Any]:
    """Return a shallow structural copy containing only scenario_ids."""
    selected = list(scenario_ids)
    selected_set = set(selected)
    original = list(instance["scenarios"])
    missing = selected_set - set(original)
    if missing:
        raise ValueError(f"unknown scenario ids: {sorted(missing)}")

    result = dict(instance)
    result["scenarios"] = selected

    probabilities = {
        w: float(instance["probability"][w]) for w in selected
    }
    total_probability = sum(probabilities.values())
    if total_probability <= 0.0:
        equal_probability = 1.0 / len(selected)
        probabilities = {w: equal_probability for w in selected}
    else:
        probabilities = {
            w: value / total_probability for w, value in probabilities.items()
        }
    result["probability"] = probabilities

    tuple_scenario_dicts = (
        "demand",
        "node_severity",
        "inventory_availability",
    )
    for name in tuple_scenario_dicts:
        values = instance.get(name)
        if isinstance(values, dict):
            result[name] = {
                key: value
                for key, value in values.items()
                if isinstance(key, tuple) and key and key[0] in selected_set
            }

    scalar_scenario_dicts = ("disaster_type",)
    for name in scalar_scenario_dicts:
        values = instance.get(name)
        if isinstance(values, dict):
            result[name] = {w: values[w] for w in selected}

    residual = instance.get("modal_residual")
    if isinstance(residual, list) and len(residual) == len(original):
        residual_by_scenario = dict(zip(original, residual))
        result["modal_residual"] = [residual_by_scenario[w] for w in selected]

    return result


def _selected_scenarios(instance: Dict[str, Any], count: int) -> List[int]:
    """Choose a deterministic probability-ranked strategic subset."""
    scenarios = list(instance["scenarios"])
    if count <= 0 or count > len(scenarios):
        raise ValueError(
            f"strategic scenario count must be in [1, {len(scenarios)}]"
        )
    return sorted(
        scenarios,
        key=lambda w: (-float(instance["probability"].get(w, 0.0)), w),
    )[:count]


def _require_solution(results: Dict[str, Any], stage: str) -> None:
    model = results["model"]
    if model.SolCount <= 0:
        status = results.get("status", model.Status)
        raise RuntimeError(f"{stage} produced no feasible solution (status={status})")


def _binary_values(family: Dict) -> Dict:
    return {key: int(var.X > 0.5) for key, var in family.items()}


def _selected_binary_values(family: Dict) -> Dict:
    return {key: 1 for key, var in family.items() if var.X > 0.5}


def _integer_values(family: Dict) -> Dict:
    return {key: int(round(var.X)) for key, var in family.items()}


def _selected_integer_values(family: Dict) -> Dict:
    return {
        key: int(round(var.X))
        for key, var in family.items()
        if var.X > 0.5
    }


def _dispose(results: Dict[str, Any]) -> None:
    model = results.get("model")
    if model is not None:
        model.dispose()


def _solver_params(time_limit: float, mip_gap: float) -> Dict[str, Any]:
    return {
        "TimeLimit": float(time_limit),
        "MIPGap": float(mip_gap),
        "CutPasses": 1,
        "DegenMoves": 0,
    }


def solve_staged_vif(
    instance: Dict[str, Any],
    strategic_scenario_count: int = 3,
    strategic_time: float = 600.0,
    scenario_time: float = 300.0,
    final_time: float = 3600.0,
    mip_gap: float = 0.05,
    site_neighborhood: int = 2,
    base_neighborhood: int = 2,
    fix_routes_final: bool = True,
    vehicle_formulation: str = "distance_state",
    distance_step_km: float = None,
    distance_buckets: int = 16,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute all four stages and return (final_results, summary)."""
    started = time.time()
    strategic_scenarios = _selected_scenarios(
        instance, strategic_scenario_count
    )

    print(
        "[stage 1/3] reduced strategic solve: "
        f"scenarios={strategic_scenarios}",
        flush=True,
    )
    strategic_instance = _subset_instance(instance, strategic_scenarios)
    strategic_results = solve_stochastic_cvar(
        strategic_instance,
        vehicle_formulation=vehicle_formulation,
        verbose=verbose,
        vif_solve_config={
            "distance_step_km": distance_step_km,
            "distance_buckets": distance_buckets,
            "params": _solver_params(strategic_time, mip_gap),
        },
    )
    _require_solution(strategic_results, "strategic solve")
    strategic_variables = strategic_results["variables"]
    strategic_p = _binary_values(strategic_variables["p"])
    strategic_b = _integer_values(strategic_variables["b"])
    strategic_objective = strategic_results["objective_value"]
    strategic_gap = strategic_results["model"].MIPGap
    _dispose(strategic_results)

    fixed_strategy = {"p": strategic_p, "b": strategic_b}
    routing_n_start: Dict[Tuple, int] = {}
    scenario_summaries = []

    print(
        "[stage 2/3] fixed-strategy scenario routing solves", flush=True
    )
    for position, scenario_id in enumerate(instance["scenarios"], start=1):
        print(
            f"  scenario {scenario_id} ({position}/{len(instance['scenarios'])})",
            flush=True,
        )
        scenario_instance = _subset_instance(instance, [scenario_id])
        scenario_results = solve_stochastic_cvar(
            scenario_instance,
            vehicle_formulation=vehicle_formulation,
            verbose=verbose,
            vif_solve_config={
                "fix": fixed_strategy,
                "distance_step_km": distance_step_km,
                "distance_buckets": distance_buckets,
                "params": _solver_params(scenario_time, mip_gap),
            },
        )
        _require_solution(scenario_results, f"scenario {scenario_id}")
        routing_n_start.update(
            _selected_integer_values(scenario_results["variables"]["n"])
        )
        scenario_summaries.append(
            {
                "scenario": scenario_id,
                "status": scenario_results["status"],
                "objective": scenario_results["objective_value"],
                "gap": scenario_results["model"].MIPGap,
                "selected_vehicle_arcs": len(
                    scenario_results["vehicle_arcs"]
                ),
            }
        )
        _dispose(scenario_results)

    if fix_routes_final:
        print(
            "[stage 3/3] full resource-allocation LP with sites, bases, "
            "and routes fixed",
            flush=True,
        )
        final_config = {
            "distance_step_km": distance_step_km,
            "distance_buckets": distance_buckets,
            "fix_defaults": {"n": 0.0},
            "fix": {
                "p": strategic_p,
                "b": strategic_b,
                "n": routing_n_start,
            },
            "relax_fixed_families": ["p", "b", "n"],
            "params": _solver_params(final_time, mip_gap),
        }
    else:
        print(
            "[stage 3/3] full joint MIP with combined route start and "
            "strategic neighborhood",
            flush=True,
        )
        final_config = {
            "distance_step_km": distance_step_km,
            "distance_buckets": distance_buckets,
            "start_defaults": {"p": 0.0, "b": 0.0, "n": 0.0},
            "start": {
                "p": strategic_p,
                "b": strategic_b,
                "n": routing_n_start,
            },
            "strategic_p": strategic_p,
            "strategic_b": strategic_b,
            "p_neighborhood": int(site_neighborhood),
            "b_neighborhood": int(base_neighborhood),
            "params": _solver_params(final_time, mip_gap),
        }

    final_results = solve_stochastic_cvar(
        instance,
        vehicle_formulation=vehicle_formulation,
        verbose=verbose,
        vif_solve_config=final_config,
    )

    final_model = final_results["model"]
    summary = {
        "elapsed_seconds": time.time() - started,
        "vehicle_formulation": vehicle_formulation,
        "distance_step_km": distance_step_km,
        "distance_buckets": distance_buckets,
        "strategic_scenarios": strategic_scenarios,
        "strategic_objective": strategic_objective,
        "strategic_gap": strategic_gap,
        "strategic_selected_sites": sorted(
            i for i, value in strategic_p.items() if value
        ),
        "strategic_basing": sorted(
            [k, j, value] for (k, j), value in strategic_b.items() if value
        ),
        "scenario_solves": scenario_summaries,
        "combined_route_start_arcs": len(routing_n_start),
        "final_mode": (
            "fixed_route_resource_allocation_lp"
            if fix_routes_final
            else "joint_neighborhood_mip"
        ),
        "site_neighborhood": (
            None if fix_routes_final else int(site_neighborhood)
        ),
        "base_neighborhood": (
            None if fix_routes_final else int(base_neighborhood)
        ),
        "final_status": final_results["status"],
        "final_solution_count": final_model.SolCount,
        "final_objective": final_results["objective_value"],
        "final_bound": final_model.ObjBound,
        "final_gap": (
            final_model.MIPGap
            if final_model.SolCount and final_model.IsMIP
            else 0.0 if final_model.SolCount else None
        ),
        "final_nodes": final_model.NodeCount,
        "final_selected_sites": final_results["selected_sites"],
        "callback_stats": final_results["subtour_callback_stats"],
        "distance_state": final_results.get("distance_state"),
    }
    return final_results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategic-scenarios", type=int, default=3)
    parser.add_argument("--strategic-time", type=float, default=600.0)
    parser.add_argument("--scenario-time", type=float, default=300.0)
    parser.add_argument("--final-time", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument(
        "--distance-step-km",
        type=float,
        default=None,
        help=(
            "Optional common distance-state bucket width. Overrides "
            "--distance-buckets. Arc consumption is rounded upward."
        ),
    )
    parser.add_argument(
        "--distance-buckets",
        type=int,
        default=16,
        help=(
            "Number of cumulative-distance buckets per vehicle type when "
            "--distance-step-km is omitted (default: 16)."
        ),
    )
    parser.add_argument("--site-neighborhood", type=int, default=2)
    parser.add_argument("--base-neighborhood", type=int, default=2)
    parser.add_argument(
        "--reoptimize-routes",
        action="store_true",
        help=(
            "Use the scenario routes only as a MIP start and jointly "
            "reoptimize routes/resources in the final stage. By default, "
            "routes are fixed and the final stage is a resource-allocation LP."
        ),
    )
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir", default=os.path.join("output", "vif_staged")
    )
    args = parser.parse_args()

    if args.site_neighborhood < 0 or args.base_neighborhood < 0:
        parser.error("neighborhood radii must be nonnegative")
    if args.distance_step_km is not None and args.distance_step_km <= 0:
        parser.error("--distance-step-km must be positive")
    if args.distance_buckets <= 0:
        parser.error("--distance-buckets must be positive")

    params = copy.deepcopy(load_parameters())
    scenario_count = args.scenarios or int(params["num_scenarios"])
    seed = args.seed if args.seed is not None else int(params["seed"])

    locations = load_locations()
    graph = build_graph(locations)
    scenarios = generate_scenarios(
        graph,
        locations,
        num_scenarios=scenario_count,
        seed=seed,
    )
    instance = build_stochastic_instance(
        locations=locations,
        scenarios=scenarios,
        params=params,
        alpha=args.alpha,
    )

    final_results, summary = solve_staged_vif(
        instance,
        strategic_scenario_count=args.strategic_scenarios,
        strategic_time=args.strategic_time,
        scenario_time=args.scenario_time,
        final_time=args.final_time,
        mip_gap=args.mip_gap,
        site_neighborhood=args.site_neighborhood,
        base_neighborhood=args.base_neighborhood,
        fix_routes_final=not args.reoptimize_routes,
        vehicle_formulation="distance_state",
        distance_step_km=args.distance_step_km,
        distance_buckets=args.distance_buckets,
        verbose=not args.quiet,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"summary_{stamp}.json")
    solution_path = os.path.join(args.output_dir, f"solution_{stamp}.sol")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    if final_results["model"].SolCount:
        final_results["model"].write(solution_path)

    print(f"staged summary -> {summary_path}")
    if final_results["model"].SolCount:
        print(f"final solution -> {solution_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    _dispose(final_results)


if __name__ == "__main__":
    main()
