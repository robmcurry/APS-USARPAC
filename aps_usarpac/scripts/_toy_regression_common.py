"""
_toy_regression_common.py

Shared helpers for capturing/checking the aggregate-vehicle-formulation
regression baseline on the toy_vehicle_test instance (4 nodes, 5 scenarios,
beta=0.9). Used by capture_baseline_aggregate_toy.py and
check_aggregate_regression.py.

NOTE: solve_stochastic_cvar() has no vehicle_formulation parameter as of
this writing -- "aggregate" (n^omega_{k,m,ij} vehicle-count variables) is
the only formulation model.py implements, so no such kwarg is passed here.
That parameter is expected to be introduced by the individual-vehicle-
indexing branch this baseline exists to protect against regressing.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.model import solve_stochastic_cvar
from analysis.problem_size_certificate import CONSTRAINT_FAMILIES, classify_constraint
from toy_vehicle_test import build_toy_instance

TIME_LIMIT = 120


def run_toy_solve():
    """Solve the toy instance once and extract the exact-match regression fields."""
    instance = build_toy_instance()
    params = load_parameters()
    results = solve_stochastic_cvar(
        instance, time_limit=TIME_LIMIT, mip_gap=params["mip_gap"], verbose=False,
    )
    model = results["model"]
    solve_info = {
        "objective_value": results["objective_value"],
        "selected_sites": sorted(results["selected_sites"]),
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "status_code": results["status_code"],
        "status": results["status"],
        "solve_time_sec": model.Runtime,
    }
    model.dispose()
    return solve_info


def run_toy_problem_size():
    """Build (never solve) the toy instance to get ground-truth var/constraint counts."""
    instance = build_toy_instance()
    build = solve_stochastic_cvar(instance, build_only=True, verbose=False)
    model = build["model"]
    v = build["variables"]

    var_counts = {
        "p": len(v["p"]),
        "x": len(v["x"]),
        "z": len(v["z"]),
        "release": len(v["release"]),
        "tau": len(v["tau"]),
        "n": len(v["n"]),
        "eta": 1,
        "xi": len(v["xi"]),
        "loss": len(v["loss"]),
        "g_turn": len(v["g"]),
    }

    model.update()
    constr_counts = Counter()
    for c in model.getConstrs():
        constr_counts[classify_constraint(c.ConstrName)] += 1

    result = {
        "variables": var_counts,
        "variables_total": sum(var_counts.values()),
        "num_vars_gurobi": model.NumVars,
        "constraints": {fam: constr_counts.get(fam, 0) for fam in CONSTRAINT_FAMILIES},
        "constraints_total": sum(constr_counts.get(fam, 0) for fam in CONSTRAINT_FAMILIES),
        "num_constrs_gurobi": model.NumConstrs,
    }
    model.dispose()
    return result
