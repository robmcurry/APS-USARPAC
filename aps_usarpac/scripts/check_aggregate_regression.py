"""
check_aggregate_regression.py

Reruns the toy_vehicle_test instance (aggregate vehicle formulation) and
diffs every field against the locked output/baseline_aggregate_toy.json.
Run this before/after any model.py change -- especially the individual-
vehicle-indexing work -- to confirm the aggregate formulation's behavior
hasn't drifted.

Exact match required for: objective_value, selected_sites, status,
status_code, mip_gap, and every variable/constraint family count (all are
deterministic outputs of a fixed instance + fixed solver params).
Tolerance: solve_time_sec is reported but never gates PASS/FAIL -- wall
-clock solve time legitimately varies run to run.

Run: python scripts/check_aggregate_regression.py   (from aps_usarpac/)
Exit code: 0 if every gating field PASSes, 1 otherwise.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._toy_regression_common import run_toy_solve, run_toy_problem_size

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "baseline_aggregate_toy.json",
)


def _check(label, expected, actual, gating=True):
    ok = expected == actual
    if gating:
        tag = "PASS" if ok else "FAIL"
    else:
        tag = "INFO"
    print(f"  [{tag}] {label}: baseline={expected!r}  current={actual!r}")
    return ok if gating else True


def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"No baseline found at {BASELINE_PATH}.")
        print("Run scripts/capture_baseline_aggregate_toy.py first (deliberately).")
        return 1

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    solve_info = run_toy_solve()
    problem_size = run_toy_problem_size()

    all_ok = True

    print("=== Solve fields ===")
    b_solve = baseline["solve"]
    all_ok &= _check("objective_value", b_solve["objective_value"], solve_info["objective_value"])
    all_ok &= _check("selected_sites", b_solve["selected_sites"], solve_info["selected_sites"])
    all_ok &= _check("status", b_solve["status"], solve_info["status"])
    all_ok &= _check("status_code", b_solve["status_code"], solve_info["status_code"])
    all_ok &= _check("mip_gap", b_solve["mip_gap"], solve_info["mip_gap"])
    _check(
        "solve_time_sec (informational only, not gating)",
        b_solve["solve_time_sec"], solve_info["solve_time_sec"], gating=False,
    )

    print("\n=== Problem size: variable families ===")
    b_vars = baseline["problem_size"]["variables"]
    for fam, expected in b_vars.items():
        all_ok &= _check(f"variables[{fam}]", expected, problem_size["variables"].get(fam))
    all_ok &= _check(
        "variables_total",
        baseline["problem_size"]["variables_total"], problem_size["variables_total"],
    )
    all_ok &= _check(
        "num_vars_gurobi",
        baseline["problem_size"]["num_vars_gurobi"], problem_size["num_vars_gurobi"],
    )

    print("\n=== Problem size: constraint families ===")
    b_constrs = baseline["problem_size"]["constraints"]
    for fam, expected in b_constrs.items():
        all_ok &= _check(f"constraints[{fam}]", expected, problem_size["constraints"].get(fam))
    all_ok &= _check(
        "constraints_total",
        baseline["problem_size"]["constraints_total"], problem_size["constraints_total"],
    )
    all_ok &= _check(
        "num_constrs_gurobi",
        baseline["problem_size"]["num_constrs_gurobi"], problem_size["num_constrs_gurobi"],
    )

    print(f"\n{'=' * 60}")
    print("ALL PASS" if all_ok else "REGRESSION DETECTED -- see FAIL lines above")
    print(f"{'=' * 60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
