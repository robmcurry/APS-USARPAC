"""
capture_baseline_aggregate_toy.py

ONE-OFF capture script. (Re)generates output/baseline_aggregate_toy.json,
the locked regression reference for the aggregate vehicle formulation --
model.py on main, before any individual-vehicle-indexing changes -- from a
fresh toy solve (toy_vehicle_test.build_toy_instance) plus a build-only pass
for exact variable/constraint counts.

output/baseline_aggregate_toy.json should never be hand-edited, and running
this script to regenerate it is a deliberate decision, not a routine action
-- routine verification is check_aggregate_regression.py, which diffs
against whatever this script last wrote.

Run: python scripts/capture_baseline_aggregate_toy.py   (from aps_usarpac/)
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._toy_regression_common import run_toy_solve, run_toy_problem_size

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "baseline_aggregate_toy.json",
)


def main():
    solve_info = run_toy_solve()
    problem_size = run_toy_problem_size()

    baseline = {
        "generated": date.today().isoformat(),
        "note": (
            "Locked regression baseline for the aggregate vehicle formulation "
            "(model.py on main, captured before individual-vehicle-indexing "
            "work begins). solve_stochastic_cvar() has no vehicle_formulation "
            "parameter as of this capture -- 'aggregate' "
            "(n^omega_{k,m,ij} vehicle-count variables) is the only "
            "formulation implemented, so no such kwarg is passed. "
            "Instance: toy_vehicle_test.build_toy_instance() "
            "(4 nodes, 5 scenarios, beta=0.9, time_limit=120s)."
        ),
        "instance": "toy_vehicle_test.build_toy_instance",
        "beta": 0.9,
        "vehicle_formulation": "aggregate (implicit -- no kwarg exists yet on main)",
        "solve": solve_info,
        "problem_size": problem_size,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote baseline to {OUTPUT_PATH}")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
