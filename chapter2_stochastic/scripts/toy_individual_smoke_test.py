"""
toy_individual_smoke_test.py

ONE-OFF smoke test for the "individual" vehicle_formulation (air-mode-only
per-vehicle-instance indexing), at the smallest possible scale: F_k=1 for
both C-17 and C-130J, on the existing 4-node/5-scenario toy network. Confirms
the new formulation produces a correct, solvable instance before any
benchmark scaling begins.

This is NOT a locked regression check -- that remains
scripts/check_aggregate_regression.py, which stays aggregate-only and is
unaffected by this script.

Run: python scripts/toy_individual_smoke_test.py   (from aps_usarpac/)
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.model import solve_stochastic_cvar
from analysis.problem_size_certificate import CONSTRAINT_FAMILIES
from toy_vehicle_test import build_toy_instance

# Constraint-name prefixes introduced by the "individual" formulation
# (air-only), on top of analysis.problem_size_certificate.CONSTRAINT_FAMILIES.
INDIVIDUAL_FAMILIES = [
    "VehicleAggregation",
    "VehicleConservationIndiv",
    "TurnExemptIndivUB1",
    "TurnExemptIndivUB2",
    "TurnExemptIndivLB",
    "DistanceBudgetIndiv",
    "VehicleSymmetryBreak",
    "DepartureNodeLink",
    "DepartureSingleNode",
]


def classify(name: str) -> str:
    for fam in CONSTRAINT_FAMILIES + INDIVIDUAL_FAMILIES:
        if name == fam or name.startswith(fam + "_"):
            return fam
    return "UNCLASSIFIED"


def main():
    instance = build_toy_instance(air_fleet_size=1)
    params = load_parameters()

    print("=" * 60)
    print("TOY INDIVIDUAL-VEHICLE SMOKE TEST (air-mode-only, F_k=1)")
    print("=" * 60)

    results = solve_stochastic_cvar(
        instance, time_limit=120, mip_gap=params["mip_gap"], verbose=True,
        vehicle_formulation="individual",
    )
    model = results["model"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Objective: {results['objective_value']}")
    print(f"Selected sites: {sorted(results['selected_sites'])}")
    print(f"MIP gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Solve time (s): {model.Runtime}")
    print(f"NumVars (Gurobi ground truth): {model.NumVars}")
    print(f"NumConstrs (Gurobi ground truth): {model.NumConstrs}")

    var_families = results["variables"]
    print("\nVariable family counts:")
    for name, v in var_families.items():
        try:
            print(f"  {name}: {len(v)}")
        except TypeError:
            print(f"  {name}: 1 (scalar)")

    constr_counts = Counter()
    for c in model.getConstrs():
        constr_counts[classify(c.ConstrName)] += 1
    print("\nConstraint family counts:")
    for fam, cnt in sorted(constr_counts.items()):
        print(f"  {fam}: {cnt}")
    if constr_counts.get("UNCLASSIFIED", 0):
        print(f"  ** UNCLASSIFIED: {constr_counts['UNCLASSIFIED']} (unexpected name pattern) **")

    print("\nIndividual vehicle movements (n_ind > 0.5):")
    flows = results.get("vehicle_flows_individual", {})
    if flows:
        for (w, k, l, m, i, j), count in sorted(flows.items()):
            print(f"  w={w}, k={k}, l={l} ({m}): {i}->{j}")
    else:
        print("  (none)")

    model.dispose()
    return 0 if results["objective_value"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
