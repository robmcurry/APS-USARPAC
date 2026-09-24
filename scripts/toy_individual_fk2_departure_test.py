"""
toy_individual_fk2_departure_test.py

ONE-OFF diagnostic. Tests whether constraint 5 (DepartureSingleNode /
DepartureNodeLink) is actually load-bearing against a genuine double-booking
of a single individual vehicle instance -- i.e. the same instance l asserted
on two DISCONNECTED flow segments in the same scenario, which would be
physically impossible and would silently inflate the aggregate n (and thus
VehicleCapFlow (18) capacity) beyond what F_k physically justifies.

Confirmed separately (see conversation) that constraint 5 carries no p_j or
home-node information -- that linkage lives entirely in
VehicleConservationIndiv (constraint 3). So removing constraint 5 cannot
silently break site-selection enforcement.

IMPORTANT terminology note (corrected after the first pass of this script):
"more than one distinct origin node used by the same (w,k,l)" is NOT by
itself evidence of double-booking -- it's also exactly what a legitimate
chained multi-leg path looks like (home -> 2 -> 3: arrival at 2 enables
departure from 2, which VehicleConservationIndiv explicitly allows). The
real test is PATH CONNECTIVITY: do all of (w,k,l)'s selected arcs chain
into a single walk starting at its home node? If yes, it's a legitimate
(if reconstructed/ambiguous-ordering, same caveat as the aggregate model's
itinerary reconstruction) multi-leg trip. If any arc is NOT reachable from
home by following the chain, that arc is a phantom/disconnected segment --
genuine double-booking.

Method: solve the toy instance at F_k=2 (air_fleet_size=2) twice:
  (a) vehicle_formulation="individual", constraint 5 ACTIVE (normal)
  (b) vehicle_formulation="individual", constraint 5 SKIPPED via the
      _debug_skip_departure_single_node diagnostic flag
and run the path-connectivity check on both.

Run: python scripts/toy_individual_fk2_departure_test.py   (from aps_usarpac/)
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from model.model import solve_stochastic_cvar, _assign_individual_homes
from toy_vehicle_test import build_toy_instance

TIME_LIMIT = 120
AIR_INDIV_TYPES = ("C-17", "C-130J")


def analyze_paths(results, instance, label):
    """For each (w, k, l) with movement, verify all its selected arcs chain
    into a single walk starting at its home node. Any arc left unexplained
    after that walk is a phantom/disconnected segment -- genuine double-
    booking, as opposed to a legitimate (if order-ambiguous) multi-leg chain."""
    flows = results.get("vehicle_flows_individual", {})
    arcs_by_vkl = defaultdict(list)
    for (w, k, l, m, i, j) in flows:
        arcs_by_vkl[(w, k, l)].append((i, j))

    home_by_k = {
        k: _assign_individual_homes(
            instance["vehicle_types"][k]["b_kj"], instance["vehicle_types"][k]["fleet_size"]
        )
        for k in AIR_INDIV_TYPES
    }

    print(f"\n--- Path/connectivity check: {label} ---")
    print(f"  Individual vehicle-arcs used: {len(flows)}")
    print(f"  Distinct (w,k,l) with any movement: {len(arcs_by_vkl)}")

    chain_count = 0
    phantom_count = 0
    for (w, k, l), arcs in sorted(arcs_by_vkl.items()):
        home = home_by_k[k][l]
        remaining = list(arcs)
        used = [False] * len(remaining)
        path = [home]
        current = home
        while True:
            nxt_idx = None
            for idx, (i, j) in enumerate(remaining):
                if not used[idx] and i == current:
                    nxt_idx = idx
                    break
            if nxt_idx is None:
                break
            used[nxt_idx] = True
            current = remaining[nxt_idx][1]
            path.append(current)

        unexplained = [arcs[idx] for idx in range(len(arcs)) if not used[idx]]
        if not unexplained:
            chain_count += 1
            tag = "multi-leg chain" if len(arcs) > 1 else "single-leg"
            print(f"    w={w}, k={k}, l={l}: {tag}, home={home}, path={path}")
        else:
            phantom_count += 1
            print(
                f"    w={w}, k={k}, l={l}: ** PHANTOM/DISCONNECTED (genuine "
                f"double-booking) ** home={home}, reachable_path={path}, "
                f"UNEXPLAINED arcs={unexplained}"
            )

    print(f"  Legitimate (single path from home, incl. single-leg): {chain_count}")
    print(f"  PHANTOM/disconnected (real double-booking): {phantom_count}")
    return phantom_count


def main():
    params = load_parameters()

    print("=" * 60)
    print("F_k=2 DEPARTURE-CONSTRAINT DIAGNOSTIC")
    print("=" * 60)

    # (a) normal: constraint 5 active. NOTE: as of the default flip
    # (_debug_skip_departure_single_node now defaults to True, i.e. skipped),
    # this must be passed explicitly to get constraint-5-active behavior.
    # The lazy subtour-elimination callback is now unconditionally attached
    # whenever vehicle_formulation="individual" (see model.py), so this case
    # actually exercises constraint 5 AND the callback together -- redundant
    # but not incorrect; the callback simply never finds anything to cut
    # here since constraint 5 already prevents any subtour from forming.
    instance_a = build_toy_instance(air_fleet_size=2)
    results_a = solve_stochastic_cvar(
        instance_a, time_limit=TIME_LIMIT, mip_gap=params["mip_gap"], verbose=False,
        vehicle_formulation="individual",
        _debug_skip_departure_single_node=False,
    )
    model_a = results_a["model"]
    print(f"\n(a) constraint 5 ACTIVE: status={results_a['status']}, "
          f"obj={results_a['objective_value']}, "
          f"NumVars={model_a.NumVars}, NumConstrs={model_a.NumConstrs}")
    phantom_a = analyze_paths(results_a, instance_a, "(a) constraint 5 ACTIVE")
    model_a.dispose()

    # (b) constraint 5 skipped
    instance_b = build_toy_instance(air_fleet_size=2)
    results_b = solve_stochastic_cvar(
        instance_b, time_limit=TIME_LIMIT, mip_gap=params["mip_gap"], verbose=False,
        vehicle_formulation="individual",
        _debug_skip_departure_single_node=True,
    )
    model_b = results_b["model"]
    print(f"\n(b) constraint 5 SKIPPED: status={results_b['status']}, "
          f"obj={results_b['objective_value']}, "
          f"NumVars={model_b.NumVars}, NumConstrs={model_b.NumConstrs}")
    phantom_b = analyze_paths(results_b, instance_b, "(b) constraint 5 SKIPPED")
    model_b.dispose()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Objective (a, constraint 5 active):   {results_a['objective_value']}")
    print(f"Objective (b, constraint 5 skipped):  {results_b['objective_value']}")
    print(f"Phantom/disconnected double-booking with constraint 5 active:  {phantom_a} instance(s)")
    print(f"Phantom/disconnected double-booking with constraint 5 skipped: {phantom_b} instance(s)")


if __name__ == "__main__":
    main()
