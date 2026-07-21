"""
toy_individual_cyclic_test.py

ONE-OFF diagnostic. The DAG-structured toy air network (toy_individual_
fk2_departure_test.py) cannot distinguish "conservation constraint 3 alone
prevents double-booking" from "this network has no cycle to exploit" --
every air arc points strictly away from the home node, so no cycle among
non-home nodes was ever reachable to test. This script adds a genuine cycle
and reruns.

Return arc chosen: (3, 2), the reverse of the existing (2, 3) air arc --
NOT (2, 1), the reverse of (1, 2). Reasoning: node 1 is home for every
individually-indexed vehicle (C-17 and C-130J both only ever base at node
1, the sole PPL-1 node). A cycle that touches home (like 1<->2) is trivially
reachable from home and doesn't stress anything new. The dangerous case --
the one this script is built to catch -- is a cycle ENTIRELY AMONG NON-HOME
NODES (here, 2<->3), where a vehicle instance could in principle show
positive flow on both cycle arcs without ever consuming its one unit of
home credit, i.e. a locally-balanced but structurally disconnected segment.
That's the same category of bug as the earlier Sydney-Adelaide circular
flow: a self-sustaining loop that inflates the aggregate n (and thus
VehicleCapFlow (18) capacity) without being traceable to any real vehicle.

This script does two things, not just one:
  1. (as literally requested) Solve the cyclic toy instance at F_k=2 with
     constraint 5 (DepartureSingleNode/DepartureNodeLink) REMOVED, and print
     the FULL path reconstruction for every active (w,k,l) instance -- not
     just a leg count -- so a disconnected-but-locally-balanced segment
     would be directly visible if the optimizer ever selects one.
  2. (additional, beyond what was literally asked -- flagged as such) A
     direct FEASIBILITY PROBE: force one vehicle instance onto exactly the
     phantom pair {(2,3), (3,2)} with zero home-departing arcs, and check
     whether the resulting model is feasible under conservation (3) ALONE
     (constraint 5 removed). This answers the structural question
     definitively, independent of whether the toy demand pattern happens to
     make the phantom pair economically attractive in an unconstrained
     solve -- a solver that never chooses to exploit a hole in this specific
     toy instance is not proof the hole doesn't exist.

Run: python scripts/toy_individual_cyclic_test.py   (from aps_usarpac/)
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
NEW_ARC = (3, 2)  # return arc for existing (2, 3); cycle among non-home nodes


def build_cyclic_toy_instance(air_fleet_size=2):
    """build_toy_instance() plus one return air arc, (3, 2), completing a
    2-3 cycle. All derived per-arc structures (distance, cost, capacity,
    per-scenario residual) are mirrored from the existing forward arc
    (2, 3) so the new arc is fully well-formed, not just present in the
    arc list. toy_vehicle_test.build_toy_instance itself is untouched --
    this is a local copy, built fresh each call, so the locked regression
    baseline is unaffected."""
    instance = build_toy_instance(air_fleet_size=air_fleet_size)

    instance["modal_arcs"]["air"] = list(instance["modal_arcs"]["air"]) + [NEW_ARC]
    instance["modal_arcs"]["all"] = sorted(set(instance["modal_arcs"]["all"]) | {NEW_ARC})

    fwd_dist = instance["modal_arc_distance"]["air"][(2, 3)]
    instance["modal_arc_distance"]["air"][NEW_ARC] = fwd_dist
    instance["modal_arc_cost"]["air"][NEW_ARC] = fwd_dist * 0.005

    instance["modal_capacity"]["air"][NEW_ARC] = dict(instance["modal_capacity"]["air"][(2, 3)])

    for w_residual in instance["modal_residual"]:
        w_residual["air"][NEW_ARC] = dict(w_residual["air"][(2, 3)])

    return instance


def reconstruct_path(arcs, home):
    """Greedy walk from home consuming available arcs. Returns
    (path, unexplained_arcs) -- unexplained_arcs is non-empty iff some
    selected arc(s) are not reachable from home, i.e. a phantom/disconnected
    segment."""
    used = [False] * len(arcs)
    path = [home]
    current = home
    while True:
        nxt_idx = None
        for idx, (i, j) in enumerate(arcs):
            if not used[idx] and i == current:
                nxt_idx = idx
                break
        if nxt_idx is None:
            break
        used[nxt_idx] = True
        current = arcs[nxt_idx][1]
        path.append(current)
    unexplained = [arcs[idx] for idx in range(len(arcs)) if not used[idx]]
    return path, unexplained


def full_path_report(results, instance, label):
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

    print(f"\n--- Full path reconstruction: {label} ---")
    print(f"  Individual vehicle-arcs used: {len(flows)}")
    print(f"  Distinct (w,k,l) with any movement: {len(arcs_by_vkl)}")

    phantom_count = 0
    for (w, k, l), arcs in sorted(arcs_by_vkl.items()):
        home = home_by_k[k][l]
        path, unexplained = reconstruct_path(arcs, home)
        if unexplained:
            phantom_count += 1
            print(
                f"    w={w}, k={k}, l={l}: home={home}, arcs={sorted(arcs)} "
                f"-> reachable_path={path}, ** UNEXPLAINED (PHANTOM) arcs={unexplained} **"
            )
        else:
            print(f"    w={w}, k={k}, l={l}: home={home}, arcs={sorted(arcs)} -> path={path}")

    print(f"  PHANTOM/disconnected instances: {phantom_count}")
    return phantom_count


def feasibility_probe(instance):
    """Direct structural test, independent of solver incentives: force one
    vehicle instance onto exactly the phantom pair {(2,3),(3,2)} with zero
    home-departing arcs, constraint 5 removed, and check feasibility."""
    build = solve_stochastic_cvar(
        instance, build_only=True, verbose=False,
        vehicle_formulation="individual", _debug_skip_departure_single_node=True,
    )
    model = build["model"]
    n_ind = build["variables"]["n_ind"]

    w_probe = 0
    k_probe = "C-130J"
    l_probe = 1
    home_by_k = _assign_individual_homes(
        instance["vehicle_types"][k_probe]["b_kj"], instance["vehicle_types"][k_probe]["fleet_size"]
    )
    home = home_by_k[l_probe]

    forced = 0
    for (w, k, l, m, i, j), var in n_ind.items():
        if w != w_probe or k != k_probe or l != l_probe:
            continue
        if (i, j) in ((2, 3), (3, 2)):
            model.addConstr(var == 1, name=f"PROBE_force_on_{i}_{j}")
        elif i == home:
            model.addConstr(var == 0, name=f"PROBE_force_off_home_{i}_{j}")
        forced += 1

    model.Params.OutputFlag = 0
    model.optimize()
    status = model.Status
    print(f"\n--- Feasibility probe: force n_ind[w={w_probe},k={k_probe},l={l_probe}] "
          f"onto phantom pair (2,3)+(3,2), zero home-departing arcs, constraint 5 removed ---")
    print(f"  Forced/constrained variables touched: {forced}")
    print(f"  Gurobi status: {status} ({'FEASIBLE' if status not in (3, 4) else 'INFEASIBLE/INF_OR_UNBD'})")
    model.dispose()
    return status


def main():
    params = load_parameters()

    print("=" * 60)
    print("CYCLIC TOY NETWORK -- F_k=2 DEPARTURE-CONSTRAINT TEST")
    print(f"Added return arc: {NEW_ARC} (reverse of existing (2, 3))")
    print("=" * 60)

    instance = build_cyclic_toy_instance(air_fleet_size=2)
    print(f"\nAir arcs now: {sorted(instance['modal_arcs']['air'])}")

    results = solve_stochastic_cvar(
        instance, time_limit=TIME_LIMIT, mip_gap=params["mip_gap"], verbose=False,
        vehicle_formulation="individual",
        _debug_skip_departure_single_node=True,
    )
    model = results["model"]
    print(f"\nSolve (constraint 5 REMOVED): status={results['status']}, "
          f"obj={results['objective_value']}, "
          f"NumVars={model.NumVars}, NumConstrs={model.NumConstrs}")
    phantom_count = full_path_report(results, instance, "constraint 5 REMOVED, cyclic network")
    model.dispose()

    print("\n" + "=" * 60)
    print("ADDITIONAL: DIRECT FEASIBILITY PROBE (beyond what was literally asked)")
    print("=" * 60)
    probe_instance = build_cyclic_toy_instance(air_fleet_size=2)
    probe_status = feasibility_probe(probe_instance)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Natural optimal solve: phantom/disconnected instances observed = {phantom_count}")
    print(f"Direct feasibility probe status code = {probe_status} "
          f"(2=OPTIMAL/feasible, 3=INFEASIBLE, 4=INF_OR_UNBD)")


if __name__ == "__main__":
    main()
