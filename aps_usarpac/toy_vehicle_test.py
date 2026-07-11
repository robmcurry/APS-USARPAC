"""
toy_vehicle_test.py

Minimal 4-node network to verify vehicle-heterogeneity constraints (16-19)
before running against the full 50-node network.

Nodes:
  1: PPL-1, A=3, S=3, L=3 — strategic hub
  2: PPL-2, A=3, S=3, L=2 — regional node
  3: PPL-3, A=2, S=1, L=1 — contingency site
  4: non-PPL, demand-only node

Air arcs: 1->2, 1->3, 1->4, 2->3, 2->4 (5 directed arcs)
Sea arcs: 1->2 (1 directed arc)
Land arcs: 2->3, 3->2 (2 directed arcs)

Vehicle types (scaled-down fleet for toy):
  C-17:     air, fleet=4, A>=3 -> eligible at {1,2}
  C-130J:   air, fleet=4, A>=2 -> eligible at {1,2,3}
  LCU-1700: sea, fleet=2, S>=2 -> eligible at {1,2}
  M1083:    land, fleet=6, L>=1 -> eligible at {1,2,3}

5 scenarios with disaster centered on node 4 (varying severity).

Run: python toy_vehicle_test.py   (from aps_usarpac/)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import load_parameters
from model.model import solve_stochastic_cvar, print_solution_summary


def _compute_basing(eligible, fleet_size, node_tier):
    tier_weights = {"PPL-1": 3, "PPL-2": 2, "PPL-3": 1}
    total_w = sum(tier_weights.get(node_tier.get(j, ""), 0) for j in eligible)
    b = {}
    if total_w > 0 and fleet_size > 0:
        for j in eligible:
            w = tier_weights.get(node_tier.get(j, ""), 0)
            b[j] = int(fleet_size * w // total_w)
        remainder = fleet_size - sum(b.values())
        if remainder > 0:
            best = max(eligible, key=lambda j: (
                tier_weights.get(node_tier.get(j, ""), 0), -j,
            ))
            b[best] += remainder
    return b


def build_toy_instance():
    nodes = [1, 2, 3, 4]
    ppl_nodes = [1, 2, 3]
    commodities = ["food", "water"]
    scenarios = list(range(5))
    modes = ["sea", "air", "land"]

    air_arcs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    sea_arcs = [(1, 2)]
    land_arcs = [(2, 3), (3, 2)]
    modal_arcs = {
        "sea": sea_arcs,
        "air": air_arcs,
        "land": land_arcs,
        "all": sorted(set(air_arcs + sea_arcs + land_arcs)),
    }

    air_dist = {(1, 2): 500, (1, 3): 600, (1, 4): 1000, (2, 3): 400, (2, 4): 800}
    sea_dist = {(1, 2): 500}
    land_dist = {(2, 3): 400, (3, 2): 400}
    modal_arc_distance = {"sea": sea_dist, "air": air_dist, "land": land_dist}

    modal_arc_cost = {
        m: {arc: d * 0.005 for arc, d in dists.items()}
        for m, dists in modal_arc_distance.items()
    }

    base_cap = {"food": 500000, "water": 500000}
    modal_capacity = {
        "sea": {arc: dict(base_cap) for arc in sea_arcs},
        "air": {arc: dict(base_cap) for arc in air_arcs},
        "land": {arc: dict(base_cap) for arc in land_arcs},
    }

    modal_residual = []
    for w in scenarios:
        degradation = max(0.0, 1.0 - 0.1 * w)
        residual = {}
        for m in modes:
            residual[m] = {}
            for arc, cap in modal_capacity[m].items():
                residual[m][arc] = {r: v * degradation for r, v in cap.items()}
        modal_residual.append(residual)

    transfer_cap = {}
    transfer_cost = {
        ("sea", "air"): 15.0, ("sea", "land"): 2.5, ("air", "land"): 4.0,
        ("land", "sea"): 3.5, ("land", "air"): 6.0, ("air", "sea"): 10.0,
    }

    node_pop = {1: 1000000, 2: 500000, 3: 100000, 4: 2000000}
    demand = {}
    for w in scenarios:
        severity = {1: 0.0, 2: 0.5, 3: 0.5, 4: 1.0 + 0.5 * w}
        for i in nodes:
            for r in commodities:
                alpha_r = 0.15 if r == "food" else 0.20
                demand[(w, i, r)] = alpha_r * severity.get(i, 0.0) * node_pop[i]

    tier_cap = {"PPL-1": 1300000, "PPL-2": 600000, "PPL-3": 150000}
    node_tier = {1: "PPL-1", 2: "PPL-2", 3: "PPL-3", 4: None}
    inventory_if_open = {}
    for i in nodes:
        cap = tier_cap.get(node_tier.get(i), 0)
        for r in commodities:
            inventory_if_open[(i, r)] = float(cap)

    inventory_availability = {
        (w, i, r): 1.0 for w in scenarios for i in nodes for r in commodities
    }

    site_cost = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    selection_budget = 12.0
    penalty = {(i, r): 500.0 for i in nodes for r in commodities}
    probability = {w: 1.0 / len(scenarios) for w in scenarios}

    # C-17 and LCU-1700: basing restricted to PPL-1 only (node 1)
    # C-130J and M1083: full eligibility set
    c17_eligible = [j for j in [1, 2] if node_tier.get(j) == "PPL-1"]   # [1]
    lcu_eligible = [j for j in [1, 2] if node_tier.get(j) == "PPL-1"]   # [1]

    vehicle_types = {
        "C-17": {
            "mode": "air",
            "fleet_size": 4,
            "capacity": {"food": 143519, "water": 5167},
            "J_k": c17_eligible,
            "b_kj": _compute_basing(c17_eligible, 4, node_tier),
            "D_k": 3.0 * 11662,
            "pi_k": (2.0 / 24.0) * 11662,
            "cruise_speed_km_day": 11662,
        },
        "C-130J": {
            "mode": "air",
            "fleet_size": 4,
            "capacity": {"food": 35280, "water": 1270},
            "J_k": [1, 2, 3],
            "b_kj": _compute_basing([1, 2, 3], 4, node_tier),
            "D_k": 3.0 * 7840,
            "pi_k": (2.0 / 24.0) * 7840,
            "cruise_speed_km_day": 7840,
        },
        "LCU-1700": {
            "mode": "sea",
            "fleet_size": 2,
            "capacity": {"food": 285594, "water": 10281},
            "J_k": lcu_eligible,
            "b_kj": _compute_basing(lcu_eligible, 2, node_tier),
            "D_k": 3.0 * 408,
            "pi_k": (6.0 / 24.0) * 408,
            "cruise_speed_km_day": 408,
        },
        "M1083": {
            "mode": "land",
            "fleet_size": 6,
            "capacity": {"food": 8400, "water": 302},
            "J_k": [1, 2, 3],
            "b_kj": _compute_basing([1, 2, 3], 6, node_tier),
            "D_k": 3.0 * 1116,
            "pi_k": (1.0 / 24.0) * 1116,
            "cruise_speed_km_day": 1116,
        },
    }
    K_m = {
        "air": ["C-17", "C-130J"],
        "sea": ["LCU-1700"],
        "land": ["M1083"],
    }

    return {
        "nodes": nodes,
        "ppl_nodes": ppl_nodes,
        "commodities": commodities,
        "scenarios": scenarios,
        "modes": modes,
        "modal_arcs": modal_arcs,
        "modal_capacity": modal_capacity,
        "modal_arc_cost": modal_arc_cost,
        "modal_arc_distance": modal_arc_distance,
        "modal_residual": modal_residual,
        "transfer_cap": transfer_cap,
        "transfer_cost": transfer_cost,
        "probability": probability,
        "demand": demand,
        "inventory_if_open": inventory_if_open,
        "inventory_availability": inventory_availability,
        "safety_stock_fraction": 0.20,
        "site_cost": site_cost,
        "selection_budget": selection_budget,
        "penalty": penalty,
        "P_max": 2,
        "beta": 0.9,
        "vehicle_types": vehicle_types,
        "K_m": K_m,
        "modal_arc_distance": modal_arc_distance,
    }


def verify_vehicle_constraints(results, instance):
    """Check all four vehicle constraint families against the solution."""
    if results["objective_value"] is None:
        print("No solution to verify.")
        return False

    vehicle_types = instance["vehicle_types"]
    K_m = instance["K_m"]
    modes = instance["modes"]
    modal_arcs = instance["modal_arcs"]
    modal_arc_distance = instance["modal_arc_distance"]
    N = instance["nodes"]
    PPL_set = set(instance["ppl_nodes"])
    Omega = instance["scenarios"]
    R = instance["commodities"]

    n_var = results["variables"]["n"]
    x_var = results["variables"]["x"]
    p_var = results["variables"]["p"]

    modal_incoming = {m: {j: [] for j in N} for m in modes}
    modal_outgoing = {m: {i: [] for i in N} for m in modes}
    for m in modes:
        for i, j in modal_arcs[m]:
            modal_outgoing[m][i].append((i, j))
            modal_incoming[m][j].append((i, j))

    all_ok = True

    # --- (16) Vehicle Conservation ---
    print("\n--- Checking Vehicle Conservation (16) ---")
    violations_16 = 0
    for w in Omega:
        for m in modes:
            for k in K_m.get(m, []):
                b_kj = vehicle_types[k]["b_kj"]
                for j in N:
                    arrivals = sum(
                        n_var[w, k, m, i_src, j].X
                        for (i_src, _) in modal_incoming[m][j]
                        if (w, k, m, i_src, j) in n_var
                    )
                    departures = sum(
                        n_var[w, k, m, j, j_dst].X
                        for (_, j_dst) in modal_outgoing[m][j]
                        if (w, k, m, j, j_dst) in n_var
                    )
                    b_val = b_kj.get(j, 0)
                    p_val = p_var[j].X if j in PPL_set else 0
                    basing = b_val * p_val if b_val > 0 else 0
                    if departures > arrivals + basing + 1e-4:
                        print(
                            f"  VIOLATION: w={w}, k={k}, j={j}: "
                            f"dep={departures:.1f} > arr={arrivals:.1f} + base={basing:.1f}"
                        )
                        violations_16 += 1
    print(f"  Conservation violations: {violations_16}")
    if violations_16 > 0:
        all_ok = False

    # --- (17) Fleet Size — REMOVED as constraint, now diagnostic only ---
    # Reports total vehicle-arcs vs F_k to confirm multi-leg routing works
    # and that (16)+(19) keep usage physically plausible.
    print("\n--- Vehicle-arc usage diagnostic (constraint 17 removed) ---")
    for w in Omega:
        for m in modes:
            for k in K_m.get(m, []):
                total_arcs = sum(
                    n_var[w, k, m, i, j].X
                    for (i, j) in modal_arcs[m]
                    if (w, k, m, i, j) in n_var
                )
                F_k = vehicle_types[k]["fleet_size"]
                if total_arcs > 0.5:
                    multi_leg = "MULTI-LEG" if total_arcs > F_k + 0.5 else "single-leg"
                    print(
                        f"  w={w}, k={k}: {total_arcs:.0f} vehicle-arcs "
                        f"(fleet={F_k}) [{multi_leg}]"
                    )

    # --- (18) Vehicle-Capacity-Constrained Flow ---
    print("\n--- Checking Vehicle Capacity Flow (18) ---")
    violations_18 = 0
    binding_count = 0
    for w in Omega:
        for m in modes:
            for (i, j) in modal_arcs[m]:
                for r in R:
                    flow_val = x_var[w, m, i, j, r].X
                    vehicle_cap = sum(
                        vehicle_types[k]["capacity"][r] * n_var[w, k, m, i, j].X
                        for k in K_m.get(m, [])
                        if (w, k, m, i, j) in n_var
                    )
                    if flow_val > vehicle_cap + 1e-4:
                        print(
                            f"  VIOLATION: w={w}, m={m}, ({i},{j}), r={r}: "
                            f"flow={flow_val:.1f} > vcap={vehicle_cap:.1f}"
                        )
                        violations_18 += 1
                    elif flow_val > 1e-4 and vehicle_cap > 0 and flow_val > 0.95 * vehicle_cap:
                        binding_count += 1
    print(f"  Vehicle capacity violations: {violations_18}")
    print(f"  Near-binding flow/capacity pairs: {binding_count}")
    if violations_18 > 0:
        all_ok = False

    # --- (19) Distance Budget with p_j-coupled turnaround ---
    print("\n--- Checking Distance Budget (19) ---")
    violations_19 = 0
    for w in Omega:
        for k_name, vtype in vehicle_types.items():
            m_v = vtype["mode"]
            F_k = vtype["fleet_size"]
            D_k = vtype["D_k"]
            pi_k = vtype["pi_k"]
            b_kj = vtype["b_kj"]

            dist_consumed = sum(
                modal_arc_distance[m_v].get((i, j), 0) * n_var[w, k_name, m_v, i, j].X
                for (i, j) in modal_arcs[m_v]
                if (w, k_name, m_v, i, j) in n_var
            )
            # Turnaround at ALL nodes, exempted at base nodes only when selected
            total_outbound = sum(
                n_var[w, k_name, m_v, j, j_dst].X
                for j in N
                for (_, j_dst) in modal_outgoing[m_v][j]
                if (w, k_name, m_v, j, j_dst) in n_var
            )
            exempted = sum(
                n_var[w, k_name, m_v, j, j_dst].X
                for j in N
                if b_kj.get(j, 0) > 0 and j in PPL_set
                   and p_var[j].X > 0.5  # only exempt when SELECTED
                for (_, j_dst) in modal_outgoing[m_v][j]
                if (w, k_name, m_v, j, j_dst) in n_var
            )
            turnaround_consumed = pi_k * (total_outbound - exempted)
            total_consumed = dist_consumed + turnaround_consumed
            budget = F_k * D_k
            if total_consumed > budget + 1e-4:
                print(
                    f"  VIOLATION: w={w}, k={k_name}: "
                    f"consumed={total_consumed:.1f} > budget={budget:.1f}"
                )
                violations_19 += 1
            elif dist_consumed > 0.5:
                base_selected = [
                    j for j in N
                    if b_kj.get(j, 0) > 0 and j in PPL_set and p_var[j].X > 0.5
                ]
                base_not_sel = [
                    j for j in N
                    if b_kj.get(j, 0) > 0 and j in PPL_set and p_var[j].X < 0.5
                ]
                print(
                    f"  w={w}, k={k_name}: dist={dist_consumed:.0f} + "
                    f"turn={turnaround_consumed:.0f} = {total_consumed:.0f} / {budget:.0f}"
                    f"  (bases_sel={base_selected}, bases_not={base_not_sel})"
                )
    print(f"  Distance budget violations: {violations_19}")
    if violations_19 > 0:
        all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("ALL VEHICLE CONSTRAINT CHECKS PASSED")
    else:
        print("CONSTRAINT VIOLATIONS DETECTED")
    print(f"{'=' * 60}")
    return all_ok


def main():
    print("=" * 60)
    print("TOY VEHICLE-HETEROGENEITY VERIFICATION TEST")
    print("4 nodes, 8 arcs, 5 scenarios, 4 vehicle types")
    print("=" * 60)

    instance = build_toy_instance()

    print("\nVehicle basing allocation (b_{k,j}):")
    for k, vtype in instance["vehicle_types"].items():
        eligible = vtype["J_k"]
        print(
            f"  {k:>8} (mode={vtype['mode']}, fleet={vtype['fleet_size']}, "
            f"eligible={eligible}): b = {vtype['b_kj']}"
        )

    print("\nDemand at node 4 (disaster target):")
    for w in instance["scenarios"]:
        food = instance["demand"][(w, 4, "food")]
        water = instance["demand"][(w, 4, "water")]
        print(f"  w={w}: food={food:.0f}, water={water:.0f}")

    print("\nSolving...")
    results = solve_stochastic_cvar(
        instance, time_limit=120, mip_gap=load_parameters()["mip_gap"], verbose=True,
    )

    print_solution_summary(results, max_flows=10)

    print("\n\n" + "=" * 60)
    print("VEHICLE FLOW SUMMARY")
    print("=" * 60)
    vf = results.get("vehicle_flows", {})
    if vf:
        for (w, k, m, i, j), count in sorted(vf.items()):
            print(f"  w={w}, {k} ({m}): {i}->{j} x{count}")
    else:
        print("  No vehicle movements in solution.")

    print("\n")
    ok = verify_vehicle_constraints(results, instance)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
