"""
vehcap_diagnostic.py

ONE-OFF diagnostic, run before Phase 5's real-network integration code.
Question: does eq:vif:vehcap's min{T^w_l,ij, cap_l} term ever have T as the
binding (smaller) side on the real 50-node network, or does cap_l always
win -- in which case arc-throughput degradation (eq:residual) can never
influence solve_vif's routing decisions no matter how severe alpha/gamma
gets, since C7 always reduces to the same cap_l ceiling either way.

No constraint in the gospel sums T or n over l (C7 is per-vehicle,
per-arc, per-scenario -- unlike C8's node handling capacity, which does
sum n over L_m). So T isn't a pooled arc-total ceiling; it only matters
if it is EVER smaller than a single vehicle's own payload cap on some
arc it might use.

Method:
  1. Load the real 50-node network (network/nodes.csv) and the real
     config (config/model_parameters.yaml).
  2. Generate one scenario set (seed=32, matching every other script in
     this repo -- solve behavior is irrelevant here, only node_severity/
     disaster_type draws matter, and this is a pure arithmetic check, not
     a solve) and pick the scenario whose epicenter severity is closest
     to 3.0, the midpoint of the gospel's [1,5] severity scale.
  3. For every individual vehicle l in L (96, matching the gospel's
     instance-dimensions table) and every arc (i,j) in that vehicle's
     mode's arc set, compute T^w_l,ij (eq:residual, alpha=1.0 -- the full
     calibrated degradation_matrix, unscaled) and cap_l (cap_tons), and
     record which side of min{} is smaller.
  4. Separately, holding gamma_l = 1.0 directly (NOT alpha=1.0 scaling
     Gamma_{m,nu} -- the literal degradation sensitivity coefficient
     pinned to its own maximum, isolating "how degraded would arc
     throughput need to be, at every mode's worst-case sensitivity, for T
     to bind" from any specific scenario's disaster type), find the
     severity level sigma in [0,5] at which T^w_l,ij = T_l,ij*(1-sigma/5)
     first drops below cap_l, per mode. T_l,ij varies by arc within a
     mode (vessel/aircraft/road-rating dependent), so this is reported as
     a range across that mode's arcs, not a single number.

Run: python scripts/vehcap_diagnostic.py   (from aps_usarpac/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from config.loader import load_parameters
from model.input_builder import build_modal_arcs, build_nominal_throughput, build_vehicle_params
from network.network_builder import load_locations
from scenarios.scenario_generator import generate_scenarios


def _degradation_factor(gamma, severity_term):
    return max(0.0, 1.0 - gamma * severity_term / 5.0)


def main():
    params = load_parameters()
    locations = load_locations()

    scenarios = generate_scenarios(G=None, locations=locations, num_scenarios=100, seed=32)
    scenario = min(scenarios, key=lambda s: abs(s["severity"] - 3.0))
    print(f"Selected scenario_id={scenario['scenario_id']}, "
          f"epicenter={scenario['epicenter']}, severity={scenario['severity']:.3f}, "
          f"disaster_type={scenario['disaster_type']}")

    node_severity = scenario["node_severity"]
    disaster_type = scenario["disaster_type"]
    degradation_matrix = params["degradation_matrix"]

    modal_arcs, _modal_capacity, _modal_arc_cost, _modal_arc_distance = build_modal_arcs()
    nominal_throughput = build_nominal_throughput(params)
    vehicle_data = build_vehicle_params(locations, params)
    vehicle_types = vehicle_data["vehicle_types"]

    K = sorted(vehicle_types.keys())
    L = []
    type_of = {}
    mode_of = {}
    _next_l = 1
    for k in K:
        for _ in range(int(vehicle_types[k]["fleet_size"])):
            L.append(_next_l)
            type_of[_next_l] = k
            mode_of[_next_l] = vehicle_types[k]["mode"]
            _next_l += 1
    print(f"|L| = {len(L)} individual vehicles across {len(K)} types: "
          f"{ {k: vehicle_types[k]['fleet_size'] for k in K} }")

    # ---- Part 1: alpha in {0.0, 1.0}, real scenario -- fraction of (l,i,j)
    # where T wins. alpha=0.0 isolates the BASELINE (undegraded) case, so
    # comparing it against alpha=1.0 separates "T<cap_l purely from tier
    # structure/asset absence, independent of any disaster" from "T<cap_l
    # caused by eq:residual's disaster-driven degradation term."
    def run_alpha(alpha):
        gamma_w_m = {
            m: degradation_matrix.get(m, {}).get(disaster_type, 0.0) * alpha
            for m in ("sea", "air", "land")
        }
        T_wins = defaultdict(int)
        total = defaultdict(int)
        T_wins_by_type = defaultdict(int)
        total_by_type = defaultdict(int)
        for l in L:
            k = type_of[l]
            m = mode_of[l]
            cap_l = vehicle_types[k]["cap_tons"]
            T_nominal_m = nominal_throughput.get(m, {})
            for (i, j) in modal_arcs[m]:
                sigma_i = node_severity.get(i, 0.0)
                sigma_j = node_severity.get(j, 0.0)
                T_residual = T_nominal_m.get((i, j), 0.0) * _degradation_factor(
                    gamma_w_m[m], max(sigma_i, sigma_j)
                )
                total[m] += 1
                total_by_type[(m, k)] += 1
                if T_residual <= cap_l:
                    T_wins[m] += 1
                    T_wins_by_type[(m, k)] += 1
        return gamma_w_m, T_wins, total, T_wins_by_type, total_by_type

    gamma_0, Twin_0, tot_0, Twin_0_type, tot_0_type = run_alpha(0.0)
    gamma_1, Twin_1, tot_1, Twin_1_type, tot_1_type = run_alpha(1.0)

    print(f"\ngamma_w_m at alpha=1.0, disaster_type={disaster_type!r}: {gamma_1}")
    print("\n--- Part 1: fraction of (l,i,j) triples where T is the binding "
          "(<=) term in min{T, cap_l} ---")
    print(f"(scenario_id={scenario['scenario_id']}, severity={scenario['severity']:.3f}, "
          f"disaster_type={disaster_type})")
    print(f"{'mode':6s} {'alpha=0.0 (baseline)':>24s} {'alpha=1.0 (full degrad.)':>26s} "
          f"{'delta (degradation-caused)':>28s}")
    for m in ("sea", "air", "land"):
        f0 = Twin_0[m] / tot_0[m] if tot_0[m] else float("nan")
        f1 = Twin_1[m] / tot_1[m] if tot_1[m] else float("nan")
        delta = Twin_1[m] - Twin_0[m]
        print(f"  {m:5s} {Twin_0[m]:6d}/{tot_0[m]:6d} ({f0:6.2%})   "
              f"{Twin_1[m]:6d}/{tot_1[m]:6d} ({f1:6.2%})    "
              f"+{delta:6d} triples ({delta/tot_1[m]:.2%} of total)")
    print("  by vehicle type (alpha=0.0 baseline -> alpha=1.0 full degradation):")
    for (m, k) in sorted(tot_1_type):
        cap_l = vehicle_types[k]["cap_tons"]
        f0 = Twin_0_type[(m, k)] / tot_0_type[(m, k)]
        f1 = Twin_1_type[(m, k)] / tot_1_type[(m, k)]
        delta = Twin_1_type[(m, k)] - Twin_0_type[(m, k)]
        print(f"    mode={m:5s} type={k:10s} cap_l={cap_l:8.2f}MT  "
              f"baseline {Twin_0_type[(m, k)]:6d}/{tot_0_type[(m, k)]:6d} ({f0:6.2%})  ->  "
              f"degraded {Twin_1_type[(m, k)]:6d}/{tot_1_type[(m, k)]:6d} ({f1:6.2%})  "
              f"[+{delta} triples caused by degradation]")

    # ---- Part 2: gamma_l pinned to 1.0 -- severity threshold per mode ----
    print("\n--- Part 2: severity (sigma, at gamma_l=1.0, i.e. the max possible "
          "mode sensitivity) at which T first falls below cap_l, per mode/type ---")
    print("  (sigma* = 5*(1 - cap_l/T_l,ij); T_l,ij varies by arc within a mode, "
          "so reporting the range across that mode's arcs)")
    for m in ("sea", "air", "land"):
        T_nominal_m = nominal_throughput.get(m, {})
        arc_T_values = [T_nominal_m.get(a, 0.0) for a in modal_arcs[m]]
        arc_T_values = [t for t in arc_T_values if t > 0.0]
        if not arc_T_values:
            print(f"  mode={m:5s}: no positive-throughput arcs found")
            continue
        for k in K:
            if vehicle_types[k]["mode"] != m:
                continue
            cap_l = vehicle_types[k]["cap_tons"]
            thresholds = []
            for t in arc_T_values:
                if t <= cap_l:
                    thresholds.append(0.0)  # T already below cap_l at sigma=0 (undegraded)
                else:
                    thresholds.append(5.0 * (1.0 - cap_l / t))
            thresholds.sort()
            n_arcs = len(thresholds)
            print(f"  mode={m:5s} type={k:10s} cap_l={cap_l:8.2f}MT  "
                  f"sigma* range=[{thresholds[0]:.3f}, {thresholds[-1]:.3f}]  "
                  f"median={thresholds[n_arcs // 2]:.3f}  "
                  f"(min/max T on this mode's arcs: {min(arc_T_values):.2f}/"
                  f"{max(arc_T_values):.2f} MT, n_arcs={n_arcs})")


    # ---- Part 3: full 100-scenario sweep, each scenario's own disaster_type
    # -- is there ANY (w,l,i,j) triple, across the whole realistic draw, where
    # degradation (alpha=1.0) pushes T below cap_l when it wasn't already
    # (alpha=0.0) below cap_l on that same arc? This checks Part 1's
    # single-scenario finding isn't a fluke of picking severity~3.0/flood.
    print("\n--- Part 3: full 100-scenario sweep -- any genuine "
          "degradation-caused T<cap_l transition, using each scenario's own "
          "disaster_type? ---")
    degradation_caused_total = 0
    degradation_caused_by_mode_type = defaultdict(int)
    degradation_caused_by_scenario = []
    max_severity_seen = defaultdict(float)
    for sc in scenarios:
        w = sc["scenario_id"]
        sc_node_severity = sc["node_severity"]
        sc_disaster_type = sc["disaster_type"]
        gamma_1_sc = {
            m: degradation_matrix.get(m, {}).get(sc_disaster_type, 0.0) * 1.0
            for m in ("sea", "air", "land")
        }
        gamma_0_sc = {m: 0.0 for m in ("sea", "air", "land")}
        n_transitions = 0
        for l in L:
            k = type_of[l]
            m = mode_of[l]
            cap_l = vehicle_types[k]["cap_tons"]
            T_nominal_m = nominal_throughput.get(m, {})
            for (i, j) in modal_arcs[m]:
                sigma_i = sc_node_severity.get(i, 0.0)
                sigma_j = sc_node_severity.get(j, 0.0)
                sev_term = max(sigma_i, sigma_j)
                if sev_term > max_severity_seen[m]:
                    max_severity_seen[m] = sev_term
                T_nom = T_nominal_m.get((i, j), 0.0)
                T_base = T_nom * _degradation_factor(gamma_0_sc[m], sev_term)
                T_deg = T_nom * _degradation_factor(gamma_1_sc[m], sev_term)
                if T_base > cap_l and T_deg <= cap_l:
                    n_transitions += 1
                    degradation_caused_by_mode_type[(m, k)] += 1
        if n_transitions > 0:
            degradation_caused_by_scenario.append((w, sc_disaster_type, sc["severity"], n_transitions))
        degradation_caused_total += n_transitions

    print(f"  Total degradation-caused T<cap_l transitions across all 100 "
          f"scenarios x 96 vehicles x their mode's arcs: {degradation_caused_total}")
    print(f"  Scenarios with >=1 transition: {len(degradation_caused_by_scenario)} / 100")
    print("  Breakdown by (mode, type):")
    for (m, k), n in sorted(degradation_caused_by_mode_type.items()):
        print(f"    mode={m:5s} type={k:10s}  transitions={n}")
    if degradation_caused_by_scenario:
        print("  Scenarios where it occurred:")
        for (w, dt, sev, n) in degradation_caused_by_scenario:
            print(f"    scenario_id={w} disaster_type={dt} severity={sev:.3f}  "
                  f"transitions={n}")
    else:
        print("  None. Zero scenarios in the 100-scenario draw (seed=32) "
              "produced any degradation-caused transition.")
    print(f"  Max node-severity term (max(sigma_i,sigma_j)) observed on any "
          f"arc, by mode, across all 100 scenarios: "
          f"{dict(max_severity_seen)}")


if __name__ == "__main__":
    main()
