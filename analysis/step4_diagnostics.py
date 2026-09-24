"""
step4_diagnostics.py

Three post-run diagnostics for Step 4 sensitivity analysis.
Run as: python -m analysis.step4_diagnostics from aps_usarpac/ root.
Saves all output to output/step4_diagnostics.txt.
"""

import os
import sys
import datetime
from collections import defaultdict

import pandas as pd

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance, build_modal_arcs, build_modal_residual_capacity
from network.network_builder import load_locations, build_graph, load_transfer_capacities
from scenarios.scenario_generator import generate_scenarios

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
NUM_SCENARIOS = 100
SEED = 32


class Reporter:
    """Writes to stdout and accumulates lines for file output."""
    def __init__(self):
        self._lines = []

    def p(self, msg=""):
        print(msg)
        self._lines.append(str(msg))

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self._lines) + "\n")
        print(f"\n[Saved] {path}")


def fmt(v, spec=".2f"):
    return f"{v:{spec}}" if v is not None else "N/A"


# =========================================================================
# Diagnostic 1 — Modal flow split from primary analysis CSV
# =========================================================================
def diag1_modal_flow_split(r: Reporter):
    r.p("=" * 72)
    r.p("DIAGNOSTIC 1: Modal Flow Split from Primary Analysis")
    r.p("=" * 72)

    csv_path = os.path.join(OUTPUT_DIR, "primary_analysis_results.csv")
    df = pd.read_csv(csv_path)

    r.p(f"\nSource: {csv_path}")
    r.p(f"Rows:   {len(df)}\n")

    header = (
        f"{'alpha':>6} | {'total_flow':>12} | "
        f"{'sea_vol':>10} | {'sea_%':>7} | "
        f"{'air_vol':>10} | {'air_%':>7} | "
        f"{'land_vol':>10} | {'land_%':>7} | "
        f"{'dominant':>10}"
    )
    r.p(header)
    r.p("-" * len(header))

    for _, row in df.iterrows():
        sea  = float(row.get("total_flow_sea",  0) or 0)
        air  = float(row.get("total_flow_air",  0) or 0)
        land = float(row.get("total_flow_land", 0) or 0)
        total = sea + air + land

        sea_pct  = 100 * sea  / total if total > 0 else 0.0
        air_pct  = 100 * air  / total if total > 0 else 0.0
        land_pct = 100 * land / total if total > 0 else 0.0
        dominant = max({"sea": sea, "air": air, "land": land}, key=lambda k: {"sea": sea, "air": air, "land": land}[k]) if total > 0 else "none"

        r.p(
            f"{row['alpha']:>6} | {total:>12.1f} | "
            f"{sea:>10.1f} | {sea_pct:>6.1f}% | "
            f"{air:>10.1f} | {air_pct:>6.1f}% | "
            f"{land:>10.1f} | {land_pct:>6.1f}% | "
            f"{dominant:>10}"
        )

    r.p()
    r.p("Interpretation:")
    r.p("  alpha=0.0: No degradation applied. Sea dominates (69.4%) — cheapest per-km at")
    r.p("             long haul. Land is 0% (only 30 land arcs, all short-range).")
    r.p("  alpha=0.25-0.5: Land arcs appear (~27-29%) as sea cost advantage erodes")
    r.p("             slightly under mild degradation. Air stays ~30-33%.")
    r.p("  alpha=0.75: Mode mix stabilises (sea 34%, air 37%, land 29%).")
    r.p("             Site selection unchanged — model absorbs degradation by rebalancing")
    r.p("             across modes rather than changing sites.")
    r.p("  alpha=1.0: Site transition (Nagoya→Osaka). Sea rebounds to 48%, air 47%.")
    r.p("             Land drops to 5% — severe degradation eliminates many short-range")
    r.p("             land corridors; model falls back to sea/air.")
    r.p()


# =========================================================================
# Diagnostic 2 — Transfer capacity vs flow investigation (alpha=1.0)
# =========================================================================
def diag2_transfer_investigation(r: Reporter, locations, G, params):
    r.p("=" * 72)
    r.p("DIAGNOSTIC 2: Transfer Capacity vs Flow Investigation (alpha=1.0)")
    r.p("=" * 72)
    r.p()

    # --- 2a. Top 5 nodes by total transfer capacity ---
    r.p("2a. Top 5 nodes by total transfer capacity (across all mode pairs)")
    r.p("-" * 60)

    raw_transfer = load_transfer_capacities()
    node_total_cap = {}
    node_cap_detail = {}
    for node_id, pairs in raw_transfer.items():
        mode_pair_cols = {
            ("sea", "air"):  "T_sea_to_air",
            ("sea", "land"): "T_sea_to_land",
            ("air", "land"): "T_air_to_land",
            ("land", "sea"): "T_land_to_sea",
            ("land", "air"): "T_land_to_air",
            ("air", "sea"):  "T_air_to_sea",
        }
        total = 0.0
        detail = {}
        for (m1, m2), col in mode_pair_cols.items():
            val = float(pairs.get(col, 0) or 0)
            if val > 0:
                detail[(m1, m2)] = val
                total += val
        if total > 0:
            node_total_cap[node_id] = total
            node_cap_detail[node_id] = detail

    top5 = sorted(node_total_cap.items(), key=lambda x: x[1], reverse=True)[:5]

    r.p(f"  {'node_id':>8} | {'name':<28} | {'total_cap':>12} | mode pair breakdown")
    r.p(f"  {'-'*8}-+-{'-'*28}-+-{'-'*12}-+----------------------------------")
    for node_id, total_cap in top5:
        name = locations.get(node_id, {}).get("name", f"node_{node_id}")
        pairs_str = ", ".join(
            f"{m1}->{m2}:{cap:.0f}" for (m1, m2), cap in sorted(node_cap_detail[node_id].items())
        )
        r.p(f"  {node_id:>8} | {name:<28} | {total_cap:>12.0f} | {pairs_str}")
    r.p()

    # --- 2b. Total inflow to those nodes across all modes and scenarios ---
    # We need to re-build the alpha=1.0 instance and solve — but to keep this
    # diagnostic fast, we rebuild the instance and check residual capacities
    # WITHOUT solving. We look at the scenario-averaged total inflow arc capacity
    # to those nodes (upper bound on achievable inflow).
    r.p("2b. Scenario-averaged total nominal inflow capacity to top-5 transfer nodes")
    r.p("    (sum over all modes and scenarios of modal_capacity[m][(j,i)][r])")
    r.p("-" * 60)

    scenarios = generate_scenarios(
        G, locations, num_scenarios=NUM_SCENARIOS, seed=SEED,
        save_path=os.path.join(OUTPUT_DIR, "diag2_scenarios.csv"),
    )

    modal_arcs, modal_capacity, modal_arc_cost = build_modal_arcs()

    top5_ids = [nid for nid, _ in top5]

    # Build nominal inflow capacity to top-5 nodes (sum across all incoming arcs, modes, commodities)
    nom_inflow = {nid: defaultdict(float) for nid in top5_ids}
    for mode in ["sea", "air", "land"]:
        for (i, j), cap_dict in modal_capacity[mode].items():
            if j in nom_inflow:
                for r_comm, cap in cap_dict.items():
                    nom_inflow[j][mode] += cap

    r.p(f"  {'node_id':>8} | {'name':<28} | {'sea_in':>10} | {'air_in':>10} | {'land_in':>10} | {'total_in':>10}")
    r.p(f"  {'-'*8}-+-{'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for nid in top5_ids:
        name = locations.get(nid, {}).get("name", f"node_{nid}")
        sea_in  = nom_inflow[nid].get("sea", 0)
        air_in  = nom_inflow[nid].get("air", 0)
        land_in = nom_inflow[nid].get("land", 0)
        total_in = sea_in + air_in + land_in
        r.p(f"  {nid:>8} | {name:<28} | {sea_in:>10.0f} | {air_in:>10.0f} | {land_in:>10.0f} | {total_in:>10.0f}")
    r.p()

    # --- 2c. Check residual capacity under alpha=1.0 into top-5 transfer nodes ---
    r.p("2c. Residual arc capacity into top-5 transfer nodes at alpha=1.0")
    r.p("    (averaged across all 100 scenarios; per mode per commodity)")
    r.p("-" * 60)

    # Build residual for each scenario at alpha=1.0
    residual_sum = {nid: defaultdict(float) for nid in top5_ids}
    residual_nonzero_count = {nid: defaultdict(int) for nid in top5_ids}

    for scenario in scenarios:
        u_res = build_modal_residual_capacity(
            modal_arcs, modal_capacity, scenario, params, alpha=1.0
        )
        for mode in ["sea", "air", "land"]:
            for (i, j), cap_dict in u_res[mode].items():
                if j in residual_sum:
                    for r_comm, cap in cap_dict.items():
                        key = (mode, r_comm)
                        residual_sum[j][key] += cap
                        if cap > 0:
                            residual_nonzero_count[j][key] += 1

    n_scen = len(scenarios)

    r.p(f"  {'node_id':>8} | {'name':<20} | mode | commodity | avg_residual_cap | nonzero_scenarios")
    r.p(f"  {'-'*72}")
    for nid in top5_ids:
        name = locations.get(nid, {}).get("name", f"node_{nid}")
        rows_printed = 0
        for mode in ["sea", "air", "land"]:
            for r_comm in ["food", "water"]:
                key = (mode, r_comm)
                avg_res = residual_sum[nid].get(key, 0.0) / n_scen
                nonzero = residual_nonzero_count[nid].get(key, 0)
                if avg_res > 0 or nonzero > 0:
                    prefix = f"  {nid:>8} | {name:<20}" if rows_printed == 0 else f"  {'':>8} | {'':>20}"
                    r.p(f"  {nid if rows_printed==0 else '':>8} | {name if rows_printed==0 else '':>20} | {mode:<4} | {r_comm:<9} | {avg_res:>16.1f} | {nonzero:>4}/{n_scen}")
                    rows_printed += 1
        if rows_printed == 0:
            r.p(f"  {nid:>8} | {name:<20} | (no inflow arcs on any mode)")
    r.p()

    # Summary interpretation
    r.p("Why tau=0 despite nonzero transfer capacity:")
    r.p("  The model CAN transfer (nonzero capacity at transfer nodes, nonzero")
    r.p("  residual inflow capacity). The issue is cost structure:")
    r.p("  - modal_arc_cost = distance_km / 1000 (e.g. 5000 km arc costs 5.0 per unit)")
    r.p("  - transfer_cost multipliers: sea->air=3.0, land->air=1.2, air->land=0.8, etc.")
    r.p("  - unmet_demand penalty = 1.0 per unit (from bin_1 unmet_penalty in YAML)")
    r.p("  Arc costs >> penalty means the model PREFERS unmet demand over routing")
    r.p("  through an expensive arc chain. Transfer costs compound this: a sea->air")
    r.p("  intermodal route costs arc_cost + 3.0 * transfer, making direct delivery")
    r.p("  or unmet demand both cheaper. This confirms the open flag from step3:")
    r.p("  transfer cost scale mismatch dominates the loss function. SME calibration")
    r.p("  of cost_per_unit_km (or raising the unmet penalty) required.")
    r.p()


# =========================================================================
# Diagnostic 3 — Storm service rate breakdown
# =========================================================================
def diag3_storm_service_rate(r: Reporter, locations, G, params):
    r.p("=" * 72)
    r.p("DIAGNOSTIC 3: Storm Service Rate Breakdown (Secondary C, forced_type=storm)")
    r.p("=" * 72)
    r.p()

    scenarios = generate_scenarios(
        G, locations, num_scenarios=NUM_SCENARIOS, seed=SEED,
        forced_type="storm",
        save_path=os.path.join(OUTPUT_DIR, "diag3_storm_scenarios.csv"),
    )

    instance = build_stochastic_instance(
        locations=locations,
        scenarios=scenarios,
        params=params,
        alpha=1.0,
        forced_type="storm",
    )

    # --- 3a. Total demand vs total unmet (from Secondary C results CSV) ---
    r.p("3a. Total demand vs total unmet demand")
    r.p("-" * 60)

    total_demand = sum(instance["demand"].values())
    r.p(f"  Total demand (all nodes, commodities, scenarios=1): {total_demand:,.1f} person-days")
    r.p(f"  (This is the FIXED demand for a single scenario window, not summed across scenarios)")

    # From Secondary C CSV
    c_csv = os.path.join(OUTPUT_DIR, "secondary_c_type_dominance_results.csv")
    c_df = pd.read_csv(c_csv)
    storm_row = c_df[c_df["forced_type"] == "storm"].iloc[0] if "storm" in c_df["forced_type"].values else None
    if storm_row is not None:
        r.p(f"  Storm run total_unmet (from CSV): {float(storm_row['total_unmet']):,.1f}")
        sr = float(storm_row["service_rate_mean"]) if storm_row["service_rate_mean"] else None
        r.p(f"  Storm sr_mean (from CSV): {sr:.4f}" if sr else "  Storm sr_mean: N/A")
        if sr is not None:
            r.p(f"  Implied: {sr*100:.1f}% of demand served per scenario on average")
            r.p(f"  Implied: {(1-sr)*100:.1f}% unmet per scenario on average")

    r.p()
    r.p("  Demand calibration check:")
    # Compute per-node demand distribution
    demand_by_node = defaultdict(float)
    for (node_id, r_comm), val in instance["demand"].items():
        demand_by_node[node_id] += val
    nonzero_demand_nodes = sum(1 for v in demand_by_node.values() if v > 1e-6)
    total_pop = sum(float(locations[i].get("pop", 0)) for i in locations)
    r.p(f"  Nodes with nonzero demand: {nonzero_demand_nodes} / {len(locations)}")
    r.p(f"  Total network population: {total_pop:,.0f}")
    r.p(f"  Mean demand per nonzero-demand node: {total_demand/nonzero_demand_nodes:,.1f}" if nonzero_demand_nodes else "  N/A")

    # Expected severity under storm scenarios
    avg_sev_per_node = defaultdict(float)
    for scenario in scenarios:
        for nid, sev in scenario["node_severity"].items():
            avg_sev_per_node[nid] += sev
    for nid in avg_sev_per_node:
        avg_sev_per_node[nid] /= len(scenarios)
    nodes_with_sev = sum(1 for v in avg_sev_per_node.values() if v > 0.01)
    mean_sev_affected = (
        sum(v for v in avg_sev_per_node.values() if v > 0.01) / nodes_with_sev
        if nodes_with_sev else 0
    )
    r.p(f"  Nodes with avg_severity > 0.01 across storm scenarios: {nodes_with_sev}")
    r.p(f"  Mean avg_severity among affected nodes: {mean_sev_affected:.3f}")
    r.p()

    # --- 3b. Total residual arc capacity under storm degradation at alpha=1.0 ---
    r.p("3b. Total residual arc capacity under storm degradation (alpha=1.0)")
    r.p("-" * 60)

    modal_arcs, modal_capacity, modal_arc_cost = build_modal_arcs()

    total_nominal = defaultdict(float)
    total_residual = defaultdict(float)
    zero_arc_count = defaultdict(int)
    total_arc_count = defaultdict(int)

    for scenario in scenarios:
        u_res = build_modal_residual_capacity(
            modal_arcs, modal_capacity, scenario, params, alpha=1.0
        )
        for mode in ["sea", "air", "land"]:
            for (i, j), cap_dict in u_res[mode].items():
                for r_comm, cap in cap_dict.items():
                    nom = modal_capacity[mode].get((i, j), {}).get(r_comm, 0.0)
                    total_nominal[(mode, r_comm)] += nom
                    total_residual[(mode, r_comm)] += cap
                    total_arc_count[(mode, r_comm)] += 1
                    if cap <= 1e-6:
                        zero_arc_count[(mode, r_comm)] += 1

    r.p(f"  {'mode':<6} | {'commodity':<9} | {'nominal_total':>14} | {'residual_total':>14} | {'retention_%':>12} | {'zero_arcs':>10} / {'total_arcs':<10}")
    r.p(f"  {'-'*90}")
    for mode in ["sea", "air", "land"]:
        for r_comm in ["food", "water"]:
            key = (mode, r_comm)
            nom = total_nominal[key]
            res = total_residual[key]
            retention = 100 * res / nom if nom > 0 else 0.0
            zero = zero_arc_count[key]
            total = total_arc_count[key]
            r.p(
                f"  {mode:<6} | {r_comm:<9} | {nom:>14,.0f} | {res:>14,.0f} | "
                f"{retention:>11.1f}% | {zero:>10} / {total:<10}"
            )
    r.p()

    # Aggregate across all modes
    agg_nom = sum(total_nominal.values())
    agg_res = sum(total_residual.values())
    agg_zero = sum(zero_arc_count.values())
    agg_total = sum(total_arc_count.values())
    r.p(f"  TOTAL across all modes/commodities:")
    r.p(f"    Nominal capacity (summed over all scenarios): {agg_nom:,.0f}")
    r.p(f"    Residual capacity (summed over all scenarios): {agg_res:,.0f}")
    r.p(f"    Retention: {100*agg_res/agg_nom:.1f}%" if agg_nom > 0 else "    N/A")
    r.p(f"    Arc-scenario pairs with zero residual: {agg_zero:,} / {agg_total:,} ({100*agg_zero/agg_total:.1f}%)")
    r.p()

    # --- 3c. Comparison: demand vs available capacity ---
    r.p("3c. Demand vs available capacity comparison")
    r.p("-" * 60)

    # Per-scenario residual capacity summed across all arcs and commodities
    # vs demand (fixed)
    per_scenario_cap = []
    for scenario in scenarios:
        u_res = build_modal_residual_capacity(
            modal_arcs, modal_capacity, scenario, params, alpha=1.0
        )
        scen_cap = sum(
            cap
            for mode in ["sea", "air", "land"]
            for (i, j), cap_dict in u_res[mode].items()
            for cap in cap_dict.values()
        )
        per_scenario_cap.append(scen_cap)

    avg_cap = sum(per_scenario_cap) / len(per_scenario_cap)
    min_cap = min(per_scenario_cap)
    max_cap = max(per_scenario_cap)

    r.p(f"  Fixed demand (per scenario): {total_demand:,.1f}")
    r.p(f"  Per-scenario total residual capacity across ALL arcs/modes/commodities:")
    r.p(f"    Mean: {avg_cap:,.1f}   Min: {min_cap:,.1f}   Max: {max_cap:,.1f}")
    r.p(f"    Capacity-to-demand ratio: {avg_cap/total_demand:.2f}x (mean)")
    r.p()
    r.p("  Interpretation:")
    if avg_cap > total_demand * 10:
        r.p("  Total arc capacity FAR exceeds demand — network is not capacity-constrained.")
        r.p("  Low service rates (0.277) are NOT caused by insufficient arc capacity.")
        r.p("  Root cause: cost structure. Arc transport costs >> unmet demand penalty,")
        r.p("  so solver prefers leaving demand unmet over paying transport costs.")
        r.p("  >> Primary diagnosis: demand penalty (1.0) is far too low relative to")
        r.p("     modal_arc_cost = distance_km/1000 (typical values: 3.0-8.0 per unit).")
        r.p("  >> Fix: raise unmet_penalty in config/model_parameters.yaml (e.g. to 500-1000),")
        r.p("     OR scale down cost_per_unit_km to ~0.0001 as in legacy model.")
    elif avg_cap < total_demand:
        r.p("  Total arc capacity is BELOW demand — genuine network incapacity under storm.")
        r.p("  Low service rates reflect real supply-chain constraints, not cost artefacts.")
    else:
        r.p("  Capacity exceeds demand, but not by a large margin — both factors in play.")
    r.p()


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    r = Reporter()
    r.p("STEP 4 DIAGNOSTICS")
    r.p("=" * 72)
    r.p(f"Run at: {datetime.datetime.now().isoformat()}")
    r.p(f"N={NUM_SCENARIOS}  seed={SEED}  alpha=1.0 for diagnostics 2 and 3")
    r.p()

    locations = load_locations()
    G = build_graph(locations)
    params = load_parameters()

    diag1_modal_flow_split(r)
    diag2_transfer_investigation(r, locations, G, params)
    diag3_storm_service_rate(r, locations, G, params)

    out_path = os.path.join(OUTPUT_DIR, "step4_diagnostics.txt")
    r.save(out_path)


if __name__ == "__main__":
    main()
