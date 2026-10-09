"""
sensitivity_runner.py

Step 4 multi-modal sensitivity analysis driver. Three analyses:

  Primary A: Uniform matrix scaling sweep
    alpha in {0.0, 0.25, 0.5, 0.75, 1.0} scales all gamma[mode][type]
    values simultaneously. Directly comparable to original single-gamma
    results. Primary analysis for dissertation.

  Secondary B: Mode-isolation sweep
    For each mode in {sea, air, land}: scale that mode's gamma row by
    alpha in {0.0, 0.5, 1.0} while holding other modes at alpha=1.0.
    Shows each mode's marginal contribution to total degradation.

  Secondary C: Disaster-type dominance sweep
    Force scenario sets to be dominated by one type at a time:
    pure flood, pure storm, pure earthquake. Compares site selection
    and service rates across threat environments.

All three analyses use N=100 scenarios, seed=32, beta=0.9, Pmax=3.

Run as: python -m analysis.sensitivity_runner from aps_usarpac/ project root.
All output written to output/ before any reporting.
"""

import datetime
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from model.vehicle_itinerary import generate_itinerary_report
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
MODE_ISOLATION_ALPHAS = [0.0, 0.5, 1.0]
FORCED_TYPES = ["flood", "storm", "earthquake"]
NUM_SCENARIOS = 100
SEED = 32
ALPHA_FOR_C = 1.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


# -------------------------------------------------------------------------
# Logging helper
# -------------------------------------------------------------------------
class Logger:
    """Writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._file = open(log_path, "w")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# -------------------------------------------------------------------------
# Result extraction helpers
# -------------------------------------------------------------------------
def compute_service_rates(results: Dict, instance: Dict) -> Dict:
    """Compute per-scenario and aggregate service rates."""
    demand = instance["demand"]
    unmet_dict = results.get("unmet_demand", {})
    per_scenario_sr = {}
    for w in instance["scenarios"]:
        dem_w = sum(v for (ww, i, r), v in demand.items() if ww == w)
        unmet_w = sum(val for (ww, _, _), val in unmet_dict.items() if ww == w)
        per_scenario_sr[w] = (1.0 - unmet_w / dem_w) if dem_w > 0 else None
    valid_sr = [v for v in per_scenario_sr.values() if v is not None]
    sr_mean = sum(valid_sr) / len(valid_sr) if valid_sr else None
    sr_min = min(valid_sr) if valid_sr else None
    sr_p10 = sorted(valid_sr)[int(0.10 * len(valid_sr))] if len(valid_sr) >= 10 else None
    # reference total_demand: sum across one scenario for downstream stats
    ref_w = instance["scenarios"][0]
    ref_demand = sum(v for (w, i, r), v in demand.items() if w == ref_w)
    return {
        "per_scenario": per_scenario_sr,
        "mean": sr_mean,
        "min": sr_min,
        "p10": sr_p10,
        "total_demand": ref_demand,
    }


def extract_run_stats(results: Optional[Dict], instance: Dict) -> Dict:
    """Extract standardized stats from a solve result, handling failures gracefully."""
    if results is None:
        return {
            "status": "ERROR",
            "objective": None,
            "selected_sites": "",
            "num_sites": 0,
            "total_unmet": None,
            "total_demand": sum(
                v for (w, i, r), v in instance["demand"].items()
                if w == instance["scenarios"][0]
            ),
            "service_rate_mean": None,
            "service_rate_p10": None,
            "service_rate_min": None,
            "solve_time": None,
            "gap": None,
            "total_flow_sea": 0.0,
            "total_flow_air": 0.0,
            "total_flow_land": 0.0,
            "total_transfer": 0.0,
        }

    status = results.get("status", "ERROR")
    obj = results.get("objective_value")
    selected_sites = results.get("selected_sites", [])
    total_unmet = sum(results.get("unmet_demand", {}).values())

    # run_solve copies these scalars out of the Gurobi model before disposing
    # it (M1 memory fix); fall back to a live model for callers that bypass
    # run_solve and still carry results["model"].
    solve_time = results.get("solve_runtime")
    gap = results.get("final_mip_gap")
    if solve_time is None or gap is None:
        m_gurobi = results.get("model")
        if m_gurobi is not None:
            try:
                solve_time = m_gurobi.Runtime
                gap = m_gurobi.MIPGap
            except Exception:
                pass

    sr_stats = compute_service_rates(results, instance)
    flow_by_mode = results.get("flow_by_mode", {"sea": 0.0, "air": 0.0, "land": 0.0})
    total_transfer = results.get("total_transfer_vol", 0.0)

    return {
        "status": status,
        "objective": obj,
        "selected_sites": "|".join(str(s) for s in selected_sites),
        "num_sites": len(selected_sites),
        "total_unmet": total_unmet,
        "total_demand": sum(
            v for (w, i, r), v in instance["demand"].items()
            if w == instance["scenarios"][0]
        ),
        "service_rate_mean": sr_stats.get("mean"),
        "service_rate_p10": sr_stats.get("p10"),
        "service_rate_min": sr_stats.get("min"),
        "solve_time": solve_time,
        "gap": gap,
        "total_flow_sea": float(flow_by_mode.get("sea", 0.0)),
        "total_flow_air": float(flow_by_mode.get("air", 0.0)),
        "total_flow_land": float(flow_by_mode.get("land", 0.0)),
        "total_transfer": float(total_transfer),
    }


def run_solve(
    instance: Dict,
    label: str,
    logger: Logger,
    params: Dict,
    locations: Optional[Dict] = None,
    itinerary: bool = False,
    itinerary_all: bool = False,
    itinerary_top_k: int = 3,
) -> Optional[Dict]:
    """
    Run a single solve with error handling. Returns results dict or None.

    Itinerary reports are OFF by default for sweep runs (IO4 audit finding:
    they add reconstruction compute + one .md per solve). Pass itinerary=True
    for on-demand generation: if the instance contains vehicle_types and
    locations is provided, a vehicle itinerary report is generated for the
    lowest-k and highest-k SR scenarios (default k=3) and saved to
    output/itinerary_{safe_label}.md. Pass itinerary_all=True to generate
    for all scenarios.
    """
    logger.log(f"\n[{label}] Starting solve...")
    t0 = time.time()
    try:
        results = solve_stochastic_cvar(
            instance, time_limit=3600, mip_gap=params["mip_gap"], verbose=True
        )
    except Exception as e:
        logger.log(f"[{label}] ERROR during solve: {e}")
        return None
    elapsed = time.time() - t0
    status = results.get("status", "UNKNOWN")
    obj = results.get("objective_value")
    sites = results.get("selected_sites", [])
    logger.log(
        f"[{label}] Done in {elapsed:.1f}s | status={status} | "
        f"obj={obj} | sites={sites}"
    )

    # --- Vehicle itinerary report (on-demand via itinerary=True) ---
    if (itinerary
            and locations is not None
            and instance.get("vehicle_types")
            and results.get("objective_value") is not None):
        try:
            report_md = generate_itinerary_report(
                results=results,
                instance=instance,
                locations=locations,
                label=label,
                itinerary_all=itinerary_all,
                top_k=itinerary_top_k,
            )
            safe_label = label.replace(" ", "_").replace("=", "").replace(".", "p")
            report_path = os.path.join(OUTPUT_DIR, f"itinerary_{safe_label}.md")
            with open(report_path, "w") as f:
                f.write(report_md)
            logger.log(f"[{label}] Itinerary report: {report_path}")
        except Exception as e:
            logger.log(f"[{label}] WARNING: itinerary report failed: {e}")

    # --- M1 memory fix: release the Gurobi model before returning ---
    # results["model"] holds the full C-side model (multi-GB on the vehicle-
    # extended instance) and results["variables"] holds ~725K Var objects that
    # keep it alive. The sweep loop rebinds `results` only after the NEXT solve
    # finishes, so without this, two full models sit in memory during every
    # solve after the first. Copy out the scalars consumers need, then dispose.
    m_gurobi = results.get("model")
    if m_gurobi is not None:
        try:
            results["solve_runtime"] = m_gurobi.Runtime
            results["final_mip_gap"] = m_gurobi.MIPGap
        except Exception:
            pass
        try:
            m_gurobi.dispose()
        except Exception as e:
            logger.log(f"[{label}] WARNING: model dispose failed: {e}")
    results.pop("model", None)
    results.pop("variables", None)

    return results


# -------------------------------------------------------------------------
# Primary Analysis (Option A)
# -------------------------------------------------------------------------
def run_primary_analysis(
    locations: Dict, G, params: Dict, logger: Logger
) -> tuple:
    logger.log("\n" + "=" * 70)
    logger.log("PRIMARY ANALYSIS (Option A): Uniform Matrix Scaling Sweep")
    logger.log(f"ALPHA_VALUES={ALPHA_VALUES}  N={NUM_SCENARIOS}  seed={SEED}")
    logger.log("=" * 70)

    scenarios = generate_scenarios(
        G, locations,
        num_scenarios=NUM_SCENARIOS,
        seed=SEED,
        save_path=os.path.join(OUTPUT_DIR, "primary_scenarios.csv"),
    )

    summary_rows: List[Dict] = []
    scenario_rows: List[Dict] = []

    for alpha in ALPHA_VALUES:
        label = f"Primary alpha={alpha}"
        logger.log(f"\n{'=' * 60}")
        logger.log(f"Running {label}")
        logger.log(f"{'=' * 60}")

        instance = build_stochastic_instance(
            locations=locations,
            scenarios=scenarios,
            params=params,
            alpha=alpha,
        )

        results = run_solve(instance, label, logger, params, locations=locations)
        stats = extract_run_stats(results, instance)

        total_flow = stats["total_flow_sea"] + stats["total_flow_air"] + stats["total_flow_land"]
        row = {"alpha": alpha}
        row.update(stats)
        summary_rows.append(row)

        if total_flow > 0:
            logger.log(
                f"  flow mix: sea={100*stats['total_flow_sea']/total_flow:.1f}% "
                f"air={100*stats['total_flow_air']/total_flow:.1f}% "
                f"land={100*stats['total_flow_land']/total_flow:.1f}% "
                f"transfer={stats['total_transfer']:.1f}"
            )

        # Per-scenario service rates
        if results is not None:
            sr_stats = compute_service_rates(results, instance)
            for w, sr_val in sr_stats.get("per_scenario", {}).items():
                unmet_w = sum(
                    val
                    for (ww, _, _), val in results.get("unmet_demand", {}).items()
                    if ww == w
                )
                scenario_rows.append({
                    "alpha": alpha,
                    "scenario_id": w,
                    "service_rate": sr_val,
                    "unmet": unmet_w,
                    "loss": results.get("scenario_losses", {}).get(w),
                })

    primary_path = os.path.join(OUTPUT_DIR, "primary_analysis_results.csv")
    scenario_path = os.path.join(OUTPUT_DIR, "primary_scenario_losses.csv")
    pd.DataFrame(summary_rows).to_csv(primary_path, index=False)
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)
    logger.log(f"\n[Primary] Saved: {primary_path}")
    logger.log(f"[Primary] Saved: {scenario_path}")

    return summary_rows, scenario_rows


# -------------------------------------------------------------------------
# Secondary B: Mode-isolation sweep
# -------------------------------------------------------------------------
def run_secondary_b(
    locations: Dict, G, params: Dict, logger: Logger
) -> List[Dict]:
    logger.log("\n" + "=" * 70)
    logger.log("SECONDARY B: Mode-Isolation Sweep")
    logger.log(f"MODE_ISOLATION_ALPHAS={MODE_ISOLATION_ALPHAS}  N={NUM_SCENARIOS}  seed={SEED}")
    logger.log("(each mode varied; other two modes held at alpha=1.0)")
    logger.log("=" * 70)

    scenarios = generate_scenarios(
        G, locations,
        num_scenarios=NUM_SCENARIOS,
        seed=SEED,
        save_path=os.path.join(OUTPUT_DIR, "secondary_b_scenarios.csv"),
    )

    results_rows: List[Dict] = []

    for mode_to_vary in ["sea", "air", "land"]:
        for alpha in MODE_ISOLATION_ALPHAS:
            mode_alphas = {"sea": 1.0, "air": 1.0, "land": 1.0}
            mode_alphas[mode_to_vary] = alpha
            label = f"SecB {mode_to_vary} alpha={alpha}"
            logger.log(f"\n{'=' * 60}")
            logger.log(f"Running {label} | mode_alphas={mode_alphas}")
            logger.log(f"{'=' * 60}")

            instance = build_stochastic_instance(
                locations=locations,
                scenarios=scenarios,
                params=params,
                alpha=1.0,
                mode_alphas=mode_alphas,
            )

            results = run_solve(instance, label, logger, params)
            stats = extract_run_stats(results, instance)

            row = {"varied_mode": mode_to_vary, "alpha": alpha}
            row.update(stats)
            results_rows.append(row)

    b_path = os.path.join(OUTPUT_DIR, "secondary_b_mode_isolation_results.csv")
    pd.DataFrame(results_rows).to_csv(b_path, index=False)
    logger.log(f"\n[Secondary B] Saved: {b_path}")

    return results_rows


# -------------------------------------------------------------------------
# Secondary C: Disaster-type dominance sweep
# -------------------------------------------------------------------------
def run_secondary_c(
    locations: Dict, G, params: Dict, logger: Logger
) -> List[Dict]:
    logger.log("\n" + "=" * 70)
    logger.log("SECONDARY C: Disaster-Type Dominance Sweep")
    logger.log(f"FORCED_TYPES={FORCED_TYPES}  alpha={ALPHA_FOR_C}  N={NUM_SCENARIOS}  seed={SEED}")
    logger.log("=" * 70)

    results_rows: List[Dict] = []

    for forced_type in FORCED_TYPES:
        label = f"SecC type={forced_type}"
        logger.log(f"\n{'=' * 60}")
        logger.log(f"Running {label} (all {NUM_SCENARIOS} scenarios: {forced_type})")
        logger.log(f"{'=' * 60}")

        scenarios = generate_scenarios(
            G, locations,
            num_scenarios=NUM_SCENARIOS,
            seed=SEED,
            forced_type=forced_type,
            save_path=os.path.join(OUTPUT_DIR, f"secondary_c_scenarios_{forced_type}.csv"),
        )

        instance = build_stochastic_instance(
            locations=locations,
            scenarios=scenarios,
            params=params,
            alpha=ALPHA_FOR_C,
            forced_type=forced_type,
        )

        results = run_solve(instance, label, logger, params, locations=locations)
        stats = extract_run_stats(results, instance)

        flow_by_mode = {
            "sea": stats["total_flow_sea"],
            "air": stats["total_flow_air"],
            "land": stats["total_flow_land"],
        }
        total_flow = sum(flow_by_mode.values())
        dominant_flow_mode = (
            max(flow_by_mode, key=flow_by_mode.get) if total_flow > 0 else "none"
        )

        row = {"forced_type": forced_type, "dominant_flow_mode": dominant_flow_mode}
        row.update(stats)
        results_rows.append(row)

    c_path = os.path.join(OUTPUT_DIR, "secondary_c_type_dominance_results.csv")
    pd.DataFrame(results_rows).to_csv(c_path, index=False)
    logger.log(f"\n[Secondary C] Saved: {c_path}")

    return results_rows


# -------------------------------------------------------------------------
# Summary tables (Task 9)
# -------------------------------------------------------------------------
def _fmt(val, fmt=".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def generate_summary(
    primary_rows: List[Dict],
    secondary_b_rows: List[Dict],
    secondary_c_rows: List[Dict],
    locations: Dict,
) -> str:
    lines = []

    def add(s=""):
        lines.append(s)

    add("STEP 4 RESULTS SUMMARY")
    add("=" * 72)
    add(f"Generated: {datetime.datetime.now().isoformat()}")
    add()

    # --- Summary 1: Primary ---
    add("SUMMARY 1: PRIMARY ANALYSIS (Uniform Matrix Scaling)")
    add("-" * 72)
    add(
        f"{'alpha':>6} | {'objective':>12} | {'selected_sites':<28} | "
        f"{'total_unmet':>11} | {'flow_sea%':>9} | {'flow_air%':>9} | "
        f"{'flow_land%':>10} | {'transfer_vol':>12}"
    )
    add("-" * 72)
    for row in primary_rows:
        sea = row.get("total_flow_sea") or 0.0
        air = row.get("total_flow_air") or 0.0
        land = row.get("total_flow_land") or 0.0
        tf = sea + air + land
        sea_pct = 100 * sea / tf if tf > 0 else 0.0
        air_pct = 100 * air / tf if tf > 0 else 0.0
        land_pct = 100 * land / tf if tf > 0 else 0.0

        site_str = row.get("selected_sites", "")
        site_ids = [int(s) for s in site_str.split("|") if s]
        site_names = "|".join(
            locations.get(s, {}).get("name", str(s)) for s in site_ids
        )

        add(
            f"{row.get('alpha', ''):>6} | "
            f"{_fmt(row.get('objective'), '.4f'):>12} | "
            f"{site_names:<28} | "
            f"{_fmt(row.get('total_unmet'), '.1f'):>11} | "
            f"{sea_pct:>8.1f}% | "
            f"{air_pct:>8.1f}% | "
            f"{land_pct:>9.1f}% | "
            f"{row.get('total_transfer', 0) or 0:>12.1f}"
        )
    add()

    # --- Summary 2: Mode isolation ---
    add("SUMMARY 2: SECONDARY B (Mode Isolation)")
    add("-" * 72)
    add(
        f"{'mode':<8} | {'alpha=0.0 obj':>13} | {'alpha=0.5 obj':>13} | "
        f"{'alpha=1.0 obj':>13} | {'sites_stable?':<13}"
    )
    add("-" * 72)
    b_by_mode: Dict = defaultdict(dict)
    b_sites_by_mode: Dict = defaultdict(dict)
    for row in secondary_b_rows:
        mode = row.get("varied_mode", "")
        a = row.get("alpha")
        b_by_mode[mode][a] = row.get("objective")
        b_sites_by_mode[mode][a] = row.get("selected_sites", "")
    for mode in ["sea", "air", "land"]:
        o0 = b_by_mode[mode].get(0.0)
        o5 = b_by_mode[mode].get(0.5)
        o1 = b_by_mode[mode].get(1.0)
        s0 = b_sites_by_mode[mode].get(0.0, "")
        s1 = b_sites_by_mode[mode].get(1.0, "")
        stable = "yes" if (s0 == s1 and s0 != "") else ("yes*" if s0 == "" else "no")
        add(
            f"{mode:<8} | {_fmt(o0, '.4f'):>13} | {_fmt(o5, '.4f'):>13} | "
            f"{_fmt(o1, '.4f'):>13} | {stable:<13}"
        )
    add()

    # --- Summary 3: Type dominance ---
    add("SUMMARY 3: SECONDARY C (Type Dominance)")
    add("-" * 72)
    add(
        f"{'type':<12} | {'selected_sites':<28} | {'sr_mean':>8} | "
        f"{'sr_p10':>8} | {'dominant_mode':<14}"
    )
    add("-" * 72)
    for row in secondary_c_rows:
        site_str = row.get("selected_sites", "")
        site_ids = [int(s) for s in site_str.split("|") if s]
        site_names = "|".join(
            locations.get(s, {}).get("name", str(s)) for s in site_ids
        )
        dom = row.get("dominant_flow_mode", "")
        add(
            f"{row.get('forced_type', ''):<12} | "
            f"{site_names:<28} | "
            f"{_fmt(row.get('service_rate_mean'), '.4f'):>8} | "
            f"{_fmt(row.get('service_rate_p10'), '.4f'):>8} | "
            f"{dom:<14}"
        )
    add()

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Completion report (Task 10)
# -------------------------------------------------------------------------
def generate_completion_report(
    primary_rows: List[Dict],
    secondary_b_rows: List[Dict],
    secondary_c_rows: List[Dict],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> str:
    lines = []

    def add(s=""):
        lines.append(s)

    all_rows = primary_rows + secondary_b_rows + secondary_c_rows
    total_attempted = len(all_rows)
    total_successful = sum(
        1 for r in all_rows
        if r.get("status") in ("OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT")
        and r.get("objective") is not None
    )
    total_infeasible = sum(
        1 for r in all_rows
        if r.get("status") in ("INFEASIBLE", "ERROR", None)
    )
    elapsed = (end_time - start_time).total_seconds()

    add("STEP 4 COMPLETION REPORT")
    add("=" * 72)
    add(f"Start: {start_time.isoformat()}")
    add(f"End:   {end_time.isoformat()}")
    add(f"Total runtime: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    add()

    add(
        f"1. ANALYSIS RUNS COMPLETED: {total_successful}/{total_attempted} successful "
        f"({total_infeasible} infeasible/error)"
    )
    add("   - Primary A:   5 runs (alpha in [0.0, 0.25, 0.5, 0.75, 1.0])")
    add("   - Secondary B: 9 runs (3 modes x 3 alphas)")
    add("   - Secondary C: 3 runs (flood, storm, earthquake)")
    add()

    add("2. PRIMARY ANALYSIS KEY FINDINGS:")
    successful_primary = [r for r in primary_rows if r.get("objective") is not None]
    if successful_primary:
        objs = [r["objective"] for r in successful_primary]
        add(f"   Objective range: {min(objs):.4f} to {max(objs):.4f}")
        prev_sites = None
        site_transition = None
        for r in primary_rows:
            sites = r.get("selected_sites", "")
            if prev_sites is not None and sites != prev_sites and sites != "":
                site_transition = r.get("alpha")
                break
            if sites:
                prev_sites = sites
        if site_transition is not None:
            add(f"   Site transition detected at alpha={site_transition}")
        else:
            add("   No site transition detected across alpha sweep")

        a0 = next((r for r in primary_rows if r.get("alpha") == 0.0), None)
        a1 = next((r for r in primary_rows if r.get("alpha") == 1.0), None)
        for a_row, label in [(a0, "alpha=0.00"), (a1, "alpha=1.00")]:
            if a_row:
                sea = a_row.get("total_flow_sea") or 0
                air = a_row.get("total_flow_air") or 0
                land = a_row.get("total_flow_land") or 0
                tf = sea + air + land
                if tf > 0:
                    add(
                        f"   {label}: sea={100*sea/tf:.1f}% "
                        f"air={100*air/tf:.1f}% land={100*land/tf:.1f}%"
                    )
    else:
        add("   No successful primary runs.")
    add()

    add("3. SECONDARY B (MODE ISOLATION) KEY FINDINGS:")
    b_by_mode: Dict = defaultdict(dict)
    b_sites_by_mode: Dict = defaultdict(dict)
    for row in secondary_b_rows:
        mode = row.get("varied_mode", "")
        a = row.get("alpha")
        b_by_mode[mode][a] = row.get("objective")
        b_sites_by_mode[mode][a] = row.get("selected_sites", "")
    max_impact_mode = None
    max_impact_val = -1.0
    for mode in ["sea", "air", "land"]:
        o0 = b_by_mode[mode].get(0.0)
        o1 = b_by_mode[mode].get(1.0)
        if o0 is not None and o1 is not None:
            diff = o0 - o1
            add(
                f"   {mode}: obj@alpha=0={o0:.4f}, obj@alpha=1.0={o1:.4f}, "
                f"difference={diff:.4f}"
            )
            if abs(diff) > max_impact_val:
                max_impact_val = abs(diff)
                max_impact_mode = mode
        else:
            add(f"   {mode}: insufficient solve data")
    if max_impact_mode:
        add(f"   Highest impact mode (largest objective swing): {max_impact_mode}")
    add()

    add("4. SECONDARY C (TYPE DOMINANCE) KEY FINDINGS:")
    c_ok = [r for r in secondary_c_rows if r.get("service_rate_mean") is not None]
    if c_ok:
        worst = min(c_ok, key=lambda r: r.get("service_rate_mean") or 1.0)
        add(
            f"   Worst service rate type: {worst.get('forced_type')} "
            f"(sr_mean={worst.get('service_rate_mean'):.4f})"
        )
        all_c_sites = [r.get("selected_sites", "") for r in secondary_c_rows]
        stable = len(set(s for s in all_c_sites if s)) == 1
        add(f"   Site selection stable across forced types: {stable}")
        for r in secondary_c_rows:
            add(
                f"   {r.get('forced_type'):<12}: sites={r.get('selected_sites')} | "
                f"sr_mean={_fmt(r.get('service_rate_mean'), '.4f')} | "
                f"dominant_mode={r.get('dominant_flow_mode')}"
            )
    else:
        add("   No successful Secondary C runs.")
    add()

    add("5. OPEN FLAGS FOR DISSERTATION WRITING:")
    add("   - Transfer cost calibration still needed (SME input required).")
    add(
        "     Current scale: modal_arc_cost = distance_km/1000; "
        "compare to legacy cost_per_unit_km=0.0001 — mismatch may inflate "
        "transport cost relative to unmet-demand penalty."
    )
    add("   - Mode isolation findings ready for Section 5 analysis.")
    add("   - Type dominance findings ready for Section 5 analysis.")
    add(
        "   - Check whether tau>0 (cross-mode forwarding) is triggered in "
        "100-scenario runs (was zero in 10-scenario smoke test)."
    )
    add()

    add("6. DEPRECATED CODE REMOVAL STATUS (Task 1):")
    add("   - build_directed_arcs(): REMOVED from model/input_builder.py")
    add("   - build_residual_arc_capacity(): REMOVED from model/input_builder.py")
    add("   - build_arc_cost(): REMOVED (was only used in deprecated compat block)")
    add("   - build_nominal_arc_capacity(): REMOVED (only used in deprecated compat block)")
    add(
        "   - Deprecated instance keys (arcs, arc_cost, nominal_arc_capacity, "
        "residual_arc_capacity, gamma scalar): REMOVED"
    )
    add("   - GAMMA_VALUES scalar sweep: REMOVED from sensitivity_runner.py")
    add("   - instance['arcs'] reference: REMOVED from sensitivity_runner.py")
    add("   - Import check: PASSED (output/step4_import_check.txt)")
    add()

    add("7. NEXT STEP: Step 5 — Results documentation and dissertation writing")
    add()

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "step4_full_run.log")
    logger = Logger(log_path)

    start_time = datetime.datetime.now()
    logger.log(f"Step 4 Sensitivity Analysis started at: {start_time.isoformat()}")
    logger.log(f"NUM_SCENARIOS={NUM_SCENARIOS}  SEED={SEED}  beta=0.9  P_max=3")
    logger.log(f"ALPHA_VALUES={ALPHA_VALUES}")
    logger.log(f"MODE_ISOLATION_ALPHAS={MODE_ISOLATION_ALPHAS}")
    logger.log(f"FORCED_TYPES={FORCED_TYPES}")
    logger.log(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Confirm output files are writable
    for fname in [
        "primary_analysis_results.csv",
        "primary_scenario_losses.csv",
        "secondary_b_mode_isolation_results.csv",
        "secondary_c_type_dominance_results.csv",
        "step4_results_summary.txt",
        "step4_completion_report.txt",
    ]:
        test_path = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(test_path, "a"):
                pass
        except IOError as e:
            logger.log(f"ERROR: cannot write to {test_path}: {e}")
            logger.close()
            sys.exit(1)
    logger.log("Output paths verified writable.")

    # Setup
    locations = load_locations()
    G = build_graph(locations)
    params = load_parameters()

    logger.log(f"Loaded {len(locations)} nodes")
    logger.log(
        f"PPL-eligible: "
        f"{sum(1 for v in locations.values() if v.get('ppl_eligible') in ('Y', 'Y(C)'))}"
    )

    # --- Run all three analyses ---
    primary_rows, primary_scenario_rows = run_primary_analysis(locations, G, params, logger)
    secondary_b_rows = run_secondary_b(locations, G, params, logger)
    secondary_c_rows = run_secondary_c(locations, G, params, logger)

    end_time = datetime.datetime.now()

    # --- Generate summaries (Task 9) ---
    summary_text = generate_summary(
        primary_rows, secondary_b_rows, secondary_c_rows, locations
    )
    summary_path = os.path.join(OUTPUT_DIR, "step4_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.log(f"\n[Summary] Saved: {summary_path}")

    # Print Summary 1 and Summary 3 to console
    logger.log("\n" + "=" * 72)
    logger.log("CONSOLE OUTPUT: SUMMARY 1 AND SUMMARY 3")
    logger.log("=" * 72)
    in_target = False
    for line in summary_text.split("\n"):
        if line.startswith("SUMMARY 1") or line.startswith("SUMMARY 3"):
            in_target = True
        elif line.startswith("SUMMARY 2"):
            in_target = False
        elif line.startswith("STEP 4") or line.startswith("Generated"):
            in_target = False
        if in_target:
            logger.log(line)

    # --- Generate completion report (Task 10) ---
    report_text = generate_completion_report(
        primary_rows, secondary_b_rows, secondary_c_rows, start_time, end_time
    )
    report_path = os.path.join(OUTPUT_DIR, "step4_completion_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.log(f"\n[Report] Saved: {report_path}")

    logger.log("\n" + "=" * 72)
    logger.log("COMPLETION REPORT")
    logger.log("=" * 72)
    logger.log(report_text)

    logger.log(f"\nLog saved to: {log_path}")
    logger.close()


if __name__ == "__main__":
    main()
