#!/usr/bin/env python3
"""
run_pipeline_report.py

Standalone, parameterized end-to-end pipeline runner + report generator for the
aps_usarpac stochastic CVaR model. Built as a standing tool: run it directly
from a normal terminal, no Claude Code involvement needed.

    python scripts/run_pipeline_report.py --alpha 0.25

alpha is REQUIRED (no default) so a stale value can never be silently reused.
N and seed come from config/model_parameters.yaml; beta and mip_gap are read
from config and reported as-is (never hardcoded, no CLI override) so the report
is always self-describing regardless of what config state exists at run time.

What it does, in order:
  1. Load config, print the exact parameters that will be used.
  2. Build network -> scenarios -> instance, printing a progress line per stage
     plus a certificate-style variable/constraint count dump.
  3. Solve via the existing solve_stochastic_cvar() (inherits Method=2 and every
     other model setting; NO solve logic is reimplemented here), while a
     background thread samples peak RSS.
  4. While the model is still alive: generate the vehicle itinerary (reuses
     model/vehicle_itinerary.generate_itinerary_report, the clean format) and
     run constraint verification, then dispose the model (same discipline as the
     M1 memory fix in sensitivity_runner.run_solve).
  5. Write a timestamped markdown report to output/ and print a final summary.

Design notes (call these out because they depart from a naive reading of the
brief, and were chosen deliberately):
  * This calls solve_stochastic_cvar() DIRECTLY rather than routing through
    sensitivity_runner.run_solve(). run_solve() applies the M1 fix by disposing
    the Gurobi model and popping results["variables"] before it returns -- but
    the itinerary generator and the constraint-verification pass both need the
    LIVE Gurobi variables (results["variables"][...].X). So this script keeps
    the model alive for that work and then disposes it itself, applying the same
    M1 dispose discipline. Method=2 is set inside solve_stochastic_cvar(), so it
    is inherited automatically regardless of the wrapper.
  * There is no pre-existing reusable constraint-verification function in the
    codebase (only toy_vehicle_test.verify_vehicle_constraints, which is
    toy-instance specific, prints instead of returning, and omits
    transfer-backing). So verify_constraints() below is written here. Its
    headline number is Gurobi's own model.MaxVio (the authoritative max
    constraint violation across the whole model, covering every family incl.
    DistanceBudget); the per-family numbers are independent recomputation
    cross-checks layered on top.
"""

import argparse
import datetime
import os
import subprocess
import sys
import threading
import time
import traceback

# --- make the aps_usarpac package importable regardless of CWD ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)  # .../aps_usarpac
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from model.vehicle_itinerary import generate_itinerary_report
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios

# Combined per-vehicle-arc epsilon cost from model.model (EPSILON_DEPLOY +
# EPSILON_EMPTY). Kept in sync manually -- if those change in model.py, change
# here too. Used only for the loss-decomposition breakdown, not the solve.
VEHICLE_DEPLOY_COST = 0.04

OUTPUT_DIR = os.path.join(_PKG_ROOT, "output")


# =========================================================================
# Small helpers
# =========================================================================
def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def rss_mb(pid: int) -> float:
    """Current RSS of pid in MB, via ps (same method as the M1 verification)."""
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) / 1024.0


class PeakRSSSampler:
    """Background thread that polls RSS and tracks the peak while active."""

    def __init__(self, pid: int, interval: float = 1.0):
        self.pid = pid
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, rss_mb(self.pid))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        try:
            self.peak_mb = rss_mb(self.pid)
        except Exception:
            self.peak_mb = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


class StageTracker:
    """Records each stage's outcome so a partial report can be written on crash."""

    def __init__(self):
        self.stages = []  # list of (name, status, detail)

    def ok(self, name, detail=""):
        self.stages.append((name, "OK", detail))
        print(f"[stage] {name}: OK  {detail}".rstrip(), flush=True)

    def fail(self, name, detail=""):
        self.stages.append((name, "FAILED", detail))
        print(f"[stage] {name}: FAILED  {detail}".rstrip(), flush=True)


# =========================================================================
# Service-rate distribution
# =========================================================================
def service_rate_distribution(results, instance):
    """Full SR distribution from extracted results (no live model needed)."""
    demand = instance["demand"]
    unmet = results.get("unmet_demand", {})
    Omega = instance["scenarios"]
    srs = []
    for w in Omega:
        dem_w = sum(v for (ww, i, r), v in demand.items() if ww == w)
        unmet_w = sum(v for (ww, _, _), v in unmet.items() if ww == w)
        if dem_w > 0:
            srs.append(max(0.0, 1.0 - unmet_w / dem_w))
    srs.sort()
    n = len(srs)

    def pct(p):
        if n == 0:
            return None
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return srs[idx]

    return {
        "n": n,
        "mean": sum(srs) / n if n else None,
        "min": srs[0] if n else None,
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "max": srs[-1] if n else None,
        "n_zero": sum(1 for s in srs if s <= 1e-9),
        "n_full": sum(1 for s in srs if s >= 1.0 - 1e-9),
    }


# =========================================================================
# 4-way loss decomposition (from extracted positive-value dicts)
# =========================================================================
def loss_decomposition(results, instance):
    penalty = instance["penalty"]
    modal_arc_cost = instance["modal_arc_cost"]
    transfer_cost = instance["transfer_cost"]

    unmet = sum(
        penalty[(j, r)] * v for (w, j, r), v in results.get("unmet_demand", {}).items()
    )
    transport = sum(
        modal_arc_cost[m].get((i, j), 0.0) * v
        for (w, m, i, j, r), v in results.get("flows", {}).items()
    )
    transfer = sum(
        transfer_cost.get((m1, m2), 0.0) * v
        for (w, i, m1, m2, r), v in results.get("tau", {}).items()
    )
    vehicle = sum(
        VEHICLE_DEPLOY_COST * cnt for cnt in results.get("vehicle_flows", {}).values()
    )
    total = unmet + transport + transfer + vehicle

    def share(x):
        return (100.0 * x / total) if total > 0 else 0.0

    return {
        "unmet": unmet, "transport": transport, "transfer": transfer,
        "vehicle": vehicle, "total": total,
        "unmet_pct": share(unmet), "transport_pct": share(transport),
        "transfer_pct": share(transfer), "vehicle_pct": share(vehicle),
    }


# =========================================================================
# Constraint verification (headline = Gurobi MaxVio; recomputation cross-checks)
# =========================================================================
def verify_constraints(results, instance, model):
    """Returns a dict of {check_name: (max_violation, passed_bool, note)}."""
    TOL = 1e-4
    N = instance["nodes"]
    R = instance["commodities"]
    Omega = instance["scenarios"]
    modes = instance["modes"]
    PPL = set(instance["ppl_nodes"])
    demand = instance["demand"]
    modal_arcs = instance["modal_arcs"]
    modal_residual = instance["modal_residual"]
    transfer_cap = instance["transfer_cap"]

    flows = results.get("flows", {})       # (w,m,i,j,r) -> val   (>1e-6 only)
    release = results.get("release", {})   # (w,i,r) -> val       (>1e-6 only)
    unmet = results.get("unmet_demand", {})  # (w,j,r) -> val     (>1e-6 only)

    checks = {}

    # Gurobi's own authoritative max constraint violation (covers EVERY family,
    # including DistanceBudget whose turnaround auxiliary isn't in results).
    try:
        maxvio = float(model.MaxVio)
    except Exception:
        maxvio = None
    checks["gurobi_max_violation (all families)"] = (
        maxvio, (maxvio is not None and maxvio < 1e-3),
        "authoritative: Gurobi's max primal constraint violation over the whole model",
    )

    # --- Independent recomputation: flow conservation ---
    # inflow + release + unmet == demand + outflow  for every (w,i,r)
    infl = {}
    outfl = {}
    for (w, m, i, j, r), v in flows.items():
        outfl[(w, i, r)] = outfl.get((w, i, r), 0.0) + v
        infl[(w, j, r)] = infl.get((w, j, r), 0.0) + v
    max_cons = 0.0
    for w in Omega:
        for i in N:
            for r in R:
                lhs = infl.get((w, i, r), 0.0) + release.get((w, i, r), 0.0) + unmet.get((w, i, r), 0.0)
                rhs = demand.get((w, i, r), 0.0) + outfl.get((w, i, r), 0.0)
                max_cons = max(max_cons, abs(lhs - rhs))
    checks["flow_conservation (recomputed)"] = (
        max_cons, max_cons < TOL * max(1.0, 1.0), "max |inflow+release+unmet - demand-outflow|",
    )

    # --- Independent recomputation: arc capacity ---
    max_cap = 0.0
    for (w, m, i, j, r), v in flows.items():
        cap = modal_residual[w].get(m, {}).get((i, j), {}).get(r, 0.0)
        max_cap = max(max_cap, v - cap)
    checks["arc_capacity (recomputed)"] = (
        max_cap, max_cap < TOL, "max (flow - residual_capacity), should be <= 0",
    )

    # --- Independent recomputation: transfer backing ---
    # tau_out on mode m <= inflow on mode m + release  (per w,i,m,r)
    tau = results.get("tau", {})  # (w,i,m1,m2,r) -> val
    infl_m = {}
    for (w, m, i, j, r), v in flows.items():
        infl_m[(w, j, m, r)] = infl_m.get((w, j, m, r), 0.0) + v
    tau_out_m = {}
    for (w, i, m1, m2, r), v in tau.items():
        tau_out_m[(w, i, m1, r)] = tau_out_m.get((w, i, m1, r), 0.0) + v
    max_tb = 0.0
    for (w, i, m1, r), tout in tau_out_m.items():
        backing = infl_m.get((w, i, m1, r), 0.0)
        if i in PPL:
            backing += release.get((w, i, r), 0.0)
        max_tb = max(max_tb, tout - backing)
    checks["transfer_backing (recomputed)"] = (
        max_tb, max_tb < TOL, "max (tau_out_m - inflow_m - release), should be <= 0",
    )

    # DistanceBudget is confirmed via gurobi_max_violation above (its turnaround
    # exemption auxiliary g is not exposed in the results dict, so an independent
    # recomputation would be only a loose bound; MaxVio is the reliable check).
    checks["distance_budget"] = (
        maxvio, (maxvio is not None and maxvio < 1e-3),
        "confirmed via gurobi_max_violation (auxiliary g not exposed for recompute)",
    )

    return checks


# =========================================================================
# Report writer
# =========================================================================
def write_report(path, ctx):
    L = []
    a = L.append
    p = ctx["params_used"]
    a(f"# Pipeline Report — alpha={p['alpha']}")
    a("")
    a(f"**Timestamp:** {ctx['timestamp']}")
    a("")
    a("## Parameters used (sourced from this run, not hardcoded)")
    a("")
    a("| Parameter | Value | Source |")
    a("|---|---|---|")
    a(f"| alpha | {p['alpha']} | CLI argument (required) |")
    a(f"| beta | {p['beta']} | config/model_parameters.yaml |")
    a(f"| mip_gap | {p['mip_gap']} | config/model_parameters.yaml |")
    a(f"| seed | {p['seed']} | config/model_parameters.yaml |")
    a(f"| N (num_scenarios) | {p['num_scenarios']} | config/model_parameters.yaml |")
    a("")

    if ctx.get("sizes"):
        s = ctx["sizes"]
        a("## Instance size")
        a("")
        a("| Quantity | Count |")
        a("|---|---|")
        a(f"| variables (model.NumVars) | {s['num_vars']} |")
        a(f"| constraints (model.NumConstrs) | {s['num_constrs']} |")
        a(f"| \\|N\\| / \\|N^P\\| | {s['N']} / {s['NP']} |")
        a(f"| \\|A_sea\\| / \\|A_air\\| / \\|A_land\\| | {s['A_sea']} / {s['A_air']} / {s['A_land']} |")
        a(f"| \\|Omega\\| | {s['Omega']} |")
        a("")

    if ctx.get("solve"):
        sv = ctx["solve"]
        a("## Solve")
        a("")
        a("| Metric | Value |")
        a("|---|---|")
        a(f"| status | {sv['status']} |")
        a(f"| solve time (Gurobi Runtime) | {sv['runtime']:.1f}s |")
        a(f"| wall time (solve call) | {sv['wall']:.1f}s |")
        a(f"| final MIP gap | {sv['gap']} |")
        a(f"| objective | {sv['objective']} |")
        a(f"| peak RSS during solve (ps-sampled) | {ctx.get('peak_rss_mb', 0):.0f} MB |")
        a(f"| selected sites | {sv['sites_named']} |")
        a("")

    if ctx.get("sr"):
        sr = ctx["sr"]
        a("## Service-rate distribution")
        a("")
        a("| Stat | Value |")
        a("|---|---|")
        for k in ["mean", "p10", "p25", "p50", "p75", "min", "max"]:
            v = sr.get(k)
            a(f"| {k} | {v:.4f} |" if v is not None else f"| {k} | N/A |")
        a(f"| n_zero (SR<=0) | {sr['n_zero']} / {sr['n']} |")
        a(f"| n_full (SR>=1) | {sr['n_full']} / {sr['n']} |")
        a("")

    if ctx.get("loss"):
        ld = ctx["loss"]
        a("## Loss decomposition (4-way)")
        a("")
        a("| Component | Value | Share |")
        a("|---|---|---|")
        a(f"| unmet-demand penalty | {ld['unmet']:.2f} | {ld['unmet_pct']:.2f}% |")
        a(f"| arc transport | {ld['transport']:.2f} | {ld['transport_pct']:.2f}% |")
        a(f"| intermodal transfer | {ld['transfer']:.2f} | {ld['transfer_pct']:.2f}% |")
        a(f"| vehicle deploy (epsilon) | {ld['vehicle']:.2f} | {ld['vehicle_pct']:.2f}% |")
        a(f"| **total** | **{ld['total']:.2f}** | 100% |")
        a("")

    if ctx.get("verify"):
        a("## Constraint verification")
        a("")
        a("| Check | Max violation | Pass | Note |")
        a("|---|---|---|---|")
        for name, (vio, passed, note) in ctx["verify"].items():
            vio_s = f"{vio:.2e}" if isinstance(vio, float) else str(vio)
            a(f"| {name} | {vio_s} | {'PASS' if passed else 'FAIL'} | {note} |")
        a("")

    if ctx.get("itinerary_md"):
        a("## Vehicle itinerary (lowest-3 and highest-3 SR scenarios)")
        a("")
        a(ctx["itinerary_md"])
        a("")

    a("## Stage completion log")
    a("")
    a("| Stage | Status | Detail |")
    a("|---|---|---|")
    for name, status, detail in ctx["tracker"].stages:
        a(f"| {name} | {status} | {detail} |")
    a("")
    if ctx.get("error"):
        a("## Error (partial report)")
        a("")
        a("```")
        a(ctx["error"])
        a("```")
        a("")

    with open(path, "w") as f:
        f.write("\n".join(L))


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline run + report.")
    ap.add_argument("--alpha", type=float, required=True,
                    help="degradation scaling factor (REQUIRED, no default)")
    args = ap.parse_args()

    tracker = StageTracker()
    pid = os.getpid()
    timestamp = now_stamp()
    ctx = {"tracker": tracker, "timestamp": timestamp}
    report_path = os.path.join(OUTPUT_DIR, f"pipeline_report_alpha{args.alpha}_{timestamp}.md")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # ---- config ----
        params = load_parameters()
        params_used = {
            "alpha": args.alpha,
            "beta": params.get("beta"),
            "mip_gap": params.get("mip_gap"),
            "seed": params.get("seed"),
            "num_scenarios": params.get("num_scenarios"),
        }
        ctx["params_used"] = params_used
        print("=" * 60)
        print("PIPELINE REPORT RUN")
        print("=" * 60)
        for k, v in params_used.items():
            print(f"  {k:14s} = {v}")
        print("=" * 60, flush=True)
        tracker.ok("config", f"alpha={args.alpha} beta={params_used['beta']} "
                             f"gap={params_used['mip_gap']} seed={params_used['seed']} "
                             f"N={params_used['num_scenarios']}")

        # ---- network ----
        locations = load_locations()
        G = build_graph(locations)
        tracker.ok("network build", f"{len(locations)} nodes")

        # ---- scenarios ----
        scenarios = generate_scenarios(
            G, locations,
            num_scenarios=params_used["num_scenarios"],
            seed=params_used["seed"],
            save_path=os.path.join(OUTPUT_DIR, f"pipeline_report_scenarios_{timestamp}.csv"),
        )
        tracker.ok("scenario generation", f"{len(scenarios)} scenarios")

        # ---- instance ----
        instance = build_stochastic_instance(
            locations=locations, scenarios=scenarios, params=params, alpha=args.alpha,
        )
        ctx["sizes_pre"] = {
            "N": len(instance["nodes"]), "NP": len(instance["ppl_nodes"]),
            "A_sea": len(instance["modal_arcs"]["sea"]),
            "A_air": len(instance["modal_arcs"]["air"]),
            "A_land": len(instance["modal_arcs"]["land"]),
            "Omega": len(instance["scenarios"]),
        }
        tracker.ok("instance build",
                   f"N={ctx['sizes_pre']['N']} N^P={ctx['sizes_pre']['NP']} "
                   f"arcs(sea/air/land)={ctx['sizes_pre']['A_sea']}/"
                   f"{ctx['sizes_pre']['A_air']}/{ctx['sizes_pre']['A_land']} "
                   f"Omega={ctx['sizes_pre']['Omega']}")

        # ---- solve (peak-RSS sampled) ----
        print("[solve] starting solve_stochastic_cvar (Method=2 inherited from model.py)...", flush=True)
        t0 = time.time()
        with PeakRSSSampler(pid) as sampler:
            results = solve_stochastic_cvar(
                instance,
                time_limit=7200,
                mip_gap=params_used["mip_gap"],
                verbose=True,
                detailed_extraction=True,
            )
        wall = time.time() - t0
        ctx["peak_rss_mb"] = sampler.peak_mb

        model = results.get("model")
        num_vars = model.NumVars if model is not None else None
        num_constrs = model.NumConstrs if model is not None else None
        runtime = model.Runtime if model is not None else wall
        gap = model.MIPGap if model is not None else None
        status = results.get("status")
        objective = results.get("objective_value")
        sites = results.get("selected_sites", [])
        sites_named = [locations[s]["name"] for s in sites]

        ctx["sizes"] = {**ctx["sizes_pre"], "num_vars": num_vars, "num_constrs": num_constrs}
        ctx["solve"] = {
            "status": status, "runtime": runtime, "wall": wall, "gap": gap,
            "objective": objective, "sites": sites, "sites_named": sites_named,
        }
        print(f"[solve] complete: status={status} runtime={runtime:.1f}s "
              f"gap={gap} peak_rss={sampler.peak_mb:.0f}MB", flush=True)
        tracker.ok("solve", f"status={status} runtime={runtime:.1f}s gap={gap}")

        # ---- live-variable work (must precede dispose) ----
        if objective is not None:
            ctx["sr"] = service_rate_distribution(results, instance)
            ctx["loss"] = loss_decomposition(results, instance)
            try:
                ctx["verify"] = verify_constraints(results, instance, model)
                tracker.ok("verification",
                           f"gurobi_max_vio={ctx['verify'].get('gurobi_max_violation (all families)', (None,))[0]}")
            except Exception as e:
                tracker.fail("verification", str(e))
            try:
                ctx["itinerary_md"] = generate_itinerary_report(
                    results=results, instance=instance, locations=locations,
                    label=f"alpha={args.alpha} gap={params_used['mip_gap']}", top_k=3,
                )
                tracker.ok("itinerary")
            except Exception as e:
                tracker.fail("itinerary", str(e))
        else:
            tracker.fail("post-solve analysis", "no feasible objective; skipped SR/loss/itinerary")

        # ---- dispose (M1 discipline) ----
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        results.pop("model", None)
        results.pop("variables", None)
        tracker.ok("model dispose")

    except Exception:
        ctx["error"] = traceback.format_exc()
        tracker.fail("run", "exception; writing partial report")
        print("[ERROR] " + ctx["error"], flush=True)

    # ---- always write a report (full or partial) ----
    ctx.setdefault("params_used", {"alpha": args.alpha, "beta": "?", "mip_gap": "?",
                                   "seed": "?", "num_scenarios": "?"})
    try:
        write_report(report_path, ctx)
        print(f"\n[report] written: {report_path}", flush=True)
        tracker.ok("report generation", report_path)
    except Exception:
        print("[ERROR] failed to write report:\n" + traceback.format_exc(), flush=True)

    # ---- final console summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    sv = ctx.get("solve")
    if sv:
        print(f"  status     : {sv['status']}")
        print(f"  sites      : {sv['sites_named']}")
        sr = ctx.get("sr")
        print(f"  sr_mean    : {sr['mean']:.4f}" if sr and sr['mean'] is not None else "  sr_mean    : N/A")
        print(f"  solve time : {sv['runtime']:.1f}s")
        print(f"  peak RSS   : {ctx.get('peak_rss_mb', 0):.0f} MB")
    else:
        print("  run did not reach a solved state — see partial report.")
    print(f"  report     : {report_path}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
