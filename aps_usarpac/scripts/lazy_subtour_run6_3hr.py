"""
lazy_subtour_run6_3hr.py

3-hour extension of the beta=0.90 node-rate diagnostic ("Run 6" config,
docs/individual_vehicle_lazy_subtour_report.md Sec 3.5, and its 45-min
repeat in lazy_subtour_run6_repeat.py), justified by that repeat's finding:
node exploration was still accelerating (not plateauing) at the 45-minute
cutoff, so a longer cap is needed to see whether it keeps accelerating,
plateaus, or ever produces an incumbent.

Exact configuration (unchanged from the 45-min repeat):
  - Real 50-node network, N=10 scenarios, seed=32
  - Real fleet: C-17 F_k=12, C-130J F_k=16 (no override)
  - vehicle_formulation="individual", lazy-callback subtour elimination
    active (_debug_skip_departure_single_node default True)
  - _debug_skip_symmetry_break=True   (no VehicleSymmetryBreak)
  - _debug_mip_focus=0                (no MIPFocus override)
  - beta=0.90

Only the TimeLimit changes: 3 hours (10800s) instead of 45 minutes.

New this run:
  - Node-count/best-bound/sol-count logging every 5 minutes throughout the
    full 3 hours (via _debug_node_log_interval_sec).
  - Immediate incumbent-found flag: the moment Gurobi's first MIPSOL fires,
    model.py writes output/_run6_3hr_incumbent_flag.txt directly from inside
    the callback (via _debug_incumbent_flag_path) -- this file's existence
    can be polled independently of this script's own progress, so an
    incumbent is detectable without waiting for the 3-hour cap or the next
    5-minute log line.
  - External memory watchdog (scripts/memory_watchdog.py), launched as a
    separate independent process against this script's PID: samples macOS
    system-wide memory pressure (kern.memorystatus_vm_pressure_level) and
    this process's own RSS every 15s, and sends a graceful SIGTERM (which
    model.py now routes to model.terminate(), not a hard kill) if pressure
    escalates -- same rationale as the earlier 3-hour MIPFocus probe, which
    hit an uncontrolled jetsam kill once on this machine before this
    watchdog pattern was adopted.
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from network.network_builder import load_locations, build_graph
from scenarios.scenario_generator import generate_scenarios
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar

TIME_LIMIT_SEC = 10800.0  # 3 hours
NODE_LOG_INTERVAL_SEC = 300.0  # 5 minutes
N = 10
SEED = 32
BETA = 0.90
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
LOG_PATH = os.path.join(OUTPUT_DIR, "lazy_subtour_run6_3hr_N10_seed32_beta090.log")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "lazy_subtour_run6_3hr_summary.txt")
INCUMBENT_FLAG_PATH = os.path.join(OUTPUT_DIR, "_run6_3hr_incumbent_flag.txt")


def rss_mb(pid: int) -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) / 1024.0


class PeakRSSSampler:
    def __init__(self, pid: int, interval: float = 2.0):
        self.pid = pid
        self.interval = interval
        self.peak_mb = 0.0
        self.samples = []  # [(elapsed_sec, rss_mb), ...]
        self._t0 = time.time()
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                val = rss_mb(self.pid)
                self.peak_mb = max(self.peak_mb, val)
                self.samples.append((time.time() - self._t0, val))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        try:
            self.peak_mb = rss_mb(self.pid)
        except Exception:
            self.peak_mb = 0.0
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(INCUMBENT_FLAG_PATH):
        os.remove(INCUMBENT_FLAG_PATH)

    print("=" * 70)
    print(f"RUN 6, 3-HOUR EXTENSION: N={N}, seed={SEED}, beta={BETA}, "
          f"real fleet (C-17=12, C-130J=16)")
    print("lazy callback ON, symmetry-break OFF, MIPFocus override OFF, 3-hour cap")
    print(f"Node-count logging every {NODE_LOG_INTERVAL_SEC:.0f}s; "
          f"incumbent flag file: {INCUMBENT_FLAG_PATH}")
    print("=" * 70)
    sys.stdout.flush()

    params = dict(load_parameters())
    params["beta"] = BETA
    locations = load_locations()
    G = build_graph(locations)
    scenarios = generate_scenarios(G, locations, num_scenarios=N, seed=SEED)
    instance = build_stochastic_instance(locations=locations, scenarios=scenarios, params=params)

    print(f"Instance: nodes={len(instance['nodes'])}, ppl_nodes={len(instance['ppl_nodes'])}, "
          f"scenarios={len(instance['scenarios'])}, air_arcs={len(instance['modal_arcs']['air'])}")
    for k in ("C-17", "C-130J"):
        vt = instance["vehicle_types"][k]
        print(f"  {k}: fleet_size={vt['fleet_size']}")
    sys.stdout.flush()

    stdout_fd = 1
    saved_stdout_fd = os.dup(stdout_fd)
    log_file = open(LOG_PATH, "w")
    os.dup2(log_file.fileno(), stdout_fd)

    t0 = time.time()
    try:
        with PeakRSSSampler(os.getpid(), interval=5.0) as sampler:
            results = solve_stochastic_cvar(
                instance,
                time_limit=TIME_LIMIT_SEC,
                mip_gap=params["mip_gap"],
                verbose=True,
                vehicle_formulation="individual",
                _debug_skip_symmetry_break=True,
                _debug_mip_focus=0,
                _debug_node_log_interval_sec=NODE_LOG_INTERVAL_SEC,
                _debug_incumbent_flag_path=INCUMBENT_FLAG_PATH,
            )
        elapsed = time.time() - t0
        peak_rss_mb = sampler.peak_mb
        rss_samples = sampler.samples
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)
        log_file.close()

    model = results.pop("model", None)
    mip_gap_final = model.MIPGap if model is not None and model.SolCount > 0 else None
    sol_count = model.SolCount if model is not None else None
    best_bound = model.ObjBound if model is not None else None
    node_count = model.NodeCount if model is not None else None
    gurobi_status = model.Status if model is not None else None
    if model is not None:
        model.dispose()
    results.pop("variables", None)

    callback_stats = results.get("subtour_callback_stats", {})
    node_log = callback_stats.get("node_log", [])

    exceeded_budget = elapsed > TIME_LIMIT_SEC * 1.05

    incumbent_info = None
    if os.path.exists(INCUMBENT_FLAG_PATH):
        with open(INCUMBENT_FLAG_PATH) as f:
            incumbent_info = f.read()

    with open(SUMMARY_PATH, "w") as sf:
        def out(line=""):
            print(line)
            sf.write(line + "\n")

        out("=" * 70)
        out("RUN 6, 3-HOUR EXTENSION -- RESULT SUMMARY")
        out("=" * 70)
        out(f"Status: {results['status']} (Gurobi status code {gurobi_status})  |  SolCount: {sol_count}")
        out(f"Objective: {results['objective_value']}")
        out(f"Wall-clock solve time: {elapsed:.1f}s ({elapsed/60:.1f} min, {elapsed/3600:.2f} hr)"
            + ("  ** EXCEEDS 3-HOUR BUDGET, FLAG FOR REVIEW **" if exceeded_budget else ""))
        out(f"Peak RSS: {peak_rss_mb:.0f} MB")
        out(f"Final MIP gap: {mip_gap_final}")
        out(f"Best bound: {best_bound}  |  Final NodeCount: {node_count}")
        out(f"Lazy constraints (cbLazy calls) added: {callback_stats.get('cuts_added')}")
        out(f"Callback (MIPSOL) invocations: {callback_stats.get('invocations')}")
        out("")
        if incumbent_info:
            out("*** INCUMBENT WAS FOUND during this run ***")
            out(incumbent_info)
        else:
            out("No incumbent found at any point during this run (flag file never written).")
        out("")
        out("-" * 70)
        out(f"NODE-EXPLORATION LOG (every ~{NODE_LOG_INTERVAL_SEC:.0f}s)")
        out("-" * 70)
        out(f"{'elapsed_sec':>12} {'elapsed_min':>12} {'node_count':>12} {'best_bound':>16} {'sol_count':>10} {'rate(nodes/min)':>18}")
        prev_t, prev_n = None, None
        for row in node_log:
            t = row["elapsed_sec"]
            n = row["node_count"]
            rate_str = ""
            if prev_t is not None and t > prev_t:
                rate = (n - prev_n) / ((t - prev_t) / 60.0)
                rate_str = f"{rate:.1f}"
            out(f"{t:12.1f} {t/60:12.2f} {n:12.0f} {row['best_bound']:16.6g} {row['sol_count']:10.0f} {rate_str:>18}")
            prev_t, prev_n = t, n
        out("")
        out("-" * 70)
        out("RSS SAMPLES (5-min buckets)")
        out("-" * 70)
        bucket = 300.0
        bucketed = {}
        for (t, val) in rss_samples:
            b = int(t // bucket) * bucket
            bucketed.setdefault(b, []).append(val)
        for b in sorted(bucketed):
            vals = bucketed[b]
            out(f"  t={b/60:.0f}-{(b+bucket)/60:.0f} min: mean={sum(vals)/len(vals):.0f} MB, "
                f"max={max(vals):.0f} MB, n_samples={len(vals)}")
        out("")
        out(f"Full Gurobi log: {LOG_PATH}")

    print("\nDONE. Summary written to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
