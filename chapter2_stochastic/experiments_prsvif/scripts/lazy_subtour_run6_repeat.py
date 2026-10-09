"""
lazy_subtour_run6_repeat.py

Relaunch of "Run 6" from docs/individual_vehicle_lazy_subtour_report.md §3.5
(beta=0.90 node-rate diagnostic), from a fresh terminal session -- the
original Run 6 was launched from a PyCharm terminal that has since been
closed.

Exact Run 6 configuration:
  - Real 50-node network, N=10 scenarios, seed=32
  - Real fleet: C-17 F_k=12, C-130J F_k=16 (no override)
  - vehicle_formulation="individual", lazy-callback subtour elimination
    active (_debug_skip_departure_single_node default True)
  - _debug_skip_symmetry_break=True   (no VehicleSymmetryBreak -- matches
    Run 1's plain-baseline config, per report §3.5)
  - _debug_mip_focus=0                (no MIPFocus override -- Gurobi
    default balanced focus, per report §3.5)
  - beta=0.90 (the one change vs. Run 1)
  - 45-minute (2700s) TimeLimit

New this run (not in the original Run 6): periodic node-exploration logging
every 5 minutes via the new _debug_node_log_interval_sec hook in
_build_subtour_callback (model/model.py), so the branching rate over the
full 45 minutes can be inspected directly rather than only the final node
count. Peak-RSS monitoring continues throughout via the same
PeakRSSSampler pattern used in every prior staged run.
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

TIME_LIMIT_SEC = 2700.0  # 45 minutes
NODE_LOG_INTERVAL_SEC = 300.0  # 5 minutes
N = 10
SEED = 32
BETA = 0.90
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
LOG_PATH = os.path.join(OUTPUT_DIR, "lazy_subtour_run6_repeat_N10_seed32_beta090.log")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "lazy_subtour_run6_repeat_summary.txt")


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

    print("=" * 70)
    print(f"RUN 6 REPEAT: N={N}, seed={SEED}, beta={BETA}, real fleet (C-17=12, C-130J=16)")
    print("lazy callback ON, symmetry-break OFF, MIPFocus override OFF, 45-min cap")
    print(f"Node-count logging every {NODE_LOG_INTERVAL_SEC:.0f}s (new this run)")
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
        with PeakRSSSampler(os.getpid(), interval=2.0) as sampler:
            results = solve_stochastic_cvar(
                instance,
                time_limit=TIME_LIMIT_SEC,
                mip_gap=params["mip_gap"],
                verbose=True,
                vehicle_formulation="individual",
                _debug_skip_symmetry_break=True,
                _debug_mip_focus=0,
                _debug_node_log_interval_sec=NODE_LOG_INTERVAL_SEC,
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
    if model is not None:
        model.dispose()
    results.pop("variables", None)

    callback_stats = results.get("subtour_callback_stats", {})
    node_log = callback_stats.get("node_log", [])

    exceeded_budget = elapsed > TIME_LIMIT_SEC * 1.05

    with open(SUMMARY_PATH, "w") as sf:
        def out(line=""):
            print(line)
            sf.write(line + "\n")

        out("=" * 70)
        out("RUN 6 REPEAT -- RESULT SUMMARY")
        out("=" * 70)
        out(f"Status: {results['status']}  |  SolCount: {sol_count}")
        out(f"Objective: {results['objective_value']}")
        out(f"Wall-clock solve time: {elapsed:.1f}s ({elapsed/60:.1f} min)"
            + ("  ** EXCEEDS 45-MIN BUDGET, FLAG FOR REVIEW **" if exceeded_budget else ""))
        out(f"Peak RSS: {peak_rss_mb:.0f} MB")
        out(f"Final MIP gap: {mip_gap_final}")
        out(f"Best bound: {best_bound}  |  Final NodeCount: {node_count}")
        out(f"Lazy constraints (cbLazy calls) added: {callback_stats.get('cuts_added')}")
        out(f"Callback (MIPSOL) invocations: {callback_stats.get('invocations')}")
        out("")
        out("-" * 70)
        out(f"NODE-EXPLORATION LOG (every ~{NODE_LOG_INTERVAL_SEC:.0f}s)")
        out("-" * 70)
        out(f"{'elapsed_sec':>12} {'elapsed_min':>12} {'node_count':>12} {'best_bound':>16} {'sol_count':>10}")
        prev_t, prev_n = None, None
        for row in node_log:
            t = row["elapsed_sec"]
            n = row["node_count"]
            rate_str = ""
            if prev_t is not None and t > prev_t:
                rate = (n - prev_n) / ((t - prev_t) / 60.0)
                rate_str = f"  ({rate:.1f} nodes/min since prior sample)"
            out(f"{t:12.1f} {t/60:12.2f} {n:12.0f} {row['best_bound']:16.6g} {row['sol_count']:10.0f}{rate_str}")
            prev_t, prev_n = t, n
        out("")
        out("-" * 70)
        out("RSS SAMPLES (every ~2s, summarized to 5-min buckets)")
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
