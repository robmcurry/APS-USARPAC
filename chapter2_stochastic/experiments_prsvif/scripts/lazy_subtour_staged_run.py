"""
lazy_subtour_staged_run.py

Staged benchmark of the "individual" vehicle_formulation (air-mode
per-vehicle-instance indexing, C-17 F_k=12 + C-130J F_k=16 = 28 instances)
on the REAL 50-node network, now that subtour elimination is enforced by a
Gurobi lazy-constraint callback (see model/model.py:_build_subtour_callback)
instead of the static DepartureSingleNode/DepartureNodeLink constraints.

Three steps, run one at a time (see run_step calls at the bottom -- comment/
uncomment as each prior step is confirmed to complete in a reasonable time):
  Step A: N=10,  seed=32, beta=0.5  -- DIAGNOSTIC CONFIGURATION. beta=0.5
          gives a CVaR tail of 5 scenarios (vs. 1 at the real beta=0.90),
          intentionally overridden here for genuine routing pressure across
          half the instance. NOT comparable to locked beta=0.90 results.
  Step B: N=25,  seed=32, beta=0.90 (real/locked value)
  Step C: N=50,  seed=32, beta=0.90

Each step is capped at a 45-minute (2700s) Gurobi TimeLimit. After each run
this script reports: total lazy constraints added, callback invocations,
wall-clock solve time, peak RSS (MB), final MIP gap, and an INDEPENDENT
post-solve path-reachability check (written fresh here, not reusing
_build_subtour_callback's internal logic) confirming the solution is
genuinely subtour-free.

Run: python scripts/lazy_subtour_staged_run.py --step A   (from aps_usarpac/)
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_parameters
from network.network_builder import load_locations, build_graph
from scenarios.scenario_generator import generate_scenarios
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar, _assign_individual_homes

TIME_LIMIT_SEC = 2700.0  # 45 minutes
AIR_INDIV_TYPES = ("C-17", "C-130J")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

STEPS = {
    "A": dict(num_scenarios=10, seed=32, beta=0.5, diagnostic=True),
    "B": dict(num_scenarios=25, seed=32, beta=0.90, diagnostic=False),
    "C": dict(num_scenarios=50, seed=32, beta=0.90, diagnostic=False),
}


def rss_mb(pid: int) -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) / 1024.0


class PeakRSSSampler:
    def __init__(self, pid: int, interval: float = 2.0):
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


def independent_subtour_check(results, instance):
    """
    Fresh, standalone verification -- deliberately NOT calling into
    _build_subtour_callback's internals -- that the final solution has no
    disconnected (non-home-reachable) n_ind arc components. Same reachability
    method as the callback (BFS forward from home over selected arcs), but
    written independently against the *final* results dict, operating after
    the solve is fully done rather than on an in-progress incumbent.

    Returns (phantom_count, total_instances_with_movement, detail_rows).
    """
    flows = results.get("vehicle_flows_individual", {})
    arcs_by_wkl = defaultdict(list)
    for (w, k, l, m, i, j) in flows:
        arcs_by_wkl[(w, k, l)].append((i, j))

    home_by_k = {
        k: _assign_individual_homes(
            instance["vehicle_types"][k]["b_kj"], instance["vehicle_types"][k]["fleet_size"]
        )
        for k in AIR_INDIV_TYPES
        if k in instance.get("vehicle_types", {})
    }

    phantom_count = 0
    detail_rows = []
    for (w, k, l), arcs in arcs_by_wkl.items():
        home = home_by_k[k][l]
        reached = {home}
        changed = True
        while changed:
            changed = False
            for (i, j) in arcs:
                if i in reached and j not in reached:
                    reached.add(j)
                    changed = True
        touched = {node for arc in arcs for node in arc}
        phantom_nodes = touched - reached
        if phantom_nodes:
            phantom_count += 1
            detail_rows.append((w, k, l, home, sorted(arcs), sorted(phantom_nodes)))

    return phantom_count, len(arcs_by_wkl), detail_rows


def run_step(step_label: str, fleet_override: dict = None, label_suffix: str = ""):
    """
    fleet_override: optional {vehicle_type_name: fleet_size} dict, applied to
        params["vehicles"][...]["fleet_size"] before instance build. Used for
        the F_k=1 root-relaxation isolation diagnostic (fleet-size-dependence
        vs. general individual-indexing-on-real-network tractability) --
        NOT part of the normal Step A/B/C progression.
    label_suffix: appended to the log filename so diagnostic runs don't
        collide with the real step's log file.
    """
    cfg = STEPS[step_label]
    N = cfg["num_scenarios"]
    seed = cfg["seed"]
    beta = cfg["beta"]
    diagnostic = cfg["diagnostic"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"STEP {step_label}{label_suffix}: N={N}, seed={seed}, beta={beta}"
          + ("  ** DIAGNOSTIC CONFIGURATION, NOT COMPARABLE TO LOCKED beta=0.90 RESULTS **" if diagnostic else "")
          + (f"  ** FLEET OVERRIDE: {fleet_override} **" if fleet_override else ""))
    print("=" * 70)

    params = dict(load_parameters())
    params["beta"] = beta
    if fleet_override:
        import copy
        params["vehicles"] = copy.deepcopy(params["vehicles"])
        for vtype, fk in fleet_override.items():
            params["vehicles"][vtype]["fleet_size"] = fk
    locations = load_locations()
    G = build_graph(locations)
    scenarios = generate_scenarios(G, locations, num_scenarios=N, seed=seed)
    instance = build_stochastic_instance(locations=locations, scenarios=scenarios, params=params)

    print(f"Instance: nodes={len(instance['nodes'])}, ppl_nodes={len(instance['ppl_nodes'])}, "
          f"scenarios={len(instance['scenarios'])}, air_arcs={len(instance['modal_arcs']['air'])}")
    for k in AIR_INDIV_TYPES:
        vt = instance["vehicle_types"][k]
        print(f"  {k}: fleet_size={vt['fleet_size']}")

    log_path = os.path.join(OUTPUT_DIR, f"lazy_subtour_step{step_label}{label_suffix}_N{N}_seed{seed}.log")
    stdout_fd = 1
    saved_stdout_fd = os.dup(stdout_fd)
    log_file = open(log_path, "w")
    os.dup2(log_file.fileno(), stdout_fd)

    t0 = time.time()
    try:
        with PeakRSSSampler(os.getpid()) as sampler:
            results = solve_stochastic_cvar(
                instance, time_limit=TIME_LIMIT_SEC, mip_gap=params["mip_gap"], verbose=True,
                vehicle_formulation="individual",
            )
        elapsed = time.time() - t0
        peak_rss_mb = sampler.peak_mb
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
    phantom_count, n_instances, phantom_detail = independent_subtour_check(results, instance)

    exceeded_budget = elapsed > TIME_LIMIT_SEC * 1.05  # small buffer over the Gurobi-enforced cap

    print("\n" + "=" * 70)
    print(f"STEP {step_label} RESULT SUMMARY")
    print("=" * 70)
    print(f"Status: {results['status']}  |  SolCount: {sol_count}")
    print(f"Objective: {results['objective_value']}")
    print(f"Wall-clock solve time: {elapsed:.1f}s ({elapsed/60:.1f} min)"
          + ("  ** EXCEEDS 45-MIN BUDGET, FLAG FOR REVIEW **" if exceeded_budget else ""))
    print(f"Peak RSS: {peak_rss_mb:.0f} MB")
    print(f"Final MIP gap: {mip_gap_final}")
    print(f"Best bound: {best_bound}  |  NodeCount (branch-and-bound nodes explored): {node_count}")
    print(f"Lazy constraints (cbLazy calls) added: {callback_stats.get('cuts_added')}")
    print(f"Callback (MIPSOL) invocations: {callback_stats.get('invocations')}")
    print(f"Independent post-solve subtour check: {phantom_count} phantom / {n_instances} vehicle-instances with movement")
    if phantom_detail:
        print("  ** PHANTOM DETAIL (should be empty) **")
        for row in phantom_detail:
            print(f"    {row}")
    if diagnostic:
        print("\n** Step A used beta=0.5 (diagnostic override) -- its subtour/iteration")
        print("   counts are a LOWER BOUND on what beta=0.90 (Steps B/C) will show, not a direct estimate. **")
    print(f"\nFull Gurobi log: {log_path}")

    return {
        "step": step_label, "N": N, "seed": seed, "beta": beta, "diagnostic": diagnostic,
        "status": results["status"], "sol_count": sol_count,
        "objective_value": results["objective_value"],
        "elapsed_sec": elapsed, "exceeded_budget": exceeded_budget,
        "peak_rss_mb": peak_rss_mb, "mip_gap_final": mip_gap_final,
        "best_bound": best_bound, "node_count": node_count,
        "lazy_cuts_added": callback_stats.get("cuts_added"),
        "callback_invocations": callback_stats.get("invocations"),
        "phantom_count": phantom_count, "n_instances_with_movement": n_instances,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["A", "B", "C"], required=True)
    parser.add_argument(
        "--fk1-diagnostic", action="store_true",
        help="Root-relaxation isolation diagnostic: same step config, but "
             "C-17/C-130J fleet_size overridden to 1 (the toy-tested "
             "minimum), to distinguish a fleet-size-dependent tractability "
             "wall from a general individual-indexing-on-real-network one.",
    )
    args = parser.parse_args()
    if args.fk1_diagnostic:
        run_step(args.step, fleet_override={"C-17": 1, "C-130J": 1}, label_suffix="_fk1diag")
    else:
        run_step(args.step)
