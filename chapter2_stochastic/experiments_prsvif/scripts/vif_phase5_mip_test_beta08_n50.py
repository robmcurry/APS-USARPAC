"""
vif_phase5_mip_test_beta08_n30.py

diagnostic: MIP solve attempt of PRS-VIF on the actual
50-node network, at any scale. Everything validated so far only confirmed the model

beta=0.8, overridden here, NOT in config/model_parameters.yaml

use a bounded model.Params.TimeLimit

adds three things for live monitoring and a durable record of the run:
  - gurobi's own solve log saved to a timestamped file (in addition to
    printing, doesn't change what gurobi does, just captures it)
  - a background thread that samples this process's memory every 15s,
    prints it, and logs it to csv -- same cadence as the report charts
  - a summary file written at the end with the same fields the earlier
    report used, so the run leaves something behind besides scrollback

Run: python scripts/vif_phase5_mip_test_beta08_n30.py   (from aps_usarpac/)
"""
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gurobipy as gp

from config.loader import load_parameters
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar
from network.network_builder import build_graph, load_locations
from scenarios.scenario_generator import generate_scenarios

NUM_SCENARIOS = 10
BETA_OVERRIDE = 0.6
SEED = 32
TIME_LIMIT_SEC = 60000  # 60 min bounded attempt -- no prior PRS-VIF MIP timing exists
MEM_SAMPLE_INTERVAL_SEC = 15

RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "mip_runs")
os.makedirs(LOG_DIR, exist_ok=True)
GUROBI_LOG_PATH = os.path.join(LOG_DIR, f"gurobi_{RUN_TAG}.log")
MEM_LOG_PATH = os.path.join(LOG_DIR, f"memory_{RUN_TAG}.csv")
SUMMARY_PATH = os.path.join(LOG_DIR, f"summary_{RUN_TAG}.json")


#gp.setParam("Threads", 4)
gp.setParam("NodefileStart", 32.0)
# saves gurobi's full presolve/barrier/crossover log to a file, on top of
# printing it. same content that got hand-copied into the report earlier,
# this time it's just saved automatically
gp.setParam("LogFile", GUROBI_LOG_PATH)


def _current_rss_gb(pid):
    # reads live current rss via ps, not python's resource module (which
    # only gives a high water mark, not a live trace). same method the
    # external shell monitors used earlier this session
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) / 1_000_000  # ps rss is in kb on mac and linux


def _memory_sampler(pid, stop_event, log_path):
    with open(log_path, "w") as f:
        f.write("elapsed_sec,rss_gb\n")
        t0 = time.time()
        while not stop_event.is_set():
            try:
                rss = _current_rss_gb(pid)
                elapsed = time.time() - t0
                line = f"{elapsed:.0f},{rss:.3f}"
                f.write(line + "\n")
                f.flush()
                print(f"[mem] t={elapsed:.0f}s  rss={rss:.2f}GB")
            except Exception as e:
                print(f"[mem] sample failed: {e}")
            stop_event.wait(MEM_SAMPLE_INTERVAL_SEC)


def main():
    print(f"run tag: {RUN_TAG}")
    print(f"gurobi log -> {GUROBI_LOG_PATH}")
    print(f"memory log -> {MEM_LOG_PATH}")
    print(f"summary    -> {SUMMARY_PATH}")

    params = load_parameters()
    locations = load_locations()
    G = build_graph(locations)

    scenarios = generate_scenarios(G, locations, num_scenarios=NUM_SCENARIOS, seed=SEED)
    instance = build_stochastic_instance(locations, scenarios, params, alpha=1.0)

    print(f"beta override: {instance['beta']} -> {BETA_OVERRIDE}")
    instance["beta"] = BETA_OVERRIDE
    tail_size = (1.0 - BETA_OVERRIDE) * NUM_SCENARIOS
    print(f"num_scenarios={NUM_SCENARIOS}  beta={BETA_OVERRIDE}  "
          f"CVaR tail size = (1-beta)*N = {tail_size:.1f} scenarios")

    # starts the memory sampler now, before the solve call, so the very
    # first reading (baseline, before gurobi allocates anything) is captured
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_memory_sampler, args=(os.getpid(), stop_event, MEM_LOG_PATH), daemon=True,
    )
    sampler.start()

    t0 = time.time()
    try:
        result = solve_stochastic_cvar(
            instance,
            vehicle_formulation="vif",
            time_limit=TIME_LIMIT_SEC,
            build_only=False,
            verbose=True,
        )
    finally:
        stop_event.set()
        sampler.join(timeout=5)
    elapsed = time.time() - t0

    print(f"\n=== RESULT (elapsed {elapsed:.1f}s) ===")
    print(f"status: {result['status']} (code {result['status_code']})")
    model = result["model"]
    print(f"NumVars={model.NumVars}  NumBinVars={model.NumBinVars}  NumConstrs={model.NumConstrs}")

    summary = {
        "run_tag": RUN_TAG,
        "num_scenarios": NUM_SCENARIOS,
        "beta": BETA_OVERRIDE,
        "tail_size": tail_size,
        "time_limit_sec": TIME_LIMIT_SEC,
        "elapsed_sec": elapsed,
        "status": result["status"],
        "status_code": result["status_code"],
        "num_vars": model.NumVars,
        "num_bin_vars": model.NumBinVars,
        "num_constrs": model.NumConstrs,
        "gurobi_log": GUROBI_LOG_PATH,
        "memory_log": MEM_LOG_PATH,
    }

    if result.get("objective_value") is not None:
        mip_gap = model.MIPGap if hasattr(model, "MIPGap") else None
        print(f"objective_value: {result['objective_value']}")
        print(f"MIPGap: {mip_gap:.4%}" if mip_gap is not None else "MIPGap: n/a")
        print(f"eta: {result.get('eta')}")
        print(f"selected_sites: {result.get('selected_sites')}")
        print(f"basing (l,j) count: {len(result.get('basing', {}))}")
        print(f"vehicle_arcs used: {len(result.get('vehicle_arcs', {}))}")
        print(f"unmet_demand nonzero entries: {len(result.get('unmet_demand', {}))}")
        summary.update({
            "objective_value": result["objective_value"],
            "mip_gap": mip_gap,
            "eta": result.get("eta"),
            "selected_sites": result.get("selected_sites"),
            "basing_count": len(result.get("basing", {})),
            "vehicle_arcs_used": len(result.get("vehicle_arcs", {})),
            "unmet_demand_nonzero": len(result.get("unmet_demand", {})),
        })
    else:
        # this branch is what actually happened both prior runs. no
        # incumbent means branch and bound never even started, the kill
        # happened during presolve/barrier/crossover before that phase began
        print("No incumbent solution found within the time limit.")
        summary["objective_value"] = None

    stats = result.get("subtour_callback_stats", {})
    print(f"subtour callback: invocations={stats.get('invocations')} cuts_added={stats.get('cuts_added')}")
    summary["subtour_callback_stats"] = stats

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
