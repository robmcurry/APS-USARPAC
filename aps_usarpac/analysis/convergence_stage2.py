"""
convergence_stage2.py

Stage 2 of the SAA convergence check: extends stage 1 (N=100) by solving the
alpha=0.5 instance (uniform degradation-matrix scaling; the post-Step-4
equivalent of the old gamma=0.5 arc-fragility scalar) at larger scenario
counts (N=200, N=500) across the same five scenario draws (seeds), to assess
how the objective value and site selection variance shrinks as the sample
size grows.

Run as `python -m analysis.convergence_stage2` from the aps_usarpac/ project
root. Results are written incrementally to output/convergence_stage2.csv
(one row per num_scenarios/seed combination) and the full Gurobi log for each
run is written to output/convergence_stage2_n{num_scenarios}_seed{seed}.log.
"""

import os

import pandas as pd

from config.loader import load_parameters
from network.network_builder import load_locations, build_graph
from scenarios.scenario_generator import generate_scenarios
from model.input_builder import build_stochastic_instance
from model.model import solve_stochastic_cvar

# Degradation intensity for the convergence instance. Pre-Step-4 this was the
# scalar gamma=0.5 arc-fragility parameter; post-Step-4 the equivalent knob is
# alpha, which uniformly scales all degradation_matrix[mode][type] entries.
# NOTE (D2 audit): prior runs of this script passed gamma=0.5 into **_kwargs,
# which silently ignored it — those runs actually solved the alpha=1.0
# (full-degradation) default, not the gamma=0.5 instance their outputs claim.
ALPHA = 0.5
SCENARIO_COUNTS = [200, 500]
SEEDS = [32, 42, 77, 99, 123]
SHANGHAI_NODE_ID = 48
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def main():
    params = load_parameters()
    locations = load_locations()
    G = build_graph(locations)
    commodities = list(params["commodities"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    out_path = os.path.join(OUTPUT_DIR, "convergence_stage2.csv")

    for num_scenarios in SCENARIO_COUNTS:
        for seed in SEEDS:
            print(f"\n{'=' * 60}")
            print(f"N = {num_scenarios} | Seed = {seed}")
            print(f"{'=' * 60}")

            scenarios = generate_scenarios(G, locations, num_scenarios=num_scenarios, seed=seed)

            instance = build_stochastic_instance(
                locations=locations,
                scenarios=scenarios,
                params=params,
                alpha=ALPHA,
            )

            total_demand = sum(instance["demand"].values())
            # demand is keyed by (w, i, r) - scenario, node, commodity - not (i, r).
            # Sum across all scenarios for this node (pre-existing bug fix: this
            # previously indexed with a 2-tuple and raised KeyError on the first
            # seed of every run since the Step 3/4 modal-arc demand schema change).
            shanghai_demand = sum(
                instance["demand"][(w, SHANGHAI_NODE_ID, r)]
                for w in instance["scenarios"] for r in commodities
            )
            shanghai_demand_share = shanghai_demand / total_demand if total_demand > 0 else None

            # capture the full gurobi log for this run by redirecting the
            # process's stdout file descriptor while optimize() runs - gurobi
            # writes its log directly to fd 1, not through python's sys.stdout
            log_path = os.path.join(
                OUTPUT_DIR, f"convergence_stage2_n{num_scenarios}_seed{seed}.log"
            )
            stdout_fd = 1
            saved_stdout_fd = os.dup(stdout_fd)
            log_file = open(log_path, "w")
            os.dup2(log_file.fileno(), stdout_fd)
            try:
                results = solve_stochastic_cvar(
                    instance, time_limit=3600, mip_gap=params["mip_gap"], verbose=True
                )
            finally:
                os.dup2(saved_stdout_fd, stdout_fd)
                os.close(saved_stdout_fd)
                log_file.close()

            # M1 memory fix: capture scalars, then dispose the Gurobi model so
            # the next run doesn't start with the previous model still resident.
            model = results.pop("model", None)
            solve_time_seconds = model.Runtime
            mip_gap_at_termination = model.MIPGap
            model.dispose()
            results.pop("variables", None)

            selected_sites = results.get("selected_sites", [])
            selected_sites_names = [locations[i]["name"] for i in selected_sites]

            total_unmet = sum(results.get("unmet_demand", {}).values())

            per_scenario_sr = {}
            for w in instance["scenarios"]:
                unmet_w = sum(
                    val for (ww, _, _), val in results.get("unmet_demand", {}).items()
                    if ww == w
                )
                per_scenario_sr[w] = (
                    1.0 - (unmet_w / total_demand) if total_demand > 0 else None
                )
            valid_sr = [v for v in per_scenario_sr.values() if v is not None]
            sr_mean = sum(valid_sr) / len(valid_sr) if valid_sr else None
            sr_min = min(valid_sr) if valid_sr else None

            row = {
                "num_scenarios": num_scenarios,
                "seed": seed,
                "objective_value": results.get("objective_value"),
                "selected_sites_ids": "|".join(str(i) for i in selected_sites),
                "selected_sites_names": "|".join(selected_sites_names),
                "sr_mean": sr_mean,
                "sr_min": sr_min,
                "total_demand": total_demand,
                "total_unmet": total_unmet,
                "solve_time_seconds": solve_time_seconds,
                "mip_gap_at_termination": mip_gap_at_termination,
                "shanghai_demand_share": shanghai_demand_share,
            }
            rows.append(row)

            # save incrementally after every run so an overnight crash
            # doesn't lose completed solves
            pd.DataFrame(rows).to_csv(out_path, index=False)

            print(
                f"N={num_scenarios} seed={seed} | status={results.get('status')} | "
                f"obj={results.get('objective_value')} | "
                f"sites={selected_sites} | "
                f"time={solve_time_seconds:.2f}s"
            )

    print(f"\nSaved {out_path}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
