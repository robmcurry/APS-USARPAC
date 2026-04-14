import os
from typing import Dict, List

import networkx as nx
import pandas as pd

from stoch_loader import load_parameters
from stoch_simulator import generate_scenarios
from stochastic_input_builder import build_stochastic_instance
from stochastic_model_course import solve_stochastic_cvar


GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
NUM_SCENARIOS = 10
OUTPUT_DIR = "output"


# --- Build locations from CSV ---
df = pd.read_csv("pacific_cities.csv")

locations = {}
for _, row in df.iterrows():
    node_id = int(row["Node ID"])
    locations[node_id] = {
        "name": row["Node Name"],
        "lat": float(row["Latitude"]),
        "lon": float(row["Longitude"]),
        "pop": float(row["Population"]),
        "country": row["Country"],
        "region": row["Region"],
    }

print("Loaded columns:", df.columns.tolist())
print("Sample location:", next(iter(locations.items())))

# --- Build a simple graph for testing ---
G = nx.Graph()
for i in locations:
    G.add_node(i)

# Simple fully connected graph (fine for testing only)
for i in locations:
    for j in locations:
        if i != j:
            G.add_edge(i, j)

# --- Load parameters and generate scenarios once ---
params = load_parameters()
scenarios = generate_scenarios(G, locations, num_scenarios=NUM_SCENARIOS)

print("\nScenario keys:", scenarios[0].keys())
print("Scenario ID:", scenarios[0]["scenario_id"])
print("Severity:", scenarios[0]["severity"])
print("Sample node severity:", list(scenarios[0]["node_severity"].items())[:5])

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows: List[Dict] = []
scenario_rows: List[Dict] = []
site_rows: List[Dict] = []

for gamma in GAMMA_VALUES:
    print(f"\n{'=' * 60}")
    print(f"Running gamma = {gamma}")
    print(f"{'=' * 60}")

    instance = build_stochastic_instance(
        locations=locations,
        undirected_edges=list(G.edges()),
        scenarios=scenarios,
        params=params,
        gamma=gamma,
    )

    results = solve_stochastic_cvar(instance, verbose=False)

    total_demand = sum(instance["demand"].values())
    total_unmet = sum(results.get("unmet_demand", {}).values())
    total_release = sum(results.get("release", {}).values())
    scenario_losses = results.get("scenario_losses", {})
    selected_sites = results.get("selected_sites", [])

    disrupted_scenarios = sum(1 for val in scenario_losses.values() if val > 1e-6)
    avg_scenario_loss = (
        sum(scenario_losses.values()) / len(scenario_losses) if scenario_losses else None
    )
    max_scenario_loss = max(scenario_losses.values()) if scenario_losses else None
    min_scenario_loss = min(scenario_losses.values()) if scenario_losses else None
    service_rate = None
    if total_demand > 0:
        service_rate = 1.0 - (total_unmet / total_demand)

    unmet_by_commodity: Dict[str, float] = {r: 0.0 for r in instance["commodities"]}
    for (_, _, commodity), unmet_val in results.get("unmet_demand", {}).items():
        unmet_by_commodity[commodity] += unmet_val

    summary_rows.append(
        {
            "gamma": gamma,
            "status": results.get("status"),
            "objective_value": results.get("objective_value"),
            "eta": results.get("eta"),
            "num_scenarios": len(instance["scenarios"]),
            "num_nodes": len(instance["nodes"]),
            "num_arcs": len(instance["arcs"]),
            "num_selected_sites": len(selected_sites),
            "selected_sites": "|".join(str(i) for i in selected_sites),
            "total_demand": total_demand,
            "total_release": total_release,
            "total_unmet": total_unmet,
            "service_rate": service_rate,
            "avg_scenario_loss": avg_scenario_loss,
            "min_scenario_loss": min_scenario_loss,
            "max_scenario_loss": max_scenario_loss,
            "disrupted_scenarios": disrupted_scenarios,
            "fraction_disrupted_scenarios": (
                disrupted_scenarios / len(instance["scenarios"]) if instance["scenarios"] else None
            ),
            "unmet_food": unmet_by_commodity.get("food", 0.0),
            "unmet_water": unmet_by_commodity.get("water", 0.0),
        }
    )

    for scenario_id, loss_val in sorted(scenario_losses.items()):
        total_unmet_in_scenario = sum(
            val
            for (w, _, _), val in results.get("unmet_demand", {}).items()
            if w == scenario_id
        )
        scenario_rows.append(
            {
                "gamma": gamma,
                "scenario_id": scenario_id,
                "loss": loss_val,
                "scenario_total_unmet": total_unmet_in_scenario,
                "scenario_disrupted": int(total_unmet_in_scenario > 1e-6),
            }
        )

    for site in selected_sites:
        site_rows.append(
            {
                "gamma": gamma,
                "node_id": site,
                "node_name": locations[site]["name"],
                "country": locations[site]["country"],
                "region": locations[site]["region"],
            }
        )

    print(
        f"gamma={gamma} | status={results.get('status')} | "
        f"objective={results.get('objective_value')} | "
        f"selected_sites={selected_sites} | total_unmet={total_unmet}"
    )

summary_df = pd.DataFrame(summary_rows)
scenario_df = pd.DataFrame(scenario_rows)
site_df = pd.DataFrame(site_rows)

summary_path = os.path.join(OUTPUT_DIR, "gamma_sensitivity_summary.csv")
scenario_path = os.path.join(OUTPUT_DIR, "gamma_sensitivity_scenario_losses.csv")
site_path = os.path.join(OUTPUT_DIR, "gamma_sensitivity_selected_sites.csv")

summary_df.to_csv(summary_path, index=False)
scenario_df.to_csv(scenario_path, index=False)
site_df.to_csv(site_path, index=False)

print("\nSaved files:")
print(f"  {summary_path}")
print(f"  {scenario_path}")
print(f"  {site_path}")
print("\nRun-level summary preview:")
print(summary_df)