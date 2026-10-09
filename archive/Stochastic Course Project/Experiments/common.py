import os
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from stoch_loader import load_parameters
from stoch_simulator import generate_scenarios
from stochastic_input_builder import build_stochastic_instance
from stochastic_model_course import solve_stochastic_cvar


def summarize_transport_penalty_scale(results: dict) -> dict:
    """
    Compare transportation cost to unmet-demand penalty from one solve.
    """
    scenario_transport = results.get("scenario_transport_cost", {})
    scenario_penalty = results.get("scenario_unmet_penalty", {})
    scenario_loss = results.get("scenario_losses", {})

    scenario_ratios = {}
    scenario_transport_shares = {}

    for w, loss_val in scenario_loss.items():
        transport_val = float(scenario_transport.get(w, 0.0))
        penalty_val = float(scenario_penalty.get(w, 0.0))

        scenario_ratios[w] = (
            transport_val / penalty_val if penalty_val > 1e-12 else None
        )
        scenario_transport_shares[w] = (
            transport_val / loss_val if loss_val > 1e-12 else None
        )

    valid_ratios = [v for v in scenario_ratios.values() if v is not None]
    valid_shares = [v for v in scenario_transport_shares.values() if v is not None]

    total_transport = sum(float(v) for v in scenario_transport.values())
    total_penalty = sum(float(v) for v in scenario_penalty.values())
    total_loss = sum(float(v) for v in scenario_loss.values())

    return {
        "total_transport_cost": total_transport,
        "total_unmet_penalty": total_penalty,
        "total_loss": total_loss,
        "transport_to_penalty_ratio_total": (
            total_transport / total_penalty if total_penalty > 1e-12 else None
        ),
        "transport_share_of_loss_total": (
            total_transport / total_loss if total_loss > 1e-12 else None
        ),
        "avg_transport_to_penalty_ratio": (
            sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
        ),
        "avg_transport_share_of_loss": (
            sum(valid_shares) / len(valid_shares) if valid_shares else None
        ),
        "scenario_transport_to_penalty_ratio": scenario_ratios,
        "scenario_transport_share_of_loss": scenario_transport_shares,
    }


def load_locations(csv_path: str = "pacific_cities.csv") -> Dict[int, dict]:
    """
    Load node metadata from CSV.
    """
    df = pd.read_csv(csv_path)

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
            "hub_type": str(row["hub_type"]).strip().lower(),
        }

    print("Loaded columns:", df.columns.tolist())
    print("Sample location:", next(iter(locations.items())))
    return locations


def build_test_graph(locations: Dict[int, dict]) -> nx.Graph:
    """
    Build a simple graph for testing.
    Replace this later with your pruned travel-time-feasible graph.
    """
    graph = nx.Graph()

    for node_id in locations:
        graph.add_node(node_id)

    for i in locations:
        for j in locations:
            if i != j:
                graph.add_edge(i, j)

    return graph


def make_output_dir(base_output_dir: str, experiment_name: str) -> str:
    """
    Create and return the output directory for one experiment.
    """
    experiment_output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    return experiment_output_dir


def save_experiment_outputs(
    experiment_output_dir: str,
    summary_rows: List[Dict],
    scenario_rows: List[Dict],
    site_rows: List[Dict],
    flow_rows: List[Dict],
) -> None:
    """
    Save standard output CSVs for one experiment.
    """
    summary_df = pd.DataFrame(summary_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    site_df = pd.DataFrame(site_rows)
    flow_df = pd.DataFrame(flow_rows)

    summary_path = os.path.join(experiment_output_dir, "summary.csv")
    scenario_path = os.path.join(experiment_output_dir, "scenario_losses.csv")
    site_path = os.path.join(experiment_output_dir, "selected_sites.csv")
    flow_path = os.path.join(experiment_output_dir, "flows.csv")

    summary_df.to_csv(summary_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    site_df.to_csv(site_path, index=False)
    flow_df.to_csv(flow_path, index=False)

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {scenario_path}")
    print(f"  {site_path}")
    print(f"  {flow_path}")

    print("\nRun-level summary preview:")
    print(summary_df)


def run_single_experiment(
    experiment_code: str,
    experiment_label: str,
    gamma: float,
    beta: float,
    p_max: int,
    budget: Optional[float],
    tau: float,
    rho: float,
    num_scenarios: int = 10,
    base_output_dir: str = "output",
    verbose: bool = False,
) -> None:
    """
    Run one experiment and save outputs to its own folder.

    This assumes your parameter loader returns a mutable dictionary-like object.
    Update the override keys below to match your actual parameter structure.
    """
    print(f"\n{'=' * 70}")
    print(f"Running {experiment_code}: {experiment_label}")
    print(f"{'=' * 70}")

    locations = load_locations()
    graph = build_test_graph(locations)

    params = load_parameters()

    # Parameter overrides
    params["beta"] = beta
    params["P_max"] = p_max

    if "inventory_disruption" not in params:
        params["inventory_disruption"] = {}
    params["inventory_disruption"]["cutoff_severity"] = tau

    if "safety_stock" not in params["inventory_disruption"]:
        params["inventory_disruption"]["safety_stock"] = {}
    params["inventory_disruption"]["safety_stock"]["fraction"] = rho

    if budget is not None:
        if "prepositioning" not in params:
            params["prepositioning"] = {}
        params["prepositioning"]["selection_budget"] = budget


    scenarios = generate_scenarios(graph, locations, num_scenarios=num_scenarios)

    print("\nScenario keys:", scenarios[0].keys())
    print("Scenario ID:", scenarios[0]["scenario_id"])
    print("Severity:", scenarios[0]["severity"])
    print("Sample node severity:", list(scenarios[0]["node_severity"].items())[:5])

    instance = build_stochastic_instance(
        locations=locations,
        undirected_edges=list(graph.edges()),
        scenarios=scenarios,
        params=params,
        gamma=gamma,
    )


    results = solve_stochastic_cvar(instance, verbose=verbose)
    diagnostics = summarize_transport_penalty_scale(results)

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

    summary_rows: List[Dict] = []
    scenario_rows: List[Dict] = []
    site_rows: List[Dict] = []
    flow_rows: List[Dict] = []

    summary_rows.append(
        {
            "experiment_code": experiment_code,
            "experiment_label": experiment_label,
            "gamma": gamma,
            "beta": beta,
            "P_max": p_max,
            "budget_input": budget,
            "budget_used": instance.get("B"),
            "tau": tau,
            "rho": rho,
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
            "total_transport_cost": diagnostics["total_transport_cost"],
            "total_unmet_penalty": diagnostics["total_unmet_penalty"],
            "transport_to_penalty_ratio_total": diagnostics["transport_to_penalty_ratio_total"],
            "transport_share_of_loss_total": diagnostics["transport_share_of_loss_total"],
            "avg_transport_to_penalty_ratio": diagnostics["avg_transport_to_penalty_ratio"],
            "avg_transport_share_of_loss": diagnostics["avg_transport_share_of_loss"],
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
                "experiment_code": experiment_code,
                "experiment_label": experiment_label,
                "gamma": gamma,
                "beta": beta,
                "P_max": p_max,
                "budget_input": budget,
                "budget_used": instance.get("B"),
                "tau": tau,
                "rho": rho,
                "scenario_id": scenario_id,
                "loss": loss_val,
                "transport_cost": results.get("scenario_transport_cost", {}).get(scenario_id, 0.0),
                "unmet_penalty": results.get("scenario_unmet_penalty", {}).get(scenario_id, 0.0),
                "transport_to_penalty_ratio": diagnostics["scenario_transport_to_penalty_ratio"].get(scenario_id),
                "transport_share_of_loss": diagnostics["scenario_transport_share_of_loss"].get(scenario_id),
                "scenario_total_unmet": total_unmet_in_scenario,
                "scenario_disrupted": int(total_unmet_in_scenario > 1e-6),
            }
        )

    for site in selected_sites:
        site_rows.append(
            {
                "experiment_code": experiment_code,
                "experiment_label": experiment_label,
                "gamma": gamma,
                "beta": beta,
                "P_max": p_max,
                "budget_input": budget,
                "budget_used": instance.get("B"),
                "tau": tau,
                "rho": rho,
                "node_id": site,
                "node_name": locations[site]["name"],
                "country": locations[site]["country"],
                "region": locations[site]["region"],
            }
        )

    flows_dict = results.get("flows", {})
    if flows_dict:
        for (scenario_id, from_node, to_node, commodity), flow_val in sorted(flows_dict.items()):
            if flow_val > 1e-6:
                flow_rows.append(
                    {
                        "experiment_code": experiment_code,
                        "experiment_label": experiment_label,
                        "gamma": gamma,
                        "beta": beta,
                        "P_max": p_max,
                        "budget_input": budget,
                        "budget_used": instance.get("B"),
                        "tau": tau,
                        "rho": rho,
                        "scenario_id": scenario_id,
                        "from_node": from_node,
                        "from_name": locations[from_node]["name"],
                        "to_node": to_node,
                        "to_name": locations[to_node]["name"],
                        "commodity": commodity,
                        "flow": flow_val,
                    }
                )
    else:
        for scenario_id, from_node, to_node, commodity, flow_val in results.get("positive_flows", []):
            if flow_val > 1e-6:
                flow_rows.append(
                    {
                        "experiment_code": experiment_code,
                        "experiment_label": experiment_label,
                        "gamma": gamma,
                        "beta": beta,
                        "P_max": p_max,
                        "budget_input": budget,
                        "budget_used": instance.get("B"),
                        "tau": tau,
                        "rho": rho,
                        "scenario_id": scenario_id,
                        "from_node": from_node,
                        "from_name": locations[from_node]["name"],
                        "to_node": to_node,
                        "to_name": locations[to_node]["name"],
                        "commodity": commodity,
                        "flow": flow_val,
                    }
                )

    total_flow = sum(results.get("flows", {}).values())
    num_flows = len(results.get("flows", {}))
    num_releases = len(results.get("release", {}))

    print(
        f"{experiment_code} | status={results.get('status')} | "
        f"objective={results.get('objective_value')} | "
        f"selected_sites={selected_sites} | total_unmet={total_unmet}"
    )
    print(
        "transport diagnostics | "
        f"total_transport={diagnostics['total_transport_cost']:.4f} | "
        f"total_penalty={diagnostics['total_unmet_penalty']:.4f} | "
        f"transport/penalty={diagnostics['transport_to_penalty_ratio_total']:.6f} | "
        f"transport share of loss={diagnostics['transport_share_of_loss_total']:.6f}"
    )
    print(
        "flow diagnostics | "
        f"total_release={total_release:.4f} | "
        f"total_flow={total_flow:.4f} | "
        f"num_release_entries={num_releases} | "
        f"num_flow_entries={num_flows}"
    )

    experiment_folder = f"{experiment_code}_{experiment_label.lower().replace(' ', '_')}"
    experiment_output_dir = make_output_dir(base_output_dir, experiment_folder)

    save_experiment_outputs(
        experiment_output_dir=experiment_output_dir,
        summary_rows=summary_rows,
        scenario_rows=scenario_rows,
        site_rows=site_rows,
        flow_rows=flow_rows,
    )