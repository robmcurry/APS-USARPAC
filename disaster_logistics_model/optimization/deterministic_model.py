# deterministic_model.py

import yaml

from gurobipy import Model, GRB, quicksum
from collections import defaultdict

def solve_deterministic_vrp_with_aps(scenario, time_periods=range(1, 6), vehicle_list=None, P_max=5, M=10000):
    if vehicle_list is None:
        vehicle_list = list(range(1, 6))

    time_period_list = list(time_periods)
    time_period_list2 = list(range(0, max(time_period_list) + 1))

    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))

    d = {(int(i), c): scenario['demand'][(i, c)] for (i, c) in scenario['demand']}
    cap = {(int(i), int(j), c): scenario['capacity'][(i, j, c)] for (i, j, c) in scenario['capacity']}

    required_safety_stock = 10

    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

    x = model.addVars(arc_list, commodity_list, time_period_list, vehicle_list, name="x", lb=0)
    y = model.addVars(node_list, commodity_list, time_period_list, name="y", lb=0)
    z = model.addVars(node_list, commodity_list, time_period_list, name="z", lb=0)
    w = model.addVars(node_list, commodity_list, time_period_list2, name="w", lb=0)
    alpha = model.addVars(node_list, commodity_list, name="alpha", lb=0)

    q = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="q", lb=0)
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    p = model.addVars(node_list, vtype=GRB.BINARY, name="p")

    m_i = model.addVars(node_list, name="m_i", lb=0)
    bar_x = model.addVars(arc_list, time_period_list, vehicle_list, vtype=GRB.BINARY, name="bar_x")
    bar_w = model.addVars(node_list, time_period_list2, vehicle_list, vtype=GRB.BINARY, name="bar_w")

    # Load objective weights from YAML config
    with open("disaster_logistics_model/config/model_parameters.yaml", "r") as f:
        param_config = yaml.safe_load(f)
    weights = param_config.get("objective_weights", {})
    w_demand = weights.get("unmet_demand", 1.0)
    w_safety = weights.get("safety_stock", 1.0)
    w_inventory = weights.get("prepositioning", 1.0)

    # Estimate normalization constants
    total_demand = sum(scenario["demand"].values())
    demand_norm = max(1, total_demand)
    safety_norm = max(1, len(node_list) * len(commodity_list) * len(time_period_list))
    q_norm = max(1, len(node_list) * len(commodity_list) * 100)  # assumes q[i,c] ≤ 100 typically

    # Define normalized terms
    normalized_unmet_demand = quicksum(z[i, c, t] for i in node_list for c in commodity_list for t in time_period_list) / demand_norm
    normalized_safety_penalty = quicksum(alpha[i, c] for i in node_list for c in commodity_list) / safety_norm
    normalized_q_cost = quicksum(q[i, c] for i in node_list for c in commodity_list) / q_norm

    # Apply weights to normalized terms
    model.setObjective(
        w_demand * normalized_unmet_demand +
        w_safety * normalized_safety_penalty +
        w_inventory * normalized_q_cost,
        GRB.MINIMIZE
    )

    model.addConstrs((w[i, c, 0] == q[i, c] for i in node_list for c in commodity_list), name="InitialInventory")

    model.addConstrs((
        quicksum(y[i, c, tau] for tau in time_period_list if tau <= t) + z[i, c, t] == d.get((i, c), 0)
        for i in node_list for c in commodity_list for t in time_period_list
    ), name="DemandSplit")

    model.addConstrs((
        w[i, c, t] == w[i, c, t - 1]
        + quicksum(x[j, i, c, t, v] for j in node_list if (j, i) in arc_list for v in vehicle_list)
        - quicksum(x[i, j, c, t, v] for j in node_list if (i, j) in arc_list for v in vehicle_list)
        - y[i, c, t]
        for i in node_list for c in commodity_list for t in time_period_list
    ), name="InventoryConservation")

    model.addConstrs((y[i, c, t] <= w[i, c, t] for i in node_list for c in commodity_list for t in time_period_list), name="DemandFromInventory")

    model.addConstrs((
        x[i, j, c, t, v] <= cap.get((i, j, c), 0)
        for (i, j) in arc_list for c in commodity_list for t in time_period_list for v in vehicle_list
    ), name="ArcCapacity")

    model.addConstrs((
        alpha[i, c] >= required_safety_stock - quicksum(w[i, c, t] for t in time_period_list)
        for i in node_list for c in commodity_list
    ), name="SafetyStock")

    model.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), name="q_r_link")
    model.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), name="r_p_link")
    model.addConstr(quicksum(p[i] for i in node_list) <= P_max, name="Max_APS_Locations")

    model.addConstr(quicksum(m_i[i] for i in node_list) <= 10, name="MaxVehicles")

    model.addConstrs((
        quicksum(bar_x[i, j, t, v] for j in node_list if (i, j) in arc_list)
        - quicksum(bar_x[j, i, t, v] for j in node_list if (j, i) in arc_list)
        + bar_w[i, t, v]
        - (bar_w[i, t - 1, v] if t > 0 else 0)
        == 0
        for i in node_list for t in time_period_list for v in vehicle_list
    ), name="VehicleFlowBalance")

    model.addConstrs((
        bar_w[i, time_period_list[-1], v] == m_i[i]
        for i in node_list for v in vehicle_list
    ), name="VehiclesReturn")

    model.optimize()

    flow_summary = defaultdict(float)
    for (i, j, c, t, v) in x.keys():
        val = x[i, j, c, t, v].x
        if val > 0:
            flow_summary[(i, j)] += val

    # Load political scores
    with open("disaster_logistics_model/config/political_scores.yaml", "r") as f:
        political_scores = yaml.safe_load(f)

    selected_aps = [i for i in node_list if p[i].x > 0.5]
    avg_political_score = (
        sum(political_scores.get(i, 0) for i in selected_aps) / max(1, len(selected_aps))
    )

    # Calculate total demand
    total_demand = sum(scenario["demand"].values())
    normalized_risk = (model.ObjVal if model.Status == GRB.OPTIMAL else 0) / max(1, total_demand)

    # Normalize cost using objective across demand scale
    normalized_cost = (model.ObjVal if model.Status == GRB.OPTIMAL else 0) / 1e9  # or use max across runs

    # Invert political score for scoring (higher is better, so subtract from 10)
    political_component = 10 - avg_political_score

    # Weight parameters for composite scoring
    w_risk = 0.5
    w_cost = 0.3
    w_political = 0.2

    composite_score = w_risk * normalized_risk + w_cost * normalized_cost + w_political * political_component

    results = {
        "scenario_id": scenario["scenario_id"],
        "objective": model.ObjVal if model.Status == GRB.OPTIMAL else None,
        "aps_locations": [i for i in node_list if p[i].x > 0.5],
        "num_q_positive": sum(1 for i in node_list for c in commodity_list if q[i, c].x > 0),
        "num_initial_inventory": sum(1 for i in node_list for c in commodity_list if w[i, c, 0].x > 0),
        "total_flow": sum(flow_summary.values()),
        "status": model.Status,
        "q": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "q_full": {(i, c): q[i, c].x for i in node_list for c in commodity_list},
        "initial_inventory": {(i, c): w[i, c, 0].x for i in node_list for c in commodity_list if w[i, c, 0].x > 0},
        "w_full": {(i, c, t): w[i, c, t].x for i in node_list for c in commodity_list for t in time_period_list2},
        "flow_summary": dict(flow_summary),
        "severity": scenario.get("severity"),
        "epicenter": scenario.get("epicenter"),
        "epicenter_neighbors": scenario.get("epicenter_neighbors"),
        "affected_nodes": scenario.get("affected_nodes"),
        "demand": scenario.get("demand"),
        "supply": scenario.get("supply"),
        "capacity": scenario.get("capacity"),
        "avg_political_score": avg_political_score,
        "composite_score": composite_score,
    }

    return results
if __name__ == "__main__":
    # Minimal test with synthetic scenario
    dummy_scenario = {
        "scenario_id": 0,
        "demand": {(1, 'food'): 100, (2, 'water'): 200},
        "supply": {(1, 'food'): 150, (2, 'water'): 250},
        "capacity": {(1, 2, 'food'): 300, (2, 1, 'water'): 300},
        "severity": 2.0,
        "epicenter": 1,
        "epicenter_neighbors": 2,
        "affected_nodes": [1, 2],
    }

    result = solve_deterministic_vrp_with_aps(dummy_scenario)
    print("APS Objective:", result["objective"])
    print("Selected APS Locations:", result["aps_locations"])