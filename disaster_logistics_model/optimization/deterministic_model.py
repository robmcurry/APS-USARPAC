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
    # Load infrastructure loss rate from scenario type_attributes (default 0.0)
    loss_rate = scenario.get("type_attributes", {}).get("infrastructure_loss_rate", 0.0)
    cap = {(int(i), int(j), c): scenario['capacity'][(i, j, c)] * (1 - loss_rate) for (i, j, c) in scenario['capacity']}

    required_safety_stock = 10

    # Parameter loading
    params = scenario.get("params", {})
    b = {c: params.get("commodity_size", {}).get(c, 1.0) for c in commodity_list}
    ell = {(i, c): params.get("safety_stock", {}).get((i, c), required_safety_stock) for i in node_list for c in commodity_list}
    mu_v = {v: params.get("vehicle_capacity", {}).get(v, 100.0) for v in vehicle_list}
    L_c = {c: params.get("min_APS_per_commodity", {}).get(c, 1) for c in commodity_list}
    M_node_capacity = {(i, t): params.get("node_capacity", {}).get((i, t), 100.0) for i in node_list for t in time_period_list}

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

    # New objective coefficients (placeholders, update as needed)
    delta = {(i, c, t): 1.0 for i in node_list for c in commodity_list for t in time_period_list}  # Placeholder, update if needed
    nu = {(i, c): 1.0 for i in node_list for c in commodity_list}  # Placeholder, update if needed

    model.setObjective(
        quicksum(delta[i, c, t] * z[i, c, t] for i in node_list for c in commodity_list for t in time_period_list) +
        quicksum(nu[i, c] * alpha[i, c] for i in node_list for c in commodity_list),
        GRB.MINIMIZE
    )

    # 1. MaxVehicles
    model.addConstr(quicksum(m_i[i] for i in node_list) <= 10, name="MaxVehicles")

    # 2. q_r_link
    model.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), name="q_r_link")

    # 3. r_p_link
    model.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), name="r_p_link")

    # 4. Max_APS_Locations
    model.addConstr(quicksum(p[i] for i in node_list) <= P_max, name="Max_APS_Locations")

    # 5. Min_APS_Per_Commodity
    model.addConstrs((
        quicksum(r[i, c] for i in node_list) >= L_c[c]
        for c in commodity_list
    ), name="Min_APS_Per_Commodity")

    # 6. InitialInventory
    model.addConstrs((w[i, c, 0] == q[i, c] for i in node_list for c in commodity_list), name="InitialInventory")

    # 7. DemandSplit
    model.addConstrs((
        quicksum(y[i, c, tau] for tau in time_period_list if tau <= t) + z[i, c, t] == d.get((i, c), 0)
        for i in node_list for c in commodity_list for t in time_period_list
    ), name="DemandSplit")

    # 8. InventoryConservation
    model.addConstrs((
        w[i, c, t] == w[i, c, t - 1]
        + quicksum(x[j, i, c, t, v] for j in node_list if (j, i) in arc_list for v in vehicle_list)
        - quicksum(x[i, j, c, t, v] for j in node_list if (i, j) in arc_list for v in vehicle_list)
        - y[i, c, t]
        for i in node_list for c in commodity_list for t in time_period_list
    ), name="InventoryConservation")

    # 9. DemandFromInventory
    #model.addConstrs((y[i, c, t] <= w[i, c, t] for i in node_list for c in commodity_list for t in time_period_list), name="DemandFromInventory")

    # 10. ArcCapacity
    model.addConstrs((
        x[i, j, c, t, v] <= cap.get((i, j, c), 0)
        for (i, j) in arc_list for c in commodity_list for t in time_period_list for v in vehicle_list
    ), name="ArcCapacity")

    # 11. MatchFlowToVehicle
    model.addConstrs((
        quicksum(b[c] * x[i, j, c, t, v] for c in commodity_list) <= mu_v[v] * bar_x[i, j, t, v]
        for (i, j) in arc_list for t in time_period_list for v in vehicle_list
    ), name="MatchFlowToVehicle")

    # 12. VehicleFlowBalance
    model.addConstrs((
        quicksum(bar_x[i, j, t, v] for j in node_list if (i, j) in arc_list)
        - quicksum(bar_x[j, i, t, v] for j in node_list if (j, i) in arc_list)
        + bar_w[i, t, v]
        - (bar_w[i, t - 1, v] if t > 0 else 0)
        == 0
        for i in node_list for t in time_period_list for v in vehicle_list
    ), name="VehicleFlowBalance")

    # 13. VehiclesReturn
    model.addConstrs((
        quicksum(bar_w[i, 0, v] for v in vehicle_list) == m_i[i]
        for i in node_list
    ), name="VehiclesStart")

    # 13b. VehiclesReturn
    model.addConstrs((
        quicksum(bar_w[i, time_period_list[-1], v] for v in vehicle_list) == m_i[i]
        for i in node_list
    ), name="VehiclesReturn")

    # 14. SafetyStock
    model.addConstrs((
        alpha[i, c] >= ell[i, c] - (
            q[i, c] - quicksum(
                x[i, j, c, t, v] - x[j, i, c, t, v]
                for t in time_period_list
                for v in vehicle_list
                for j in node_list if (i, j) in arc_list and (j, i) in arc_list
            )
        )- M * (1 - p[i])
        for i in node_list for c in commodity_list
    ), name="SafetyStock")

    # 15. NodeCapacityLimit
    model.addConstrs((
        quicksum(b[c] * w[i, c, t] for c in commodity_list) <= M_node_capacity[i, t]
        for i in node_list for t in time_period_list
    ), name="NodeCapacityLimit")

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
        "scenario_type": scenario.get("scenario_type", "Unknown"),
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
    # Carry forward scenario type_attributes for batch summary
    results["impact_radius_km"] = scenario.get("type_attributes", {}).get("impact_radius_km")
    results["infrastructure_loss_rate"] = scenario.get("type_attributes", {}).get("infrastructure_loss_rate")
    results["geographic_tag"] = scenario.get("type_attributes", {}).get("geographic_tag")

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