# deterministic_model.py

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

    model.setObjective(
        quicksum(z[i, c, t] for i in node_list for c in commodity_list for t in time_period_list) +
        quicksum(alpha[i, c] for i in node_list for c in commodity_list) +
        0.001 * quicksum(q[i, c] for i in node_list for c in commodity_list),
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
        "flow_summary": dict(flow_summary)
    }

    return results
if __name__ == "__main__":
    # Minimal test with synthetic scenario
    dummy_scenario = {
        "scenario_id": 0,
        "demand": {(1, 'food'): 100, (2, 'water'): 200},
        "supply": {(1, 'food'): 150, (2, 'water'): 250},
        "capacity": {(1, 2, 'food'): 300, (2, 1, 'water'): 300}
    }

    result = solve_deterministic_vrp_with_aps(dummy_scenario)
    print("APS Objective:", result["objective"])
    print("Selected APS Locations:", result["aps_locations"])