# deterministic_model.py

from gurobipy import Model, GRB, quicksum
from collections import defaultdict
import networkx as nx


def solve_deterministic_vrp_with_aps(scenario, time_periods=range(1, 6), vehicle_list=None, P_max=5, M=1000000):
    if vehicle_list is None:
        vehicle_list = list(range(1, 6))

    time_period_list = list(time_periods)

    time_period_list2 = list(range(0, max(time_period_list) + 1))

    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))
    print(node_list)
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))

    d = {(int(i), c): int(scenario['demand'][(i, c)]) for (i, c) in scenario['demand']}

    destroy = {(int(i), c): 1 if d.get((i, c), 0) > 55500 else 0 for (i, c) in scenario['demand']}

    cap = {(int(i), int(j), c): scenario['capacity'][(i, j, c)] for (i, j, c) in scenario['capacity']}

    required_safety_stock = {(i, c): 10 for i in node_list for c in commodity_list}

    L = {c: 2 for c in commodity_list}

    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

    x = model.addVars(arc_list, commodity_list, time_period_list, vehicle_list, name="x", lb=0)
    y = model.addVars(node_list, commodity_list, time_period_list, name="y", lb=0)
    z = model.addVars(node_list, commodity_list, time_period_list, name="z", lb=0)
    w = model.addVars(node_list, commodity_list, time_period_list2, name="w", lb=0)
    alpha = model.addVars(commodity_list, name="alpha", lb=0)

    q = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="q", lb=0)
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    p = model.addVars(node_list, vtype=GRB.BINARY, name="p")

    m_i = model.addVars(node_list, name="m_i", lb=0)
    bar_x = model.addVars(arc_list, time_period_list, vehicle_list, vtype=GRB.BINARY, name="bar_x")
    bar_w = model.addVars(node_list, time_period_list2, vehicle_list, vtype=GRB.BINARY, name="bar_w")

    model.setObjective(
        100*quicksum((1/t)*z[i, c, t] for i in node_list for c in commodity_list for t in time_period_list)+
            quicksum(alpha[c] for c in commodity_list) ,
        GRB.MINIMIZE
    )

    model.addConstrs((w[i, c, 0] <= q[i, c] for i in node_list for c in commodity_list), name="InitialInventory")

    model.addConstrs((w[i, c, 0] <= 10000*(1-destroy[i, c]) for i in node_list for c in commodity_list), name="InitialInventory")

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

    # Hmmmm. Not sure what this is doing for sure. The y-variable is what is consumed, but the w is what is stored.
    # model.addConstrs((y[i, c, t] <= w[i, c, t] for i in node_list for c in commodity_list for t in time_period_list), name="DemandFromInventory")

    model.addConstrs((
        x[i, j, c, t, v] <= cap.get((i, j, c), 0)
        for (i, j) in arc_list for c in commodity_list for t in time_period_list for v in vehicle_list
    ), name="ArcCapacity")

    # This is close. We need to update this to not be summed over all time periods. Instead it needs to be the w for the last time period.
    model.addConstrs((
        alpha[c] >= 1100 - quicksum(w[i, c, time_period_list[-1]]
        for i in node_list) for c in commodity_list
    ), name="SafetyStock")

    model.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), name="q_r_link")
    model.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), name="r_p_link")
    model.addConstr(quicksum(p[i] for i in node_list) <= P_max, name="Max_APS_Locations")

    model.addConstr(quicksum(m_i[i] for i in node_list) <= 10, name="MaxVehicles")

    model.addConstrs((
        quicksum(bar_x[i, j, t, v] for j in node_list if (i, j) in arc_list)
        - quicksum(bar_x[j, i, t, v] for j in node_list if (j, i) in arc_list)
        + bar_w[i, t, v]
        - (bar_w[i, t - 1, v])
        == 0
        for i in node_list for t in time_period_list for v in vehicle_list
    ), name="VehicleFlowBalance")

    model.addConstrs((
        quicksum(bar_w[i, time_period_list[-1], v] for v in vehicle_list) == m_i[i]
        for i in node_list
    ), name="VehiclesReturn")

    model.addConstrs((
        quicksum(bar_w[i, 0, v] for v in vehicle_list) == m_i[i]
        for i in node_list
    ), name="VehiclesStart")

    model.addConstrs((
        quicksum(bar_w[i, 0, v] for i in node_list) <= 1
        for v in vehicle_list
    ), name="SingleNodePerVehicle")

    model.addConstrs(
        (
            quicksum(x[i, j, c, t, v] for c in commodity_list) <= 1000 * bar_x[i, j, t, v]
            for (i, j) in arc_list
            for t in time_period_list
            for v in vehicle_list
        ),
        "VehicleOnPositiveFlowArcs",
    )
    model.addConstrs(
        (
            quicksum(bar_x[i, j, 1, v] for (i, j) in arc_list) <=
            quicksum(bar_x[i, j, 1, v + 1] for (i, j) in arc_list)
            for v in range(1, len(vehicle_list))  # v = 1, ..., |V|-1
        ),
        "VehicleFlowMonotonicity"
    )

    # Add subtour elimination constraints

    counter = 1
    while True:

        print("Solve ", counter)
        model.optimize()

        # Check for flow cycles in each time period
        def find_flow_cycles(x_vals, time_period):
            # Create separate flow graphs for each vehicle 
            for v in vehicle_list:
                flow_graph = nx.DiGraph()
                for (i, j, c, t, vh) in x_vals:
                    if t == time_period and vh == v and x_vals[(i, j, c, t, vh)].x > 0:
                        if not flow_graph.has_edge(i, j):
                            flow_graph.add_edge(i, j)
                try:
                    cycle = nx.find_cycle(flow_graph)
                    return cycle
                except nx.NetworkXNoCycle:
                    continue
            return None

        # Check all time periods for cycles
        found_cycles = False
        for t in time_period_list:
            cycles = find_flow_cycles(x, t)
            if cycles:
                found_cycles = True
                print(f"Flow cycle found in period {t}:")
                cycle_str = []
                for i, j in cycles:
                    flow_sum = sum(
                        x[i, j, c, t, v].x for c in commodity_list for v in vehicle_list if x[i, j, c, t, v].x > 0)
                    cycle_str.append(f"{i} --({flow_sum:.2f})--> {j}")
                print(" ".join(cycle_str))

                # Add subtour elimination constraint
                model.addConstr(
                    quicksum(bar_x[i, j, t, v] for (i, j) in cycles for v in vehicle_list)
                    <= len(cycles) - 1
                )

        # Break if no cycles found
        if not found_cycles:
            print("no more cycles")
            break
        else:
            counter = counter + 1

    # Print variable values
    print("\nVariable Values:")
    print("\nFlow variables (x):")
    for (i, j, c, t, v) in x.keys():
        val = x[i, j, c, t, v].x
        if val > 0:
            print(f"x[{i},{j},{c},{t},{v}] = {val}")



    print("\nDemand fulfillment variables (y):")
    for i in node_list:
        for c in commodity_list:
            for t in time_period_list:
                val = y[i, c, t].x
                if val > 0:
                    print(f"y[{i},{c},{t}] = {val}")

    print("\nUnmet demand variables (z):")
    for i in node_list:
        for c in commodity_list:
            for t in time_period_list:
                val = z[i, c, t].x
                if val > 0:
                    print(f"z[{i},{c},{t}] = {val}")

    print("\nInventory variables (w):")
    for i in node_list:
        for c in commodity_list:
            for t in time_period_list2:
                val = w[i, c, t].x
                if val > 0:
                    print(f"w[{i},{c},{t}] = {val}")

    print("\nSafety stock shortage variables (alpha):")

    for c in commodity_list:
        val = alpha[c].x
        if val > 0:
            print(f"alpha[{c}] = {val}")

    print("\nInitial inventory variables (q):")
    for i in node_list:
        for c in commodity_list:
            val = q[i, c].x
            if val > 0:
                print(f"q[{i},{c}] = {val}")

    print("\nAPS location indicators (p):")
    for i in node_list:
        val = p[i].x
        if val > 0:
            print(f"p[{i}] = {val}")

    print("\nInventory location indicators (r):")
    for i in node_list:
        for c in commodity_list:
            val = r[i, c].x
            if val > 0:
                print(f"r[{i},{c}] = {val}")

    print("\nVehicle assignments (m_i):")
    for i in node_list:
        val = m_i[i].x
        if val > 0:
            print(f"m_i[{i}] = {val}")

    print("\nVehicle flow indicators (bar_x):")
    # for (i, j) in arc_list:
    #     for t in time_period_list:
    #         for v in vehicle_list:
    #             val = bar_x[i, j, t, v].x
    #             if val > 0:
    #                 print(f"bar_x[{i},{j},{t},{v}] = {val}")

    print("\nVehicle location indicators (bar_w):")
    for i in node_list:
        for t in time_period_list2:
            for v in vehicle_list:
                val = bar_w[i, t, v].x
                if val > 0:
                    print(f"bar_w[{i},{t},{v}] = {val}")

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

    
