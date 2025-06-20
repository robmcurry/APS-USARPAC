# Import required libraries
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
import networkx as nx
import time
from typing import Dict, Set, List


def solve_deterministic_vrp_with_aps_single_stage(scenario, vehicle_list=None, P_max=3, M=3000):
    # Start timing execution
    start_time = time.time()

    # Initialize vehicle list if none provided
    if vehicle_list is None:
        vehicle_list = list(range(1, 6))

    # Extract node list from scenario demand
    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))

    # Add dummy leftover node for excess inventory
    dummy_leftover_node = max(node_list) + 1
    node_list.append(dummy_leftover_node)

    # Add dummy deficit node for unmet demand
    dummy_deficit_node = max(node_list) + 1
    node_list.append(dummy_deficit_node)

    # Extract commodity list and arc list from scenario
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))
    arc_list = sorted(set(arc_list))

    # Initialize demand and capacity dictionaries 
    d = {(int(i), c): 0 for i in node_list for c in commodity_list}
    d = {(int(i), c): int(scenario['demand'][(i, c)]) for (i, c) in scenario['demand']}
    cap = {(int(i), int(j), c): scenario['capacity'][(i, j, c)] for (i, j, c) in scenario['capacity']}

    # Degradation levels at node i for commodity c. We do want this to be scenario based at some point.
    g = {(int(i), c): .25 for i in node_list for c in commodity_list}

    # Set safety stock requirements
    required_safety_stock = {c: 1000 for c in commodity_list}

    # Redundancy parameter
    L = {c: 2 for c in commodity_list}

    # objective weights
    weight = {}
    weight['deficit'] = 100
    weight['shortfall'] = 0

    # maximum number of vehicles
    max_num_vehicles = 300

    # Initialize optimization model
    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

    # Old Variables
    # # Initial inventory level at each node for each commodity
    # y = model.addVars(node_list, commodity_list, name="y", lb=0)
    # # Variables tracking unmet demand at each node for each commodity
    # z = model.addVars(node_list, commodity_list, name="z", lb=0)
    # # Variables tracking excess inventory at each node for each commodity
    # w = model.addVars(node_list, commodity_list, name="w", lb=0)
    # Binary variables for vehicle usage
    # m_i = model.addVars(vehicle_list, vtype=GRB.BINARY, name="m_i", lb=0)

    # Define decision variables
    # Flow variables for each arc, commodity, and node
    x = model.addVars(arc_list, commodity_list, node_list, vtype=GRB.INTEGER, name="x", lb=0)
    # Variables tracking safety stock shortage for each commodity
    alpha = model.addVars(commodity_list, name="alpha", lb=0)
    # Initial inventory quantities at each node for each commodity
    q = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="q", lb=0)
    # real scenario-based inventory quantities at each node for each commodity
    bar_q = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="q", lb=0)
    # Binary variables for node-to-facility assignments
    f = model.addVars(node_list, node_list, vtype=GRB.BINARY, name="nodefacilityassignment")
    # Binary variables indicating if commodity stored at node
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    # Binary variables indicating if node is an APS facility
    p = model.addVars(node_list, ub=1, lb=0, name="p")

    # Total flow on each arc for each commodity
    total_flow = model.addVars(arc_list, commodity_list, lb=0, name="totalflow")
    # Flow variables for spanning tree per facility
    tree_flow = model.addVars(arc_list, node_list, lb=0, name="treeflows")
    # Binary variables indicating if arc used in spanning tree
    bar_x = model.addVars(arc_list, node_list, vtype=GRB.BINARY, name="bar_x")
    # Variables tracking excess inventory sent to dummy node
    y = model.addVars(node_list, commodity_list, name="leftover")
    # Variables tracking unmet demand sent to dummy node  
    z = model.addVars(node_list, commodity_list, name="deficit")
    #Variable for maximum vehicle distance
    max_vehicle_distance = model.addVar(name="max_vehicle_distance")
    #Variable for the distance from that node to the root node for its given tree
    dist = model.addVars(node_list, node_list, lb=0, name="distance_var")
    # max tree flow
    max_tree_flow = model.addVars(node_list, name="max_tree_flow")
    # variable signifying a leaf node
    leaf_node_var = model.addVars(node_list, node_list,vtype=GRB.BINARY, name="leaf_node_var")

    # Define objective function
    model.setObjective(
        weight['deficit']*quicksum(z[i, c] for i in node_list for c in commodity_list)
        + weight['shortfall']*quicksum(alpha[c] for c in commodity_list),
        GRB.MINIMIZE
    )

    # CONSTRAINTS
    # Flow conservation constraint for each node and commodity
    model.addConstrs((
        quicksum(tree_flow[i, j, k] for i in node_list if (i, j) in arc_list)
        - quicksum(tree_flow[j, i, k] for i in node_list if (j, i) in arc_list)
        == f[j, k]
        for k in node_list
        for j in node_list if j != k
    ), name="facility_flow_conservation")

    # Constraint that captures the maximum sized vehicle
    model.addConstrs((
        max_tree_flow[k] >= tree_flow[i,j,k]
        for (i,j) in arc_list
        for k in node_list
    ), name="max_tree_flow_constraint")

    model.addConstrs((
        bar_q[i,c] == g[i,c]*q[i,c]
        for i in node_list
        for c in commodity_list
    ), name="max_tree_flow_constraint")

    # Constraint that determines the distance variables
    # model.addConstrs((
    #     dist[j, k] >= dist[i, k] + bar_x[i, j, k] - len(node_list) * (bar_x[j, i, k])
    #     for (i, j) in arc_list
    #     for k in node_list
    # ), name="distance_constraint")


    model.addConstrs((
        # The constraint ensures that a node j is a leaf node in facility k's tree if it has no outgoing flow
        # Left side: leaf_node_var[j,k] indicates if node j is a leaf node for facility k  
        # Right side: 
        # - 2 minus sum of incoming flows to node j from facility k 
        # - Large term to force the constraint to hold only when j is assigned to facility k
        leaf_node_var[j, k] >= 2 - quicksum(tree_flow[i, j, k] for i in node_list if (i, j) in arc_list)
        - len(node_list)*(1 - f[j, k])
        for j in node_list
        for k in node_list
        if j!=k
    ), name="leafnodeconstraint")

    model.addConstr((
        quicksum(leaf_node_var[j,k] for j in node_list for k in node_list) <=
        max_num_vehicles
    ),name="maximum_num_vehicles_constraint")


    # Total flow for each arc and commodity across all facilities
    model.addConstrs((
        total_flow[i, j, c] == quicksum(x[i, j, c, k] for k in node_list)
        for (i, j) in arc_list
        for c in commodity_list
        ), name="total_flow_calculation")

    # Upper bound on tree flow based on number of nodes
    model.addConstrs((
        tree_flow[i, j, k] <= (len(node_list) - 1) * bar_x[i, j, k]
        for (i, j) in arc_list
        for k in node_list
    ), name="tree_flow_upper_bound")

    # Prevent bidirectional flow between nodes for each facility
    model.addConstrs((
        bar_x[i, j, k] + bar_x[j, i, k] <= 1
        for k in node_list
        for (i, j) in arc_list if (j, i) in arc_list
        ), name="prevent_bidirectional_flow")

    # Maximum one incoming arc per node per facility
    model.addConstrs((
        quicksum(bar_x[i, j, k] for i in node_list if (i, j) in arc_list) <= 1
        for k in node_list
        for j in node_list
    ), name="max_one_incoming_arc")

    # Minimum tree flow requirement
    model.addConstrs((
        tree_flow[i, j, k] >= bar_x[i, j, k]
        for (i, j) in arc_list
        for k in node_list
    ), name="min_tree_flow")

    # Link arc usage to source facility assignment
    model.addConstrs((
        bar_x[i, j, k] <= f[i, k]
        for (i, j) in arc_list
        for k in node_list
    ), name="arc_source_facility_link")

    # Link arc usage to destination facility assignment
    model.addConstrs((
        bar_x[i, j, k] <= f[j, k]
        for (i, j) in arc_list
        for k in node_list
    ), name="arc_dest_facility_link")

    # Each node must be assigned to at least one facility
    model.addConstrs((
        quicksum(f[i, k] for k in node_list) >= 1
        for i in node_list
    ), name="node_facility_assignment")

    # Require incoming arc for each node assigned to a facility
    model.addConstrs((
        quicksum(bar_x[j, i, k] for j in node_list if (j, i) in arc_list) >= f[i, k]
        for i in node_list
        for k in node_list if i != k
    ), name="facility_connectivity")

    # Flow balance at facility node
    model.addConstrs((
        quicksum(tree_flow[k, j, k] for j in node_list if (k, j) in arc_list)
        == quicksum(f[j, k] for j in node_list) - 1
        for k in node_list
    ), name="facility_flow_balance")

    # Link commodity flow to arc usage
    model.addConstrs((
        x[i, j, c, k]  <= cap[i,j,c] * bar_x[i, j, k]
        for (i, j) in arc_list
        for k in node_list
        for c in commodity_list
    ), name="commodity_flow_link")

    # Define maximum distance per vehicle
    MAX_DISTANCE_PER_VEHICLE = 10000

    # Flow conservation constraints
    model.addConstrs(
        (bar_q[i, c] + quicksum(total_flow[j, i, c] for j in node_list if (j, i) in arc_list) + z[i, c] ==
         d.get((i, c), 0) + quicksum(total_flow[i, j, c] for j in node_list if (i, j) in arc_list) + y[i, c]
         for i in node_list
         for c in commodity_list
         ), name="FlowConservationConstraint")

    # Constraint to ensure flow only occurs through APS facilities
    model.addConstrs((
        bar_x[i, j, k] <= p[k]
        for (i, j) in arc_list
        for k in node_list
    ), name="APSFlowRestriction")

    # Safety stock constraints
    model.addConstrs((
        alpha[c] >= required_safety_stock[c] - quicksum(y[i, c] for i in node_list)
        for c in commodity_list
    ), name="SafetyStock")

    # Ensures minimum number L[c] of commodity c storage locations
    model.addConstrs((
        quicksum(r[i, c] for i in node_list) >= L[c]
        for c in commodity_list
    ), name="CommodityRedundancy")

    # Variable linking constraints
    # Constraint linking quantity stored (q) to binary variable (r) indicating if commodity c stored at node i
    model.addConstrs((
        q[i, c] <= M * r[i, c]
        for i in node_list
        for c in commodity_list
    ), name="q_r_link")

    # Constraint linking binary variable (r) indicating commodity storage to binary variable (p) indicating APS facility
    model.addConstrs((
        r[i, c] <= p[i]
        for i in node_list
        for c in commodity_list
    ), name="r_p_link")

    # Constraint limiting total number of APS facilities to P_max
    model.addConstr(
        quicksum(p[i] for i in node_list) <= P_max,
        name="Max_APS_Locations")

    counter = 1

    print("Solve ", counter)
    # Enable Gurobi node logging
    model.setParam("OutputFlag", 1)
    model.setParam("LogToConsole", 1)
    model.optimize()

    # Print variable values
    print("\nVariable Values:")
    print("\nTree flow variables:")
    for k in node_list:
        if p[k].x > 0:  # Only show for APS nodes
            print(f"\nTree flows for APS at node {k}:")
            for (i, j) in arc_list:
                if tree_flow[i, j, k].x > 0:
                    print(f"tree_flow[{i},{j},{k}] = {tree_flow[i, j, k].x:.2f}")


    print("\nDistance label values (d):")
    for i in node_list:
        for k in node_list:
            val = dist[i,k].x
            if val > 0:
                print(f"dist[{i},{k}] = {val:.2f}")


    print("\nLeaf node variables:")
    for j in node_list:
        for k in node_list:
            if leaf_node_var[j, k].x > 0:
                print(f"leaf_node_var[{j},{k}] = {leaf_node_var[j, k].x:.2f}")

    # Print positive flow variables
    print("\nPositive flow variables (x):")
    for k in node_list:
        for (i, j) in arc_list:
            for c in commodity_list:
                if x[i, j, c, k].x > 0.001:
                    print(f"x[{i},{j},{c},{k}] = {x[i, j, c, k].x:.2f}")

    parent = {}
    # Build flow network for each APS 
    for k in node_list:
        if not p[k].x > 0:  # Skip if not an APS
            continue

    value = 0
    print("\nUnmet demand variables (z):")
    for i in node_list:
        for c in commodity_list:
            val = z[i, c].x
            if val > 0:
                print(f"deficit[{i},{c}] = {val}")

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

    print("\nReal inventory variables (q):")
    for i in node_list:
        for c in commodity_list:
            val = bar_q[i, c].x
            if val > 0:
                print(f"bar_q[{i},{c}] = {val}")

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

    print("\nVehicle flow indicators (bar_x):")
    for k in node_list:
        for (i, j) in arc_list:
            val = bar_x[i, j, k].x
            if val > 0:
                print(f"bar_x[{i},{j},{k}] = {val:.2f}")

    flow_summary = defaultdict(float)
    for (i, j, c, k) in x.keys():
        val = x[i, j, c, k].x
        if val > 0:
            flow_summary[(i, j)] += val

    # Calculate computational time
    computational_time = time.time() - start_time

    # Collect and return results
    results = {
        "scenario_id": scenario["scenario_id"],
        "objective": model.ObjVal if model.Status == GRB.OPTIMAL else None,
        "aps_locations": [i for i in node_list if p[i].x > 0.5],
        "num_q_positive": sum(1 for i in node_list for c in commodity_list if q[i, c].x > 0),
        "num_initial_inventory": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "total_flow": sum(flow_summary.values()),
        "status": model.Status,
        "q": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "q_full": {(i, c): q[i, c].x for i in node_list for c in commodity_list},
        "initial_inventory": {(i, c): q[i, c]},
        "w_full": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "flow_summary": dict(flow_summary),
        "computational_time": computational_time
    }

    print(f"\nComputational time: {computational_time:.2f} seconds")
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
    print("Computational Time:", result["computational_time"], "seconds")
