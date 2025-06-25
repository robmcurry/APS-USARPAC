# Import required libraries
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
import networkx as nx
import time
from typing import Dict, Set, List


    # def print_model_parameters(scenario, vehicle_list, P_max, M, required_safety_stock, L, weight, max_num_vehicles):
    #
def solve_deterministic_vrp_with_aps_single_stage(scenario, vehicle_list=None, P_max=7, M=9900,
                                                  ):
    # Start timing execution
    start_time = time.time()

    # Extract node list from scenario demand
    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))

    # Initialize region mapping dictionary
    region_mapping = {node: 0 for node in node_list}
    # Add nodes 1 through 10 to region 1
    # Nodes 11 through 20 in region 2
    # nodes 21 through 41 in region 3
    for node in node_list:
        if node < 11:
            region_mapping[node] = 1
        elif node < 21:
            region_mapping[node] = 2
        else:
            region_mapping[node] = 3

    print(region_mapping)
    # Extract commodity list and arc list from scenario
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))
    arc_list = sorted(set(arc_list))

    # Initialize demand and capacity dictionaries
    d = {(int(i), c): 0 for i in node_list for c in commodity_list}
    d = {(int(i), c): int(scenario['demand'][(i, c)]) for (i, c) in scenario['demand']}
    cap = {(int(i), int(j), c): int(scenario['capacity'][(i, j, c)]) for (i, j, c) in scenario['capacity']}

    # commented redundancy dictionary initialization
    redundancy = {int(i): 2 for i in node_list}  # Defines redundancy requirements for each node

    # Degradation levels at node i for commodity c. We do want this to be scenario based at some point.
    g = {(int(i), c): 1 for i in node_list for c in commodity_list}

    # Set safety stock requirements
    required_safety_stock = {c: 1000 for c in commodity_list}

    # Redundancy parameter
    L = {c: 2 for c in commodity_list}

    # The minimum proportion of remaining stock within each region
    region_proportions = {1: 0.6, 2: 0.1, 3: 0.1}

    # objective weights
    weight = {}
    weight['deficit'] = .3
    weight['shortfall'] = .3
    weight['balance'] = .4

    # maximum number of vehicles
    max_num_vehicles = 300

    # Initialize optimization model
    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

    # Define decision variables
    # Flow variables for each arc, commodity, and node
    x = model.addVars(arc_list, commodity_list, node_list, vtype=GRB.INTEGER, name="x", lb=0)
    # Variables tracking safety stock shortage for each commodity
    alpha = model.addVars(commodity_list, vtype=GRB.INTEGER, name="alpha", lb=0)
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
    total_flow = model.addVars(arc_list, commodity_list, vtype=GRB.INTEGER, lb=0, name="totalflow")
    # Flow variables for spanning tree per facility
    tree_flow = model.addVars(arc_list, node_list, vtype=GRB.INTEGER, lb=0, name="treeflows")
    # Binary variables indicating if arc used in spanning tree
    bar_x = model.addVars(arc_list, node_list, vtype=GRB.BINARY, name="bar_x")
    # Variables tracking excess inventory sent to dummy node
    y = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="leftover")
    # Variables tracking unmet demand sent to dummy node  
    z = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="deficit")
    # max tree flow
    max_tree_flow = model.addVars(node_list, name="max_tree_flow")
    # variable signifying a leaf node
    leaf_node_var = model.addVars(node_list, node_list,vtype=GRB.BINARY, name="leaf_node_var")
    # Variable for maximum difference in safety stock among all remaining APS locations
    ss_balance = model.addVar(lb=0, vtype=GRB.INTEGER,name='safety stock balance')
    # Define objective function
    model.setObjective(
        weight['deficit']*quicksum(z[i, c] for i in node_list for c in commodity_list)
        + weight['shortfall']*quicksum(alpha[c] for c in commodity_list)
        + weight['balance']*ss_balance,
        GRB.MINIMIZE
    )

    # CONSTRAINTS
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

    # Constraint to ensure flow only occurs through APS facilities
    model.addConstrs((
        bar_x[i, j, k] <= p[k]
        for (i, j) in arc_list
        for k in node_list
    ), name="APSFlowRestriction")

    # Flow conservation constraint for each node and commodity
    model.addConstrs((
        quicksum(tree_flow[i, j, k] for i in node_list if (i, j) in arc_list)
        - quicksum(tree_flow[j, i, k] for i in node_list if (j, i) in arc_list)
        == f[j, k]
        for k in node_list
        for j in node_list if j != k
    ), name="facility_flow_conservation")

    model.addConstrs((
        bar_q[i,c] == g[i,c]*q[i,c]
        for i in node_list
        for c in commodity_list
    ), name="max_tree_flow_constraint")

    model.addConstrs((
        # The constraint ensures that a node j is a leaf node in facility k's tree if it has no outgoing flow
        # Left side: leaf_node_var[j,k] indicates if node j is a leaf node for facility k
        # Right side:
        # - 2 minus sum of incoming flows to node j from facility k
        # - Large term to force the constraint to hold only when j is assigned to facility k
        leaf_node_var[j, k] >= 2 - quicksum(tree_flow[i, j, k] for i in node_list if (i, j) in arc_list)
        - len(node_list) * (1 - f[j, k])
        for j in node_list
        for k in node_list
        if j != k
    ), name="leafnodeconstraint")

    # Constraint to enforce maximum number of vehicles across all facilities (leaf nodes)
    model.addConstr((
            quicksum(leaf_node_var[j, k] for j in node_list for k in node_list) <=
            max_num_vehicles
    ), name="maximum_num_vehicles_constraint")


    # Total flow for each arc and commodity across all facilities
    model.addConstrs((
        total_flow[i, j, c] == quicksum(x[i, j, c, k] for k in node_list)
        for (i, j) in arc_list
        for c in commodity_list
        ), name="total_flow_calculation")

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
        quicksum(f[i, k] for k in node_list) >= redundancy[i]
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

    # Flow conservation constraints
    model.addConstrs(
        (bar_q[i, c] + quicksum(total_flow[j, i, c] for j in node_list if (j, i) in arc_list) + z[i, c] ==
         d.get((i, c), 0) + quicksum(total_flow[i, j, c] for j in node_list if (i, j) in arc_list) + y[i, c]
         for i in node_list
         for c in commodity_list
         ), name="FlowConservationConstraint")


    # Safety stock constraints
    model.addConstrs((
        alpha[c] >= required_safety_stock[c] - quicksum(y[i, c] for i in node_list)
        for c in commodity_list
    ), name="SafetyStock")


    # Captures the maximum difference between remaining stocks at nodes i and k
    model.addConstrs((
        ss_balance >= y[i,c] - y[k,c]
        for i in node_list
        for k in node_list
        for c in commodity_list
    ),name='safetystockbalance')

    # Regional distribution constraints for y-variables
    model.addConstrs((
        quicksum(y[i, c] for i in node_list if region_mapping[i] == r) >=
        region_proportions[r] * quicksum(y[i, c] for i in node_list)
        for r in region_proportions.keys()
        for c in commodity_list
    ), name='regional_distribution')

    counter = 1

    print("Solve ", counter)
    # Enable Gurobi node logging
    model.setParam("OutputFlag", 1)
    model.setParam("LogToConsole", 1)
    model.optimize()

    # Print variable values
    print("\nVariable Values:")
    print("\nTree flow variables:")
    # for k in node_list:
    #     if p[k].x > 0:  # Only show for APS nodes
    #         print(f"\nTree flows for APS at node {k}:")
    #         for (i, j) in arc_list:
    #             if tree_flow[i, j, k].x > 0:
    #                 print(f"tree_flow[{i},{j},{k}] = {tree_flow[i, j, k].x:.2f}")
    #
    # print("\nLeaf node variables:")
    # for j in node_list:
    #     for k in node_list:
    #         if leaf_node_var[j, k].x > 0:
    #             print(f"leaf_node_var[{j},{k}] = {leaf_node_var[j, k].x:.2f}")
    #
    # # Print positive flow variables
    # print("\nPositive flow variables (x):")
    for k in node_list:
        for (i, j) in arc_list:
            for c in commodity_list:
                if x[i, j, c, k].x > 0.001:
                    print(f"x[{i},{j},{c},{k}] = {x[i, j, c, k].x:.2f}")
    #
    # parent = {}
    # # Build flow network for each APS
    # for k in node_list:
    #     if not p[k].x > 0:  # Skip if not an APS
    #         continue
    #
    # value = 0
    # print("\nUnmet demand variables (z):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = z[i, c].x
    #         if val > 0:
    #             print(f"deficit[{i},{c}] = {val}")
    #
    print("\nSafety stock shortage variables (alpha):")
    for c in commodity_list:
        val = alpha[c].x
        if val > 0:
            print(f"alpha[{c}] = {val}")
    #
    # print("\nInitial inventory variables (q):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = q[i, c].x
    #         if val > 0:
    #             print(f"q[{i},{c}] = {val}")
    #
    # print("\nReal inventory variables (q):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = bar_q[i, c].x
    #         if val > 0:
    #             print(f"bar_q[{i},{c}] = {val}")
    #
    # print("\nAPS location indicators (p):")
    # for i in node_list:
    #     val = p[i].x
    #     if val > 0:
    #         print(f"p[{i}] = {val}")
    #
    # print("\nInventory location indicators (r):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = r[i, c].x
    #         if val > 0:
    #             print(f"r[{i},{c}] = {val}")

    print("\nVehicle flow indicators (bar_x):")
    for k in node_list:
        for (i, j) in arc_list:
            val = bar_x[i, j, k].x
            if val > 0:
                print(f"bar_x[{i},{j},{k}] = {val:.2f}")

    flow_summary = defaultdict(float)
    # for (i, j, c, k) in x.keys():
    #     val = x[i, j, c, k].x
    #     if val > 0:
    #         flow_summary[(i, j)] += val

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
        "existing_inventory": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "initial_inventory": {(i, c): q[i, c]},
        "w_full": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
        "flow_summary": dict(flow_summary),
        "computational_time": computational_time
    }

    print(f"\nComputational time: {computational_time:.2f} seconds")
    return results


def solve_deterministic_vrp_with_aps_single_stage_commodity(scenario, vehicle_list=None, P_max=7, M=9900):
    # Start timing execution
    start_time = time.time()

    # Extract node list from scenario demand
    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))

    # Extract commodity list and arc list from scenario
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))
    arc_list = sorted(set(arc_list))

    # Initialize demand and capacity dictionaries
    d = {(int(i), c): 0 for i in node_list for c in commodity_list}
    d = {(int(i), c): int(scenario['demand'][(i, c)]) for (i, c) in scenario['demand']}
    cap = {(int(i), int(j), c): int(scenario['capacity'][(i, j, c)]) for (i, j, c) in scenario['capacity']}

    redundancy = {int(i): 2 for i in node_list}

    # Initialize region mapping dictionary
    region_mapping = {node: 0 for node in node_list}
    # Add nodes 1 through 10 to region 1
    # Nodes 11 through 20 in region 2
    # nodes 21 through 41 in region 3
    for node in node_list:
        if node < 11:
            region_mapping[node] = 1
        elif node < 21:
            region_mapping[node] = 2
        else:
            region_mapping[node] = 3

    # Degradation levels at node i for commodity c. We do want this to be scenario based at some point.
    g = {(int(i), c): 1 for i in node_list for c in commodity_list}

    # Set safety stock requirements
    required_safety_stock = {c: 1000 for c in commodity_list}

    # Redundancy parameter
    L = {c: 2 for c in commodity_list}

    # The minimum proportion of remaining stock within each region
    region_proportions = {1: 0.6, 2: 0.1, 3: 0.1}

    # objective weights
    weight = {}
    weight['deficit'] = 1000000
    weight['shortfall'] = 1
    weight['balance'] = .4

    # maximum number of vehicles
    max_num_vehicles = 300

    # Initialize optimization model
    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

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
    f = model.addVars(node_list, commodity_list, node_list, vtype=GRB.BINARY, name="nodefacilityassignment")
    # Binary variables indicating if commodity stored at node
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    # Binary variables indicating if node is an APS facility
    p = model.addVars(node_list, ub=1, lb=0, name="p")

    # Total flow on each arc for each commodity
    total_flow = model.addVars(arc_list, commodity_list, lb=0, name="totalflow")
    # Flow variables for spanning tree per facility
    tree_flow = model.addVars(arc_list, commodity_list, node_list, lb=0, name="treeflows")
    # Binary variables indicating if arc used in spanning tree
    bar_x = model.addVars(arc_list, commodity_list, node_list, vtype=GRB.BINARY, name="bar_x")
    # Variables tracking excess inventory sent to dummy node
    y = model.addVars(node_list, commodity_list, name="leftover")
    # Variables tracking unmet demand sent to dummy node
    z = model.addVars(node_list, commodity_list, name="deficit")
    # # max tree flow
    # max_tree_flow = model.addVars(node_list, name="max_tree_flow")
    # variable signifying a leaf node
    leaf_node_var = model.addVars(node_list, commodity_list, node_list, vtype=GRB.BINARY, name="leaf_node_var")

    # Define objective function
    model.setObjective(
        weight['deficit'] * quicksum(z[i, c] for i in node_list for c in commodity_list)
        + weight['shortfall'] * quicksum(alpha[c] for c in commodity_list),
        GRB.MINIMIZE
    )
    # Should we include some measure of "balance"??

    # CONSTRAINTS
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

    # Constraint to ensure flow only occurs through APS facilities
    model.addConstrs((
        bar_x[i, j, c, k] <= p[k]
        for (i, j) in arc_list
        for c in commodity_list
        for k in node_list
    ), name="APSFlowRestriction")

    # Flow conservation constraint for each node and commodity
    model.addConstrs((
        quicksum(tree_flow[i, j, c, k] for i in node_list if (i, j) in arc_list)
        - quicksum(tree_flow[j, i, c, k] for i in node_list if (j, i) in arc_list)
        == f[j, c, k]
        for k in node_list
        for c in commodity_list
        for j in node_list if j != k
    ), name="facility_flow_conservation")

    # Constraint that captures the maximum sized vehicle
    # model.addConstrs((
    #     max_tree_flow[k] >= tree_flow[i, j, c, k]
    #     for (i, j) in arc_list
    #     for k in node_list
    #     for c in commodity_list
    # ), name="max_tree_flow_constraint")

    model.addConstrs((
        bar_q[i, c] == g[i, c] * q[i, c]
        for i in node_list
        for c in commodity_list
    ), name="max_tree_flow_constraint")

    model.addConstrs((
        # The constraint ensures that a node j is a leaf node in facility k's tree if it has no outgoing flow
        # Left side: leaf_node_var[j,k] indicates if node j is a leaf node for facility k
        # Right side:
        # - 2 minus sum of incoming flows to node j from facility k
        # - Large term to force the constraint to hold only when j is assigned to facility k
        leaf_node_var[j, c, k] >= 2 - quicksum(tree_flow[i, j, c, k] for i in node_list if (i, j) in arc_list)
        - len(node_list) * (1 - f[j, c, k])
        for j in node_list
        for c in commodity_list
        for k in node_list
        if j != k
    ), name="leafnodeconstraint")

    # Constraint to enforce maximum number of vehicles across all facilities (leaf nodes)
    model.addConstr((
            quicksum(leaf_node_var[j, c, k] for j in node_list for k in node_list for c in commodity_list) <=
            max_num_vehicles
    ), name="maximum_num_vehicles_constraint")

    # Total flow for each arc and commodity across all facilities
    model.addConstrs((
        total_flow[i, j, c] == quicksum(x[i, j, c, k] for k in node_list)
        for (i, j) in arc_list
        for c in commodity_list
    ), name="total_flow_calculation")

    # Prevent bidirectional flow between nodes for each facility
    model.addConstrs((
        bar_x[i, j, c, k] + bar_x[j, i, c, k] <= 1
        for k in node_list
        for c in commodity_list
        for (i, j) in arc_list if (j, i) in arc_list
    ), name="prevent_bidirectional_flow")

    # Maximum one incoming arc per node per facility
    model.addConstrs((
        quicksum(bar_x[i, j, c, k] for i in node_list if (i, j) in arc_list) <= 1
        for k in node_list
        for c in commodity_list
        for j in node_list
    ), name="max_one_incoming_arc")

    # Minimum tree flow requirement
    model.addConstrs((
        tree_flow[i, j, c, k] >= bar_x[i, j, c, k]
        for (i, j) in arc_list
        for c in commodity_list
        for k in node_list
    ), name="min_tree_flow")

    # Link arc usage to source facility assignment
    model.addConstrs((
        bar_x[i, j, c, k] <= f[i, c, k]
        for (i, j) in arc_list
        for c in commodity_list
        for k in node_list
    ), name="arc_source_facility_link")

    # Link arc usage to destination facility assignment
    model.addConstrs((
        bar_x[i, j, c, k] <= f[j, c, k]
        for (i, j) in arc_list
        for c in commodity_list
        for k in node_list
    ), name="arc_dest_facility_link")

    # Each node must be assigned to at least one facility
    model.addConstrs((
        quicksum(f[i, c, k] for k in node_list) >= redundancy[i]
        for i in node_list
        for c in commodity_list
    ), name="node_facility_assignment")

    # Require incoming arc for each node assigned to a facility
    model.addConstrs((
        quicksum(bar_x[j, i, c, k] for j in node_list if (j, i) in arc_list) >= f[i, c, k]
        for i in node_list
        for c in commodity_list
        for k in node_list if i != k
    ), name="facility_connectivity")

    # Flow balance at facility node
    model.addConstrs((
        quicksum(tree_flow[k, j, c, k] for j in node_list if (k, j) in arc_list)
        == quicksum(f[j, c, k] for j in node_list) - 1
        for k in node_list
        for c in commodity_list
    ), name="facility_flow_balance")

    # Link commodity flow to arc usage
    model.addConstrs((
        x[i, j, c, k] <= cap[i, j, c] * bar_x[i, j, c, k]
        for (i, j) in arc_list
        for k in node_list
        for c in commodity_list
    ), name="commodity_flow_link")

    # Flow conservation constraints
    model.addConstrs(
        (bar_q[i, c] + quicksum(total_flow[j, i, c] for j in node_list if (j, i) in arc_list) + z[i, c] ==
         d.get((i, c), 0) + quicksum(total_flow[i, j, c] for j in node_list if (i, j) in arc_list) + y[i, c]
         for i in node_list
         for c in commodity_list
         ), name="FlowConservationConstraint")

    # Safety stock constraints
    model.addConstrs((
        alpha[c] >= required_safety_stock[c] - quicksum(y[i, c] for i in node_list)
        for c in commodity_list
    ), name="SafetyStock")

    # Captures the maximum difference between remaining stocks at nodes i and k
    model.addConstrs((
        ss_balance >= y[i, c] - y[k, c]
        for i in node_list
        for k in node_list
        for c in commodity_list
    ), name='safetystockbalance')

    # Define variable for tracking safety stock shortage for each commodity 
    # alpha[c] represents the shortage amount below required safety stock level for commodity c
    # Lower bound of 0 means shortages can't be negative
    alpha = model.addVars(commodity_list, name="alpha", lb=0)

    # Define variable for storing initial inventory quantities for each node-commodity pair
    # q[i,c] represents the inventory of commodity c stored at node i
    # Must be integer values with lower bound of 0 (can't have negative inventory)
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
                for c in commodity_list:
                    if tree_flow[i, j, c, k].x > 0:
                        print(f"tree_flow[{i},{j},{c},{k}] = {tree_flow[i, j, c, k].x:.2f}")
    #
    # print("\nLeaf node variables:")
    # for j in node_list:
    #     for k in node_list:
    #         if leaf_node_var[j, k].x > 0:
    #             print(f"leaf_node_var[{j},{k}] = {leaf_node_var[j, k].x:.2f}")
    #
    # # Print positive flow variables
    print("\nPositive flow variables (x):")
    for k in node_list:
        for (i, j) in arc_list:
            for c in commodity_list:
                if x[i, j, c, k].x > 0.001:
                    print(f"x[{i},{j},{c},{k}] = {x[i, j, c, k].x:.2f}")
    #
    # parent = {}
    # # Build flow network for each APS
    # for k in node_list:
    #     if not p[k].x > 0:  # Skip if not an APS
    #         continue
    #
    # value = 0
    print("\nUnmet demand variables (z):")
    for i in node_list:
        for c in commodity_list:
            val = z[i, c].x
            if val > 0:
                print(f"deficit[{i},{c}] = {val}")
    #
    print("\nSafety stock shortage variables (alpha):")
    for c in commodity_list:
        val = alpha[c].x
        if val > 0:
            print(f"alpha[{c}] = {val}")
    #
    # print("\nInitial inventory variables (q):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = q[i, c].x
    #         if val > 0:
    #             print(f"q[{i},{c}] = {val}")
    #
    # print("\nReal inventory variables (q):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = bar_q[i, c].x
    #         if val > 0:
    #             print(f"bar_q[{i},{c}] = {val}")
    #
    # print("\nAPS location indicators (p):")
    # for i in node_list:
    #     val = p[i].x
    #     if val > 0:
    #         print(f"p[{i}] = {val}")
    #
    # print("\nInventory location indicators (r):")
    # for i in node_list:
    #     for c in commodity_list:
    #         val = r[i, c].x
    #         if val > 0:
    #             print(f"r[{i},{c}] = {val}")

    print("\nVehicle flow indicators (bar_x):")
    for k in node_list:
        for (i, j) in arc_list:
            for c in commodity_list:
                val = bar_x[i, j,c, k].x
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
        "existing_inventory": {(i, c): q[i, c].x for i in node_list for c in commodity_list if q[i, c].x > 0},
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
