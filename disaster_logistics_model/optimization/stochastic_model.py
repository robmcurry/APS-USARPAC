
# Import required libraries
from gurobipy import Model, GRB, quicksum
from collections import defaultdict
import networkx as nx
import time

def solve_deterministic_vrp_with_aps_single_stage(scenario, vehicle_list=None, P_max=7, M=9900,):
    # Extract node list from scenario demand
    node_list = sorted(set(int(i) for (i, _) in scenario['demand']))

    # Extract commodity list and arc list from scenario
    commodity_list = sorted(set(c for (_, c) in scenario['demand']))
    arc_list = sorted(set((int(i), int(j)) for (i, j, c) in scenario['capacity']))
    arc_list = sorted(set(arc_list))

    num_scenarios = 100
    redundancy = {int(i): 2 for i in node_list}  # Defines redundancy requirements for each node

    scenario_list = list(range(1, num_scenarios + 1))
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

    # The minimum proportion of remaining stock within each region
    region_proportions = {1: 0.4, 2: 0.1, 3: 0.1}
    # Initialize scenario parameters
    scenarios = {s: {} for s in scenario_list}  # S is number of scenarios 
    probability = {s: 1 / S for s in scenario_list}  # Equal probability for each scenario

    # Initialize scenario-specific parameters
    scenario_demand = {(i, c, s): scenario['demand'].get((i, c), 0) for i in node_list
                       for c in commodity_list for s in scenario_list}

    scenario_capacity = {(i, j, c, s): scenario['capacity'].get((i, j, c), 0)
                         for (i, j) in arc_list for c in commodity_list for s in scenario_list}

    g = {(i, c, s): .5 for i in node_list for c in commodity_list for s in scenario_list}

    # Initialize optimization model
    model = Model(f"APS_Scenario_{scenario['scenario_id']}")
    model.setParam("OutputFlag", 0)

    # Total flow on each arc for each commodity
    total_flow = model.addVars(arc_list, commodity_list, scenario_list, lb=0, name="totalflow")

    # Modify decision variables to be scenario-dependent
    x = model.addVars(arc_list, commodity_list, node_list, scenario_list,
                      vtype=GRB.INTEGER, name="x", lb=0)

    bar_x = model.addVars(arc_list, commodity_list, node_list, scenario_list,
                      vtype=GRB.BINARY, name="x", lb=0)

    z = model.addVars(node_list, commodity_list, scenario_list, vtype=GRB.INTEGER, name="deficit")
    y = model.addVars(node_list, commodity_list, scenario_list, vtype=GRB.INTEGER, name="leftover")
    bar_q = model.addVars(node_list, commodity_list, scenario_list, name="actual_inventory")
    # Variables tracking safety stock shortage for each commodity
    alpha = model.addVars(commodity_list, scenario_list, vtype=GRB.INTEGER, name="alpha", lb=0)
    ss_balance = model.addVars(scenario_list, vtype=GRB.INTEGER, name="balancemeasure", lb=0)

    # Redundancy parameter
    L = {c: 2 for c in commodity_list}


    q = model.addVars(node_list, commodity_list, vtype=GRB.INTEGER, name="q", lb=0)
    # real scenario-based inventory quantities at each node for each commodity
    bar_q = model.addVars(node_list, commodity_list, scenario_list, vtype=GRB.INTEGER, name="q", lb=0)
    # Binary variables for node-to-facility assignments
    f = model.addVars(node_list, node_list, scenario_list, vtype=GRB.BINARY, name="nodefacilityassignment")
    # Binary variables indicating if commodity stored at node
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    # Binary variables indicating if node is an APS facility
    p = model.addVars(node_list, ub=1, lb=0, name="p")

    # Set safety stock requirements
    required_safety_stock = {c: 1000 for c in commodity_list}


    # maximum number of vehicles
    max_num_vehicles = 300


    # Flow variables for spanning tree per facility
    tree_flow = model.addVars(arc_list, node_list, scenario_list, lb=0, name="treeflows")

    leaf_node_var = model.addVars(node_list, node_list, scenario_list, vtype=GRB.BINARY, name="leaf_node_var")

    # objective weights
    weight = {}
    weight['deficit'] = 1000000
    weight['shortfall'] = 1
    weight['balance'] = .4

    # Update objective function with scenario expectations
    model.setObjective(
        quicksum(probability[s] * (
                weight['deficit'] * quicksum(z[i, c, s] for i in node_list for c in commodity_list)
                + weight['shortfall'] * quicksum(alpha[c,s] for c in commodity_list for s in scenario_list)
                + weight['balance'] * ss_balance)
                 for s in scenario_list),
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
        for s in scenario_list
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
        quicksum(tree_flow[i, j, k,s] for i in node_list if (i, j) in arc_list)
        - quicksum(tree_flow[j, i, k,s] for i in node_list if (j, i) in arc_list)
        == f[j, k]
        for k in node_list
        for j in node_list if j != k
        for s in scenario_list
    ), name="facility_flow_conservation")

    model.addConstrs((
        bar_q[i,c,s] == g[i,c,s]*q[i,c]
        for i in node_list
        for c in commodity_list
        for s in scenario_list
    ), name="max_tree_flow_constraint")

    model.addConstrs((
        # The constraint ensures that a node j is a leaf node in facility k's tree if it has no outgoing flow
        # Left side: leaf_node_var[j,k] indicates if node j is a leaf node for facility k
        # Right side:
        # - 2 minus sum of incoming flows to node j from facility k
        # - Large term to force the constraint to hold only when j is assigned to facility k
        leaf_node_var[j, k,s] >= 2 - quicksum(tree_flow[i, j, k,s] for i in node_list if (i, j) in arc_list)
        - len(node_list) * (1 - f[j, k,s])
        for j in node_list
        for k in node_list
        if j != k
        for s in scenario_list
    ), name="leafnodeconstraint")

    # Constraint to enforce maximum number of vehicles across all facilities (leaf nodes)
    model.addConstrs((
            quicksum(leaf_node_var[j, k,s] for j in node_list for k in node_list) <=
            max_num_vehicles
            for s in scenario_list
    ), name="maximum_num_vehicles_constraint")


    # Total flow for each arc and commodity across all facilities
    model.addConstrs((
        total_flow[i, j, c,s] == quicksum(x[i, j, c, k] for k in node_list)
        for (i, j) in arc_list
        for c in commodity_list
        for s in scenario_list
        ), name="total_flow_calculation")

    # Prevent bidirectional flow between nodes for each facility
    model.addConstrs((
        bar_x[i, j, k,s] + bar_x[j, i, k,s] <= 1
        for k in node_list
        for s in scenario_list
        for (i, j) in arc_list if (j, i) in arc_list
        ), name="prevent_bidirectional_flow")

    # Maximum one incoming arc per node per facility
    model.addConstrs((
        quicksum(bar_x[i, j, k,s] for i in node_list if (i, j) in arc_list) <= 1
        for k in node_list
        for j in node_list
        for s in scenario_list
    ), name="max_one_incoming_arc")

    # Minimum tree flow requirement
    model.addConstrs((
        tree_flow[i, j, k,s] >= bar_x[i, j, k,s]
        for (i, j) in arc_list
        for k in node_list
        for s in scenario_list
    ), name="min_tree_flow")

    # Link arc usage to source facility assignment
    model.addConstrs((
        bar_x[i, j, k,s] <= f[i, k,s]
        for (i, j) in arc_list
        for k in node_list
        for s in scenario_list
    ), name="arc_source_facility_link")

    # Link arc usage to destination facility assignment
    model.addConstrs((
        bar_x[i, j, k,s] <= f[j, k,s]
        for (i, j) in arc_list
        for k in node_list
        for s in scenario_list
    ), name="arc_dest_facility_link")

    # Each node must be assigned to at least one facility
    model.addConstrs((
        quicksum(f[i, k,s] for k in node_list) >= redundancy[i]
        for i in node_list
        for s in scenario_list
    ), name="node_facility_assignment")

    # Require incoming arc for each node assigned to a facility
    model.addConstrs((
        quicksum(bar_x[j, i, k,s] for j in node_list if (j, i) in arc_list) >= f[i, k,s]
        for i in node_list
        for s in scenario_list
        for k in node_list if i != k
    ), name="facility_connectivity")

    # Flow balance at facility node
    model.addConstrs((
        quicksum(tree_flow[k, j, k,s] for j in node_list if (k, j) in arc_list)
        == quicksum(f[j, k,s] for j in node_list) - 1
        for k in node_list
        for s in scenario_list
    ), name="facility_flow_balance")

    # Link commodity flow to arc usage
    model.addConstrs((
        x[i, j, c, k,s]  <= scenario_capacity[i,j,c,s] * bar_x[i, j, k,s]
        for (i, j) in arc_list
        for k in node_list
        for s in scenario_list
        for c in commodity_list
    ), name="commodity_flow_link")

    # Flow conservation constraints
    model.addConstrs(
        (bar_q[i, c,s] + quicksum(total_flow[j, i, c,s] for j in node_list if (j, i) in arc_list) + z[i, c,s] ==
         scenario_demand.get((i, c,s), 0) + quicksum(total_flow[i, j, c,s] for j in node_list if (i, j) in arc_list) + y[i, c,s]
         for i in node_list
         for c in commodity_list
         for s in scenario_list
         ), name="FlowConservationConstraint")


    # Safety stock constraints
    model.addConstrs((
        alpha[c,s] >= required_safety_stock[c] - quicksum(y[i, c,s] for i in node_list)
        for c in commodity_list
        for s in scenario_list
    ), name="SafetyStock")


    # Captures the maximum difference between remaining stocks at nodes i and k
    model.addConstrs((
        ss_balance >= y[i,c,s] - y[k,c,s]
        for i in node_list
        for s in scenario_list
        for k in node_list
        for c in commodity_list
    ),name='safetystockbalance')

    # Regional distribution constraints for y-variables
    model.addConstrs((
        quicksum(y[i, c,s] for i in node_list if region_mapping[i] == r) >=
        region_proportions[r] * quicksum(y[i, c,s] for i in node_list)
        for r in region_proportions.keys()
        for c in commodity_list
        for s in scenario_list
    ), name='regional_distribution')