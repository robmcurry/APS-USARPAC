### Complete Code Expansion
# Below is the continuation of the skeleton and full implementation:
import random
import gurobipy as gp
from gurobipy import GRB


def solve_stochastic_vrp():
    # Data initialization


    num_time_periods = 5 # Number of time periods
    num_nodes = 50  # Number of nodes
    num_vehicles = 30  # Number of vehicles
    num_commodities = 5  # Number of commodities
    num_scenarios = 10  # Number of scenarios
    time_period_list = range(1, num_time_periods)  # Time periods 1 to 10

    time_period_list2 = range(0, num_time_periods)  # Time periods 0 to 10
    vehicle_list = range(1, num_vehicles)  # Vehicles 1 to 10
    # arc_list = random.sample([(i, j) for i in range(1, num_nodes) for j in range(1, num_nodes) if i != j], 30)

    arc_list = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
                (10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19), (18, 20), (19, 21),
                (20, 22), (21, 23), (22, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 1)]

    node_list = list(range(1, num_nodes))  # Nodes 1 to 50
    commodity_list = [f"Commodity{k}" for k in range(1, num_commodities)]  # Commodities 1 to 10
    scenario_list = range(1, num_scenarios+1)  # Scenarios 1 to 15

    print(arc_list)
    # Parameters (same as before)
    phi = {s: 1 / len(scenario_list) for s in scenario_list}  # Uniform scenario probabilities
    print("phi", phi)
    delta = {
        (i, c, t, s): 5000 * ((i + t + s) % 10 + 1)
        for i in node_list for c in commodity_list for t in time_period_list for s in scenario_list
    }

    delta2 = {
        (i, c, s): 5000 * ((i + s) % 10 + 1)
        for i in node_list for c in commodity_list for s in scenario_list
    }

    # Create the model problem
    model = gp.Model("MasterProblem")
    # model.Params.OutputFlag = 0
    m_i = model.addVars(node_list, name="m_i", lb=0)
    q = model.addVars(node_list, commodity_list, name="q", lb=0)
    p = model.addVars(node_list, vtype=GRB.BINARY, name="p")
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")

    theta = model.addVar(name="theta", lb=0)
    L = {f"Commodity{k}": 2 for k in range(1, num_commodities)}
    P = 8

    ell = {(i, c): 1 for i in node_list for c in commodity_list}
    M_node = {(i, t): 20 for i in node_list for t in time_period_list}
    d = {
        (i, c, t): 30 + ((i + t) % 11) for i in node_list for c in commodity_list for t in time_period_list
    }
    # Model objective
    model.setObjective(theta,
                       GRB.MINIMIZE
                       )


    # Model constraints
    model.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 8, "MaxVehicles")
    model.addConstr(gp.quicksum(p[i] for i in node_list) <= P, "MaxAPSLocationsLimit")
    model.addConstr(gp.quicksum(p[i] for i in node_list) <= 8, "MaxAPSLocations")
    model.addConstrs((q[i, c] <= 25 * r[i, c] for i in node_list for c in commodity_list), "RestrictedAPS")
    model.addConstrs((gp.quicksum(q[i, c] for c in commodity_list) <= 2000 for i in node_list), "MaxCommodityPerNode")

    model.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), "RestrictedAssignment")
    model.addConstrs((r[i, c] <= m_i[i] for i in node_list for c in commodity_list),
                     "CommodityPlacementRequiresVehicle")

    model.addConstrs((gp.quicksum(r[i, c] for i in node_list) >= L[c] for c in commodity_list), "MinAPSPerCommodity")

    model.update()

    # subproblem constraints here

    b = {f"Commodity{k}": 1 for k in range(1, num_commodities)}
    mu = {i: 30000 for i in range(1, num_nodes)}

    # Add second-stage decision variables (flow, inventory, unmet demand, etc.)
    x = model.addVars(arc_list, commodity_list, time_period_list, vehicle_list, scenario_list, name="x", lb=0)
    # print(x)
    y = model.addVars(node_list, commodity_list, time_period_list, scenario_list, name="y", lb=0)
    z = model.addVars(node_list, commodity_list, time_period_list, scenario_list, name="z", lb=0)
    w = model.addVars(node_list, commodity_list, time_period_list2, scenario_list, name="w", lb=0)
    alpha = model.addVars(node_list, commodity_list, scenario_list, name="alpha", lb=0)
    bar_x = model.addVars(arc_list, time_period_list, vehicle_list, scenario_list, name="bar_x", lb=0, ub=1)
    bar_w = model.addVars(node_list, time_period_list2, vehicle_list, scenario_list, name="bar_w", lb=0, ub=1)

    # Subproblem objective
    # delta = 5000  # Example, replace with actual data
    model.setObjective(
        gp.quicksum(phi[s] * gp.quicksum(
            delta[i, c, t, s] * z[i, c, t, s] for i in node_list for c in commodity_list for t in time_period_list) + phi[s]*gp.quicksum(delta2[i, c, s] * alpha[i, c, s] for i in node_list for c in commodity_list)
                    for s in scenario_list),
        GRB.MINIMIZE
    )

    model.addConstrs(
        (
            gp.quicksum(y[i, c, tprime, s] for tprime in range(1, t + 1)) + z[i, c, t, s] == d.get((i, c, s), 0)
            for i in node_list
            for c in commodity_list
            for t in time_period_list
            for s in scenario_list
        ),
        "DemandConstraints",
    )

    # Flow balance constraint: flow into a node equals flow out plus carried inventory
    model.addConstrs(
        (
            gp.quicksum(x[i, j, c, t, v, s] for v in vehicle_list for j in node_list if (i, j) in arc_list)
            - gp.quicksum(x[j, i, c, t, v, s] for v in vehicle_list for j in node_list if (j, i) in arc_list)
            + w[i, c, t, s]
            - (w[i, c, t - 1, s]) + y[i, c, t, s]
            == 0
            for i in node_list
            for c in commodity_list
            for t in time_period_list
            for s in scenario_list
        ),
        "CommodityFlowBalance",
    )

    # Capacity constraint: cumulative commodity inventory at a node must not exceed max capacity
    model.addConstrs(
        (
            gp.quicksum(b.get(c, 0) * w[i, c, t, s] for c in commodity_list) <= M_node.get((i, t), 0)
            for i in node_list
            for t in time_period_list
            for s in scenario_list
        ),
        "MaxNodeCap",
    )
    #
    model.addConstrs(
        (w[i, c, 0, s] == q[i,c] for i in node_list for c in commodity_list for s in scenario_list),
        "StartingInventory",
    ),

    model.addConstrs(
        (
            alpha[i, c, s] >= ell.get((i, c), 0) - (
                    q[i,c]
                    - (gp.quicksum(
                x[i, j, c, t, v, s] for j in node_list if (i, j) in arc_list for t in time_period_list for v in
                vehicle_list
            )
                       - gp.quicksum(
                        x[j, i, c, t, v, s] for j in node_list if (j, i) in arc_list for t in time_period_list for v in
                        vehicle_list
                    ))
            )
            for i in node_list
            for c in commodity_list
            for s in scenario_list
        ),
        "SafetyStockConstraint",
    )

    model.addConstrs(
        (
            gp.quicksum(bar_x[i, j, t, v, s] for j in node_list if (i, j) in arc_list)
            - gp.quicksum(bar_x[j, i, t, v, s] for j in node_list if (j, i) in arc_list)
            + bar_w[i, t, v, s]
            - (bar_w[i, t - 1, v, s])
            == 0
            for i in node_list
            for t in time_period_list
            for v in vehicle_list
            for s in scenario_list
        ),
        "VehicleFlowBalance",
    )
    model.addConstrs(
        (
            gp.quicksum(bar_w[i, 0, v, s] for v in vehicle_list) == m_i[i]
            for i in node_list
            for s in scenario_list
        ),
        "VehicleStartingInventory",
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, c, t, v, s] for c in commodity_list) <= 100 * bar_x[i, j, t, v, s]
            for (i, j) in arc_list
            for t in time_period_list
            for v in vehicle_list
            for s in scenario_list
        ),
        "VehicleOnPositiveFlowArcs",
    )

    T = max(time_period_list)  # Assuming T is the last time period

    model.addConstrs(
        (
            gp.quicksum(bar_w[i, T, v, s] for v in vehicle_list) == m_i[i]
            for i in node_list
            for s in scenario_list
        ),
        "VehiclesReturn"
    )
    model.addConstrs(
        (
            gp.quicksum(bar_x[i, j, 1, v, s] for (i, j) in arc_list) <=
            gp.quicksum(bar_x[i, j, 1, v + 1, s] for (i, j) in arc_list)
            for v in range(1, len(vehicle_list))  # v = 1, ..., |V|-1
            for s in scenario_list
        ),
        "VehicleFlowMonotonicity"
    )

    print("scenario_list", scenario_list)
    for s in scenario_list:
        print ("s", s)
    model.Params.MipGap = 0.00000000001
    model.optimize()

    # Print only q variables
    for i in node_list:
        for c in commodity_list:
            if q[i, c].X != 0:  # Only print non-zero values
                print(f"q[{i},{c}] = {q[i, c].X}")


if __name__ == "__main__":
    solve_stochastic_vrp()
# if __name__ == "__main__":
#     # Instantiate the model
#     # stochastic_vrp_model = build_stochastic_vrp_model()
#     # deterministic_vrp_model = build_deterministic_vrp_model()
#     #
#     # # Solve with a solver
#     # solver = SolverFactory("gurobi")  # Replace with a solver like CPLEX or Gurobi, if needed
#     # solver.solve(stochastic_vrp_model)
#     # solver.solve(deterministic_vrp_model)
#
#     # Display only the variable values for both models
#     # print("Stochastic Model Variable Values:")
#     # stochastic_vrp_model.display(ostream=None)
#
#     # print("\nDeterministic Model Variable Values:")
#     # deterministic_vrp_model.display(ostream=None)
#     # solve_deterministic_vrp()
#     solve_stochastic_vrp()
#
#     # solve stochastic vrp
#
# ### Execution
# # - Implement the subproblem constraints (e.g., flow balance).
# # - Run the full iterative decomposition.
# # - Output both the first-stage (master) and second-stage (subproblem) decisions.
# #
# # Let me know if you'd like more details on specific constraints or cuts!