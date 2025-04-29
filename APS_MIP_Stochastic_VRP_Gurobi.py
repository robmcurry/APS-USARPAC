import random


import gurobipy as gp
from gurobipy import GRB


def solve_deterministic_vrp():
    # Data initialization
    time_period_list = range(1, 11)  # Time periods 1 to 10
    time_period_list2 = range(0, 11)  # Time periods 0 to 10
    vehicle_list = range(1, 101)  # Vehicles 1 to 100
    arc_list = random.sample([(i, j) for i in range(1, 51) for j in range(1, 51) if i != j],
                             120)  # Random 120 directed arcs
    node_list = list(range(1, 51))  # Nodes 1 to 50
    commodity_list = [f"Commodity{i}" for i in range(1, 11)]  # Commodities 1 to 10

    # Parameters
    delta = {
        (i, c, t): random.randint(1000, 5000)
        for i in node_list
        for c in commodity_list
        for t in time_period_list
    }
    nu = {(i, c): round(random.uniform(0.5, 2.0), 2) for i in node_list for c in commodity_list}
    M = 1000
    m = {i: random.randint(10, 100) for i in node_list}
    L = {c: random.randint(1, 3) for c in commodity_list}
    P = 5
    d = {
        (i, c): random.randint(0, 30)
        for i in node_list
        for c in commodity_list
    }
    b = {c: 0 for c in commodity_list}
    M_node = {(i, t): random.randint(50, 200) for i in node_list for t in time_period_list}
    mu = {i: random.randint(100, 300) for i in vehicle_list}
    ell = {(i, c): random.randint(1, 5) for i in node_list for c in commodity_list}

    # Create a Gurobi model
    model = gp.Model("Deterministic_VRP")

    # Decision variables
    x = model.addVars(arc_list, commodity_list, time_period_list, vehicle_list, name="x", lb=0)
    y = model.addVars(node_list, commodity_list, time_period_list, name="y", lb=0)
    z = model.addVars(node_list, commodity_list, time_period_list, name="z", lb=0)
    w = model.addVars(node_list, commodity_list, time_period_list2, name="w", lb=0)
    m_i = model.addVars(node_list, name="m_i", lb=0)
    q = model.addVars(node_list, commodity_list, name="q", lb=0)
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    p = model.addVars(node_list, vtype=GRB.BINARY, name="p")
    alpha = model.addVars(node_list, commodity_list, name="alpha", lb=0)
    bar_x = model.addVars(arc_list, time_period_list, vehicle_list, vtype=GRB.BINARY, name="bar_x")
    bar_w = model.addVars(node_list, time_period_list2, vehicle_list, vtype=GRB.BINARY, name="bar_w")

    # Objective function
    model.setObjective(
        gp.quicksum(
            delta.get((i, c, t), 0) * z[i, c, t]
            for i in node_list
            for c in commodity_list
            for t in time_period_list
        ) + gp.quicksum(
            nu.get((i, c), 0) * alpha[i, c]
            for i in node_list
            for c in commodity_list
        ),
        GRB.MINIMIZE,
    )

    # Constraints
    model.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 10, "MaxVehicles")

    model.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), "APSBinary1")

    model.addConstrs((r[i, c] <= M * p[i] for i in node_list for c in commodity_list), "APSBinary3")

    model.addConstr(gp.quicksum(p[i] for i in node_list) <= P, "MaxAPSLocations")

    model.addConstrs((gp.quicksum(r[i, c] for i in node_list) >= L[c] for c in commodity_list), "MinAPSPerCommodity")

    model.addConstrs(
        (
            gp.quicksum(y[i, c, t] for t in range(1, t + 1)) + z[i, c, t] == d.get((i, c), 0)
            for i in node_list
            for c in commodity_list
            for t in time_period_list
        ),
        "DemandConstraints",
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, c, t, v] for j in node_list if (i, j) in arc_list for v in vehicle_list)
            - gp.quicksum(x[j, i, c, t, v] for j in node_list if (j, i) in arc_list for v in vehicle_list)
            + w[i, c, t]
            - (w[i, c, t - 1] ) + y[i, c, t]
            == 0
            for i in node_list
            for c in commodity_list
            for t in time_period_list
        ),
        "CommodityFlowBalance",
    )

    model.addConstrs(
        (
            gp.quicksum(b[c] * w[i, c, t] for c in commodity_list) <= M_node.get((i, t), 0)
            for i in node_list
            for t in time_period_list
        ),
        "MaxNodeCap",
    )

    model.addConstrs(
        (w[i, c, 0] == q[i, c] for i in node_list for c in commodity_list),
        "StartingInventory",
    )

    model.addConstrs(
        (
            alpha[i, c] >= ell.get((i, c), 0) - (
                    q[i, c]
                    - gp.quicksum(
                x[i, j, c, t, v] for j in node_list if (i, j) in arc_list for t in time_period_list for v in
                vehicle_list
            )
                    - gp.quicksum(
                x[j, i, c, t, v] for j in node_list if (j, i) in arc_list for t in time_period_list for v in
                vehicle_list
            )
            )
            for i in node_list
            for c in commodity_list
        ),
        "SafetyStockConstraint",
    )

    model.addConstrs(
        (
            gp.quicksum(bar_x[i, j, t, v] for j in node_list if (i, j) in arc_list)
            - gp.quicksum(bar_x[j, i, t, v] for j in node_list if (j, i) in arc_list)
            + bar_w[i, t, v]
            - (bar_w[i, t - 1, v] if t > 0 else 0)
            == 0
            for i in node_list
            for t in time_period_list
            for v in vehicle_list
        ),
        "VehicleFlowBalance",
    )

    model.addConstrs(
        (
            bar_w[i, time_period_list[-1], v] == m_i[i]
            for i in node_list
            for v in vehicle_list
        ),
        "VehiclesReturn",
    )

    # Optimize
    model.optimize()

    # Results
    if model.status == GRB.OPTIMAL:
        print("Optimal Objective Value:", model.objVal)
        # for var in model.getVars():
        #     if var.x > 0:
        #         print(f"{var.varName}: {var.x}")

def solve_stochastic_vrp():
    # Data initialization
    time_period_list = range(1, 11)  # Time periods 1 to 10
    time_period_list2 = range(0, 11)  # Time periods 0 to 10
    vehicle_list = range(1, 11)  # Vehicles 1 to 100
    arc_list = random.sample([(i, j) for i in range(1, 51) for j in range(1, 51) if i != j],
                             120)  # Random set of 120 arcs
    node_list = list(range(1, 51))  # Nodes 1 to 50
    commodity_list = [f"Commodity{k}" for k in range(1, 11)]  # Commodities 1 to 10
    scenario_list = range(1, 6)  # Scenarios 1 to 10

    # Parameters
    phi = {s: 1 / len(scenario_list) for s in scenario_list}  # Uniform scenario probabilities
    delta = {
        (i, c, t, s): 5000 * ((i + t + s) % 10 + 1) for i in node_list for c in commodity_list
        for t in time_period_list for s in scenario_list
    }
    nu = {(i, c): 1.0 + (j / 10) for j, (i, c) in enumerate([(i, c) for i in node_list for c in commodity_list])}
    M = 1000
    m = {i: (i * 10) % 100 + 10 for i in node_list}
    L = {f"Commodity{k}": 1 for k in range(1, 11)}
    P = 10
    d = {
        (i, c, t): (i + t) % 10 * 5 for i in node_list for c in commodity_list for t in time_period_list
    }
    b = {f"Commodity{k}": 1 for k in range(1, 11)}
    M_node = {(i, t): 100 * ((i + t) % 5 + 1) for i in node_list for t in time_period_list}
    mu = {i: 300 for i in range(1, 11)}
    ell = {(i, c): 5 for i in node_list for c in commodity_list}

    # Create a Gurobi model
    model = gp.Model("Stochastic_VRP")

    # Decision variables
    x = model.addVars(arc_list, commodity_list, time_period_list, vehicle_list, scenario_list, name="x", lb=0)
    y = model.addVars(node_list, commodity_list, time_period_list, scenario_list, name="y", lb=0)
    z = model.addVars(node_list, commodity_list, time_period_list, scenario_list, name="z", lb=0)
    w = model.addVars(node_list, commodity_list, time_period_list2, scenario_list, name="w", lb=0)
    m_i = model.addVars(node_list, name="m_i", lb=0)
    q = model.addVars(node_list, commodity_list, name="q", lb=0)
    r = model.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    p = model.addVars(node_list, vtype=GRB.BINARY, name="p")
    alpha = model.addVars(node_list, commodity_list, scenario_list, name="alpha", lb=0)
    bar_x = model.addVars(arc_list, time_period_list, vehicle_list, scenario_list, vtype=GRB.BINARY, name="bar_x")
    bar_w = model.addVars(node_list, time_period_list2, vehicle_list, scenario_list, vtype=GRB.BINARY, name="bar_w")

    # Objective function
    model.setObjective(
        gp.quicksum(
            phi[s] * (
                    gp.quicksum(
                        delta.get((i, c, t, s), 0) * z[i, c, t, s]
                        for i in node_list
                        for c in commodity_list
                        for t in time_period_list
                    ) + gp.quicksum(
                nu.get((i, c), 0) * alpha[i, c, s]
                for i in node_list
                for c in commodity_list
            )
            )
            for s in scenario_list
        ),
        GRB.MINIMIZE,
    )

    # Constraints
    model.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 10, "MaxVehicles")

    model.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), "APSBinary1")

    model.addConstrs((r[i, c] <= M * p[i] for i in node_list for c in commodity_list), "APSBinary3")

    model.addConstr(gp.quicksum(p[i] for i in node_list) <= P, "MaxAPSLocations")

    model.addConstrs((gp.quicksum(r[i, c] for i in node_list) >= L[c] for c in commodity_list), "MinAPSPerCommodity")

    model.addConstrs(
        (
            gp.quicksum(y[i, c, t, s] for t in range(1, t + 1)) + z[i, c, t, s] == d.get((i, c, s), 0)
            for i in node_list
            for c in commodity_list
            for t in time_period_list
            for s in scenario_list
        ),
        "DemandConstraints",
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, c, t, v, s] for j in node_list if (i, j) in arc_list for v in vehicle_list)
            - gp.quicksum(x[j, i, c, t, v, s] for j in node_list if (j, i) in arc_list for v in vehicle_list)
            + w[i, c, t, s]
            - (w[i, c, t - 1, s] if t > 1 else 0) + y[i, c, t, s]
            == 0
            for i in node_list
            for c in commodity_list
            for t in time_period_list
            for s in scenario_list
        ),
        "CommodityFlowBalance",
    )

    model.addConstrs(
        (
            gp.quicksum(b.get(c, 0) * w[i, c, t, s] for c in commodity_list) <= M_node.get((i, t), 0)
            for i in node_list
            for t in time_period_list
            for s in scenario_list
        ),
        "MaxNodeCap",
    )

    model.addConstrs(
        (w[i, c, 0, s] == q[i, c] for i in node_list for c in commodity_list for s in scenario_list),
        "StartingInventory",
    )

    model.addConstrs(
        (
            alpha[i, c, s] >= ell.get((i, c), 0) - (
                    q[i, c]
                    - gp.quicksum(
                x[i, j, c, t, v, s] for j in node_list if (i, j) in arc_list for t in time_period_list for v in
                vehicle_list
            )
                    - gp.quicksum(
                x[j, i, c, t, v, s] for j in node_list if (j, i) in arc_list for t in time_period_list for v in
                vehicle_list
            )
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
            - (bar_w[i, t - 1, v, s] if t > 0 else 0)
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
            gp.quicksum(bar_w[i, time_period_list[-1], v, s] for v in vehicle_list) == m_i[i]
            for i in node_list
            for s in scenario_list
        ),
        "VehiclesReturn",
    )

    # Optimize
    model.optimize()

    # Results
    if model.status == GRB.OPTIMAL:
        print("Optimal Objective Value:", model.objVal)
        # for var in model.getVars():
        #     if var.x > 0:
        #         print(f"{var.varName}: {var.x}")
            # print(f"{var.varName}: {var.x}")





if __name__ == "__main__":
    # Instantiate the model
    # stochastic_vrp_model = build_stochastic_vrp_model()
    # deterministic_vrp_model = build_deterministic_vrp_model()
    #
    # # Solve with a solver
    # solver = SolverFactory("gurobi")  # Replace with a solver like CPLEX or Gurobi, if needed
    # solver.solve(stochastic_vrp_model)
    # solver.solve(deterministic_vrp_model)



    # Display only the variable values for both models
    # print("Stochastic Model Variable Values:")
    # stochastic_vrp_model.display(ostream=None)

    # print("\nDeterministic Model Variable Values:")
    # deterministic_vrp_model.display(ostream=None)
    solve_deterministic_vrp()
    #solve stochastic vrp


    solve_stochastic_vrp()

    # stochastic_vrp_model.display()
    #
    # deterministic_vrp_model.display()



