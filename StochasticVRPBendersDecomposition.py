
### Complete Code Expansion

# Below is the continuation of the skeleton and full implementation:
import random
import gurobipy as gp
from gurobipy import GRB


def solve_stochastic_vrp_benders():

    # Data initialization
    time_period_list = range(1, 6)  # Time periods 1 to 10
    vehicle_list = range(1, 6)  # Vehicles 1 to 10
    arc_list = random.sample([(i, j) for i in range(1, 6) for j in range(1, 6) if i != j], 20)
    node_list = list(range(1, 6))  # Nodes 1 to 50
    commodity_list = [f"Commodity{k}" for k in range(1, 6)]  # Commodities 1 to 10
    scenario_list = range(1, 6)  # Scenarios 1 to 15

    print(arc_list)
    # Parameters (same as before)
    phi = {s: 1 / len(scenario_list) for s in scenario_list}  # Uniform scenario probabilities
    delta = {
        (i, c, t, s): 5000 * ((i + t + s) % 10 + 1)
        for i in node_list for c in commodity_list for t in time_period_list for s in scenario_list
    }

    # Create the master problem
    master = gp.Model("MasterProblem")
    master.Params.OutputFlag = 0
    m_i = master.addVars(node_list, name="m_i", lb=0)
    q = master.addVars(node_list, commodity_list, name="q", lb=0)
    p = master.addVars(node_list, vtype=GRB.BINARY, name="p")
    r = master.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    
    
    theta = master.addVar(name="theta", lb=0)
    L = {f"Commodity{k}": 2 for k in range(1, 6)}
    P = 10

    ell = {(i, c): 1 for i in node_list for c in commodity_list}
    M_node = {(i, t): 2500 for i in node_list for t in time_period_list}
    d = {
        (i, c, t): (i + t) % 3 * 5 for i in node_list for c in commodity_list for t in time_period_list
    }
    # Master objective
    master.setObjective(theta,
        GRB.MINIMIZE
    )

    # Master constraints
    master.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 8, "MaxVehicles")
    master.addConstr(gp.quicksum(p[i] for i in node_list) <= P, "MaxAPSLocationsLimit")
    master.addConstr(gp.quicksum(p[i] for i in node_list) <= 8, "MaxAPSLocations")
    master.addConstrs((q[i, c] <= 1000 * r[i, c] for i in node_list for c in commodity_list), "RestrictedAPS")
    master.addConstrs((gp.quicksum(q[i, c] for c in commodity_list) <= 2000 for i in node_list), "MaxCommodityPerNode")

    master.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), "RestrictedAssignment")
    master.addConstrs((r[i, c] <= m_i[i] for i in node_list for c in commodity_list),"CommodityPlacementRequiresVehicle")

    master.addConstrs((gp.quicksum(r[i, c] for i in node_list) >= L[c] for c in commodity_list), "MinAPSPerCommodity")

    master.update()
    

    # Benders decomposition
    max_iters = 300
    tol = 1e-4
    iteration = 0

    converged = False

    while not converged and iteration < max_iters:
        print(f"Starting Benders iteration: {iteration + 1}")
        master.optimize()

        if master.status == GRB.OPTIMAL:
            print(f"Objective function value of the master problem: {master.objVal}")
            print(f"Objective function value: {master.getObjective().getValue()}")
        else:
            print("Master problem did not converge.")
            break

        # Extract master solution
        m_i_sol = {i: m_i[i].x for i in node_list}
        q_sol = {i: {c: q[i, c].x for c in commodity_list} for i in node_list}
        p_sol = {i: p[i].x for i in node_list}
        r_sol = {i: {c: r[i, c].x for c in commodity_list} for i in node_list}

        

        print("Positive variable values from the master problem:")
        positive_vars_master = {var.VarName: var.X for var in master.getVars() if var.X > 0}
        for var_name, var_value in positive_vars_master.items():
            print(f"{var_name}: {var_value:.2f}")

        # Solve subproblems for all scenarios
        subproblem_costs = []
        infeasible_scenarios = []
        # for s in scenario_list:
        subproblem, subproblem_cost = solve_subproblem(
                m_i_sol, q_sol, p_sol, r_sol, node_list, commodity_list, time_period_list, arc_list, vehicle_list,d, M_node,ell
        )

        if subproblem_cost == float('inf'):  # Subproblem infeasible
            print(f"Subproblem is infeasible. Extracting extreme rays.")

            # Compute extreme rays from the dual problem of the subproblem
            subproblem.computeIIS()
            extreme_rays = {}

            for constr in subproblem.getConstrs():
                if constr.IISConstr:
                    extreme_rays[constr.ConstrName] = constr.Pi
                    print(f"Constraint {constr.ConstrName} has an extreme ray value of {constr.Pi}.")

            # Add feasibility cut (to be customized as needed)

            dual_demand = {
                (i, c, t, s): subproblem.getConstrByName(f"DemandConstraints[{i},{c},{t},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for t in time_period_list
                for s in scenario_list
            }

            dual_capacity = {
                (i, t, s): subproblem.getConstrByName(f"MaxNodeCap[{i},{t},{s}]").Pi
                for i in node_list
                for t in time_period_list
                for s in scenario_list
            }

            print("output dual capacity", dual_capacity)
            dual_start_inventory = {
                (i, c, s): subproblem.getConstrByName(f"StartingInventory[{i},{c},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for s in scenario_list
            }
            print("output dual start inventory", dual_start_inventory)
            dual_vehicle_start_inventory = {
                (i, s): subproblem.getConstrByName(f"VehicleStartingInventory[{i},{s}]").Pi
                for i in node_list
                for s in scenario_list

            }

            print("output dual vehicle start inventory", dual_vehicle_start_inventory)
            dual_max_node_cap = {
                (i, t, s): subproblem.getConstrByName(f"MaxNodeCap[{i},{t},{s}]").Pi
                for i in node_list
                for t in time_period_list
                for s in scenario_list
            }
            print("output dual max node cap", dual_max_node_cap)

            # dual_safety

            dual_safety = {
                (i, c, s): subproblem.getConstrByName(f"SafetyStockConstraint[{i},{c},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for s in scenario_list
            }
            # print("output dual safety", dual_safety)
            dual_vehicle_return = {
                (i, s): subproblem.getConstrByName(f"VehiclesReturn[{i},{s}]").Pi
                for i in node_list
                for s in scenario_list
            }
            # print("output dual vehicle return", dual_vehicle_return)
            # Construct the optimality cut
            feasibility_cut = (gp.quicksum(
                dual_demand[i, c, t, s] * d.get((i, c, s), 0) for i in node_list for c in commodity_list for s in
                scenario_list for t in time_period_list)
                              + gp.quicksum(
                        M_node.get((i, t), 0) * dual_capacity[i, t, s] for i in node_list for t in time_period_list for
                        s in scenario_list)
                              + gp.quicksum((ell.get((i, c), 0) - q[i, c]) * dual_safety[i, c, s]
                                            for i in node_list for c in commodity_list for s in scenario_list)
                              + gp.quicksum(
                        dual_start_inventory[i, c, s] * q[i, c] for i in node_list for c in commodity_list for s in
                        scenario_list)
                              + gp.quicksum(
                        dual_vehicle_return[i, s] * m_i[i] for i in node_list for s in scenario_list)
                              + gp.quicksum(
                        dual_vehicle_start_inventory[i, s] * m_i[i] for i in node_list for s in scenario_list)
                              )


            # Ensure master variable theta bounds the subproblem cost
            master.addConstr(0 >= feasibility_cut, name=f"FeasibilityCut")
        else:

            dual_demand = {
                (i, c,t, s): subproblem.getConstrByName(f"DemandConstraints[{i},{c},{t},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for t in time_period_list
                for s in scenario_list
            }

            print("output dual demand",dual_demand)
            dual_capacity = {
                (i, t, s): subproblem.getConstrByName(f"MaxNodeCap[{i},{t},{s}]").Pi
                for i in node_list
                for t in time_period_list
                for s in scenario_list
            }

            print("output dual capacity",dual_capacity)
            dual_start_inventory = {
                (i, c, s): subproblem.getConstrByName(f"StartingInventory[{i},{c},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for s in scenario_list
            }
            print("output dual start inventory",dual_start_inventory)
            dual_vehicle_start_inventory = {
                (i, s): subproblem.getConstrByName(f"VehicleStartingInventory[{i},{s}]").Pi
                for i in node_list
                for s in scenario_list

            }

            print("output dual vehicle start inventory",dual_vehicle_start_inventory)
            dual_max_node_cap = {
                (i, t, s): subproblem.getConstrByName(f"MaxNodeCap[{i},{t},{s}]").Pi
                for i in node_list
                for t in time_period_list
                for s in scenario_list
            }
            print("output dual max node cap",dual_max_node_cap)

            # dual_safety


            dual_safety = {
                (i, c, s): subproblem.getConstrByName(f"SafetyStockConstraint[{i},{c},{s}]").Pi
                for i in node_list
                for c in commodity_list
                for s in scenario_list
            }
            print("output dual safety",dual_safety)
            dual_vehicle_return = {
                (i,s): subproblem.getConstrByName(f"VehiclesReturn[{i},{s}]").Pi
                for i in node_list
                for s in scenario_list
            }
            print("output dual vehicle return",dual_vehicle_return)
            # Construct the optimality cut
            optimality_cut = (gp.quicksum(
                  dual_demand[i, c,t, s] * d.get((i, c, s), 0) for i in node_list for c in commodity_list for s in scenario_list for t in time_period_list)
                    + gp.quicksum(M_node.get((i,t),0)*dual_capacity[i,t,s] for i in node_list for t in time_period_list for s in scenario_list)
                      + gp.quicksum((ell.get((i,c),0)-q[i, c])*dual_safety[i, c, s]
                                    for i in node_list for c in commodity_list for s in scenario_list)
                              + gp.quicksum(dual_start_inventory[i, c, s] * q[i, c] for i in node_list for c in commodity_list for s in scenario_list)
                              + gp.quicksum(dual_vehicle_return[i,s]*m_i[i] for i in node_list for s in scenario_list)
                              + gp.quicksum(dual_vehicle_start_inventory[i,s]*m_i[i] for i in node_list for s in scenario_list)
            )

            # Ensure master variable theta bounds the subproblem cost
            master.addConstr(theta >= optimality_cut, name=f"OptimalityCut")
            theta_value = theta.x

        # Check convergence
        approx_cost = sum(phi[s] * subproblem_cost for s in scenario_list)

        print(f"Iteration {iteration + 1}: Theta = {theta_value}, Approx = {approx_cost}")
        print(f"Master objective function value: {master.objVal}")
        if abs(theta_value - approx_cost) < tol:
            converged = True

        iteration += 1

    # Final solution extraction
    if master.status == GRB.OPTIMAL:
        print(f"Converged solution with objective: {master.objVal}")
        print("First-stage decisions:")
        for i in node_list:
            print(f"m_i[{i}] = {m_i_sol[i]:.2f}, p[{i}] = {p_sol[i]:.2f}")
            for c in commodity_list:
                print(f"q[{i}, {c}] = {q_sol[i][c]:.2f}, r[{i}, {c}] = {r_sol[i][c]:.2f}")

        print("Positive variable values:")
        positive_vars = {var.VarName: var.X for var in master.getVars() if var.X > 0}
        for var_name, var_value in positive_vars.items():
            print(f"{var_name}: {var_value:.2f}")



# ```python
def solve_subproblem(m_i, q, p, r, node_list, commodity_list, time_period_list, arc_list,vehicle_list,d,M_node,ell):
    subproblem = gp.Model(f"Subproblem_Scenario")
    subproblem.Params.OutputFlag = 0
    scenario_list = range(1, 6)  # Scenarios 1 to 10
    time_period_list2 = range(0, 6)  # Time periods 0 to 10



    b = {f"Commodity{k}": 1 for k in range(1, 6)}
    mu = {i: 30000 for i in range(1, 6)}

    # Add second-stage decision variables (flow, inventory, unmet demand, etc.)
    x = subproblem.addVars(arc_list, commodity_list, time_period_list,vehicle_list, scenario_list, name="x", lb=0)
    # print(x)
    y = subproblem.addVars(node_list, commodity_list, time_period_list, scenario_list, name="y", lb=0)
    z = subproblem.addVars(node_list, commodity_list, time_period_list, scenario_list, name="z", lb=0)
    w = subproblem.addVars(node_list, commodity_list, time_period_list2, scenario_list, name="w", lb=0)
    alpha = subproblem.addVars(node_list, commodity_list, scenario_list, name="alpha", lb=0)
    bar_x = subproblem.addVars(arc_list, time_period_list, vehicle_list, scenario_list, name="bar_x", lb=0, ub=1)
    bar_w = subproblem.addVars(node_list, time_period_list2, vehicle_list, scenario_list, name="bar_w", lb=0, ub=1)

    # Subproblem objective
    delta = 5000  # Example, replace with actual data
    subproblem.setObjective(
        gp.quicksum(delta * z[i, c, t,s] for i in node_list for c in commodity_list for t in time_period_list for s in scenario_list),
        GRB.MINIMIZE
    )

    subproblem.addConstrs(
        (
            gp.quicksum(y[i, c, t, s] for t in range(1, t + 1)) + z[i, c, t, s] == d.get((i, c, s), 0)
            for i in node_list
            for c in commodity_list
            for t in time_period_list
            for s in scenario_list
        ),
        "DemandConstraints",
    )

    # Flow balance constraint: flow into a node equals flow out plus carried inventory
    subproblem.addConstrs(
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
    subproblem.addConstrs(
        (
            gp.quicksum(b.get(c, 0) * w[i, c, t, s] for c in commodity_list) <= M_node.get((i, t), 0)
            for i in node_list
            for t in time_period_list
            for s in scenario_list
        ),
        "MaxNodeCap",
    )
    #
    subproblem.addConstrs(
        (w[i, c, 0, s] == q[i][c] for i in node_list for c in commodity_list for s in scenario_list),
        "StartingInventory",
    )

    subproblem.addConstrs(
        (
            alpha[i, c, s] >= ell.get((i, c), 0) - (
                    q[i][c]
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

    subproblem.addConstrs(
        (
            gp.quicksum(bar_x[i, j, t, v, s] for j in node_list if (i, j) in arc_list)
            - gp.quicksum(bar_x[j, i, t, v, s] for j in node_list if (j, i) in arc_list)
            + bar_w[i, t, v, s]
            - (bar_w[i, t - 1, v, s] )
            == 0
            for i in node_list
            for t in time_period_list
            for v in vehicle_list
            for s in scenario_list
        ),
        "VehicleFlowBalance",
    )
    subproblem.addConstrs(
        (
            gp.quicksum(bar_w[i, 0, v, s] for v in vehicle_list) == m_i[i]
            for i in node_list
            for s in scenario_list
        ),
        "VehicleStartingInventory",
    )

    subproblem.addConstrs(
        (
            gp.quicksum(x[i, j, c, t, v, s] for c in commodity_list) <= 100*bar_x[i, j, t, v, s]
            for (i, j) in arc_list
            for t in time_period_list
            for v in vehicle_list
            for s in scenario_list
        ),
        "VehicleOnPositiveFlowArcs",
    )

    T = max(time_period_list)  # Assuming T is the last time period

    subproblem.addConstrs(
        (
            gp.quicksum(bar_w[i, T, v, s] for v in vehicle_list) == m_i[i]
            for i in node_list
            for s in scenario_list
        ),
        "VehiclesReturn"
    )
    subproblem.addConstrs(
        (
            gp.quicksum(bar_x[i, j, 1, v, s] for (i, j) in arc_list) <=
            gp.quicksum(bar_x[i, j, 1, v + 1, s] for (i, j) in arc_list)
            for v in range(1, len(vehicle_list))  # v = 1, ..., |V|-1
            for s in scenario_list
        ),
        "VehicleFlowMonotonicity"
    )

    subproblem.optimize()

    # print("output subproblem objective value", subproblem.ObjVal)
    if subproblem.status == GRB.INFEASIBLE:
        return subproblem, float('inf')  # Indicates infeasibility
    else:
        positive_z_vars = {z[i, c, t, s].VarName: z[i, c, t, s].X for i in node_list for c in commodity_list for t in
                           time_period_list for s in scenario_list if z[i, c, t, s].X > 0}
        print("Positive z variable values:")
        for var_name, var_value in positive_z_vars.items():
            print(f"{var_name}: {var_value:.2f}")

        positive_y_vars = {y[i, c, t, s].VarName: y[i, c, t, s].X for i in node_list for c in commodity_list for t in
                           time_period_list for s in scenario_list if y[i, c, t, s].X > 0}
        print("Positive y variable values:")
        for var_name, var_value in positive_y_vars.items():
            print(f"{var_name}: {var_value:.2f}")

        positive_x_vars = {x[i, j, c, t, v, s].VarName: x[i, j, c, t, v, s].X for i, j in arc_list for c in
                           commodity_list for t in
                           time_period_list for v in vehicle_list for s in scenario_list if x[i, j, c, t, v, s].X > 0}
        print("Positive x variable values:")
        for var_name, var_value in positive_x_vars.items():
            print(f"{var_name}: {var_value:.2f}")

        return subproblem, subproblem.ObjVal


# ```

#### Feasibility Cuts
# Add constraints to forbid infeasible solutions:
# ```python
# def add_feasibility_cut(master, m_i, q, p, r, s):
#     cone = ...  # Feasible region approximation
#     master.addConstr(...)  # Add feasibility constraint
# ```

#### Optimality Cuts
# Update `theta` with effective bounds:
# ```python
# def add_optimality_cut(master, theta, m_i, q, p, r, subproblem_cost, s):
#     master.addConstr(...)  # Benders optimality cuts
# ```
#
# ---



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
    # solve_deterministic_vrp()
    solve_stochastic_vrp_benders()


    #solve stochastic vrp

### Execution
# - Implement the subproblem constraints (e.g., flow balance).
# - Run the full iterative decomposition.
# - Output both the first-stage (master) and second-stage (subproblem) decisions.
#
# Let me know if you'd like more details on specific constraints or cuts!