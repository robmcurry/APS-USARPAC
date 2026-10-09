def solve_stochastic_vrp_benders():
    import random  # For arc sampling
    import gurobipy as gp
    from gurobipy import GRB

    # Data initialization (same as before)
    time_period_list = range(1, 11)  # Time periods 1 to 10
    vehicle_list = range(1, 11)  # Vehicles 1 to 10
    arc_list = random.sample([(i, j) for i in range(1, 51) for j in range(1, 51) if i != j], 120)
    node_list = list(range(1, 51))  # Nodes 1 to 50
    commodity_list = [f"Commodity{k}" for k in range(1, 11)]  # Commodities 1 to 10
    scenario_list = range(1, 16)  # Scenarios 1 to 15

    # Parameters
    phi = {s: 1 / len(scenario_list) for s in scenario_list}  # Uniform scenario probabilities
    delta = {
        (i, c, t, s): 5000 * ((i + t + s) % 10 + 1)
        for i in node_list for c in commodity_list for t in time_period_list for s in scenario_list
    }
    nu = {(i, c): 1.0 + (j / 10) for j, (i, c) in enumerate([(i, c) for i in node_list for c in commodity_list])}
    M = 1000
    m = {i: (i * 10) % 100 + 10 for i in node_list}
    P = 10
    mu = {i: 300 for i in range(1, 11)}
    
    # Create master problem
    master = gp.Model("MasterProblem")

    # Decision variables (first-stage decisions)
    m_i = master.addVars(node_list, name="m_i", lb=0)  # Vehicles assigned to nodes
    q = master.addVars(node_list, commodity_list, name="q", lb=0)  # Inventory quantities
    p = master.addVars(node_list, vtype=GRB.BINARY, name="p")  # APS opening decisions
    r = master.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")  # APS commodity assignment
    theta = master.addVar(name="theta", lb=0)  # Approximation of subproblem cost

    # Master objective function
    master.setObjective(
        gp.quicksum(mu[i] * m_i[i] for i in node_list) + gp.quicksum(nu[i, c] * q[i, c] for i in node_list for c in commodity_list) + theta,
        GRB.MINIMIZE,
    )

    # Master problem constraints
    master.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 10, "MaxVehicles")
    master.addConstr(gp.quicksum(p[i] for i in node_list) <= P, "MaxAPSLocations")

    master.addConstrs((q[i, c] <= M * r[i, c] for i in node_list for c in commodity_list), "FirstStageBinary1")
    master.addConstrs((r[i, c] <= M * p[i] for i in node_list for c in commodity_list), "FirstStageBinary2")

    master.update()

    # Start Benders decomposition iterations
    tol = 1e-4
    max_iters = 50
    iteration = 0

    while iteration < max_iters:
        print(f"Benders iteration {iteration + 1}")

        # Solve the master problem
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print("Master problem did not converge.")
            break

        # Extract first-stage solution
        m_i_sol = {i: m_i[i].x for i in node_list}
        q_sol = {i: {c: q[i, c].x for c in commodity_list} for i in node_list}
        p_sol = {i: p[i].x for i in node_list}
        r_sol = {i: {c: r[i, c].x for c in commodity_list} for i in node_list}

        # For all scenarios, solve the subproblem
        worst_case_cost = 0
        for s in scenario_list:
            # Create subproblem for scenario `s` (pass master variables as parameters)
            subproblem, subproblem_cost = solve_subproblem(m_i_sol, q_sol, p_sol, r_sol, s, node_list, commodity_list, time_period_list, arcade_list)
            worst_case_cost += phi[s] * sub_problem_cost

            if sub_problem infeasible---cut
Here’s how you can finish framing the **Benders decomposition** logic for your stochastic VRP. Let’s break it into the **master problem**, **subproblem**, and introduce **Benders cuts** based on the subproblem results. I’ll refine the above incomplete outline into a coherent structure.

---

### 1. **Master Problem**
The **master problem** makes the first-stage decisions (`m_i`, `q`, `p`, `r`) and has an approximated cost for the subproblems (`theta`). This remains largely the same as described earlier, but now with additional constraints added dynamically from the subproblems in each iteration.

---

### 2. **Subproblem**
The **subproblem** is scenario-specific. For a given master decision (`m_i`, `q`, `p`, `r`), you solve the second-stage optimization (allocation flows, unmet demand, etc.).

**Subproblem Cost Calculation**:
- Each solved subproblem contributes to refining `theta` in the master problem.
- If the subproblem is infeasible, it indicates that the master problem decisions violate feasibility and the system cannot handle the specific demand/load for the scenario.

---

### 3. **Benders Cuts**
Two types of **cuts** are added dynamically in the master problem:
1. **Feasibility Cut**: If the subproblem is infeasible for a given scenario, a cut is added to the master problem to enforce valid decisions.
2. **Optimality Cut**: If the subproblem is feasible, constraints (using dual variables or value approximations) are added that bound the subproblem cost in the master.

---

### Complete Code Expansion

Below is the continuation of the skeleton and full implementation:

```python
def solve_stochastic_vrp_benders():
    import random
    import gurobipy as gp
    from gurobipy import GRB

    # Data initialization
    time_period_list = range(1, 11)  # Time periods 1 to 10
    vehicle_list = range(1, 11)  # Vehicles 1 to 10
    arc_list = random.sample([(i, j) for i in range(1, 51) for j in range(1, 51) if i != j], 120)
    node_list = list(range(1, 51))  # Nodes 1 to 50
    commodity_list = [f"Commodity{k}" for k in range(1, 11)]  # Commodities 1 to 10
    scenario_list = range(1, 16)  # Scenarios 1 to 15

    # Parameters (same as before)
    phi = {s: 1 / len(scenario_list) for s in scenario_list}  # Uniform scenario probabilities
    delta = {
        (i, c, t, s): 5000 * ((i + t + s) % 10 + 1)
        for i in node_list for c in commodity_list for t in time_period_list for s in scenario_list
    }

    # Create the master problem
    master = gp.Model("MasterProblem")
    m_i = master.addVars(node_list, name="m_i", lb=0)
    q = master.addVars(node_list, commodity_list, name="q", lb=0)
    p = master.addVars(node_list, vtype=GRB.BINARY, name="p")
    r = master.addVars(node_list, commodity_list, vtype=GRB.BINARY, name="r")
    theta = master.addVar(name="theta", lb=0)

    # Master objective
    master.setObjective(
        gp.quicksum(m_i[i] for i in node_list) + gp.quicksum(q[i, c] for i in node_list for c in commodity_list) + theta,
        GRB.MINIMIZE
    )

    # Master constraints
    master.addConstr(gp.quicksum(m_i[i] for i in node_list) <= 10, "MaxVehicles")
    master.addConstr(gp.quicksum(p[i] for i in node_list) <= 10, "MaxAPSLocations")
    master.addConstrs((q[i, c] <= 1000 * r[i, c] for i in node_list for c in commodity_list), "RestrictedAPS")
    master.addConstrs((r[i, c] <= p[i] for i in node_list for c in commodity_list), "RestrictedAssignment")

    master.update()

    # Benders decomposition
    max_iters = 50
    tol = 1e-4
    iteration = 0
    converged = False

    while not converged and iteration < max_iters:
        print(f"Starting Benders iteration: {iteration + 1}")
        master.optimize()

        if master.status != GRB.OPTIMAL:
            print("Master problem did not converge.")
            break

        # Extract master solution
        m_i_sol = {i: m_i[i].x for i in node_list}
        q_sol = {i: {c: q[i, c].x for c in commodity_list} for i in node_list}
        p_sol = {i: p[i].x for i in node_list}
        r_sol = {i: {c: r[i, c].x for c in commodity_list} for i in node_list}

        # Solve subproblems for all scenarios
        subproblem_costs = []
        infeasible_scenarios = []
        for s in scenario_list:
            subproblem, subproblem_cost = solve_subproblem(
                m_i_sol, q_sol, p_sol, r_sol, s, node_list, commodity_list, time_period_list, arc_list
            )
            if subproblem_cost == float('inf'):  # Subproblem infeasible
                infeasible_scenarios.append(s)
                add_feasibility_cut(master, m_i_sol, q_sol, p_sol, r_sol, s)
            else:
                subproblem_costs.append(subproblem_cost)
                add_optimality_cut(master, theta, m_i_sol, q_sol, p_sol, r_sol, subproblem_cost, s)

        # Check convergence
        theta_value = theta.x
        approx_cost = sum(phi[s] * subproblem_costs[s - 1] for s in scenario_list if s not in infeasible_scenarios)

        print(f"Iteration {iteration + 1}: Theta = {theta_value}, Approx = {approx_cost}")
        if abs(theta_value - approx_cost) < tol:
            converged = True

        iteration += 1

    # Final solution extraction
    if master.status == GRB.OPTIMAL:
        print(f"Converged solution with objective: {master.objVal}")
        print("First-stage decisions:")
        for i in node_list:
            print(f"m_i[{i}] = {m_i_sol[i]:.2f}, p[{i}] = {p_sol[i]:.2f}")
```

---

### Helper Functions
Add helper functions for the subproblem solver and to generate feasibility/optimality cuts:

#### Subproblem Solver

```python
def solve_subproblem(m_i, q, p, r, s, node_list, commodity_list, time_period_list, arc_list):
    subproblem = gp.Model(f"Subproblem_Scenario{s}")

    # Add second-stage decision variables (flow, inventory, unmet demand, etc.)
    x = subproblem.addVars(arc_list, commodity_list, time_period_list, name="x", lb=0)
    y = subproblem.addVars(node_list, commodity_list, time_period_list, name="y", lb=0)
    z = subproblem.addVars(node_list, commodity_list, time_period_list, name="z", lb=0)

    # Subproblem objective
    delta = 5000  # Example, replace with actual data
    subproblem.setObjective(
        gp.quicksum(delta * z[i, c, t] for i in node_list for c in commodity_list for t in time_period_list),
        GRB.MINIMIZE
    )

    # Add subproblem constraints (e.g., flow balance, demand, capacity)
    # ...

    subproblem.optimize()
    if subproblem.status == GRB.INFEASIBLE:
        return subproblem, float('inf')  # Indicates infeasibility
    else:
        return subproblem, subproblem.objVal
```

#### Feasibility Cuts
Add constraints to forbid infeasible solutions:
```python
def add_feasibility_cut(master, m_i, q, p, r, s):
    cone = ...  # Feasible region approximation
    master.addConstr(...)  # Add feasibility constraint
```

#### Optimality Cuts
Update `theta` with effective bounds:
```python
def add_optimality_cut(master, theta, m_i, q, p, r, subproblem_cost, s):
    master.addConstr(...)  # Benders optimality cuts
```

---

### Execution
- Implement the subproblem constraints (e.g., flow balance).
- Run the full iterative decomposition.
- Output both the first-stage (master) and second-stage (subproblem) decisions.

Let me know if you'd like more details on specific constraints or cuts!