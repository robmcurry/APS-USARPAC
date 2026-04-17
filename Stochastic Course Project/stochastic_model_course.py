"""
stochastic_model_course.py

Extended form two-stage stochastic prepositioning model with CVaR,
for the stochastic course project.

Model structure:
    First stage:
        p[i]         = 1 if node i is selected as a prepositioning location

    Second stage (for each scenario w):
        x[w,i,j,r]   = flow of commodity r on arc (i,j)
        z[w,j,r]     = unmet demand of commodity r at node j
        xi[w]        = CVaR excess loss variable

    Risk variables:
        eta          = VaR-like threshold for CVaR

Unit convention:
    1 unit of any commodity = 1 person-day of support
"""

from typing import Dict, Any, Tuple, List

import gurobipy as gp
from gurobipy import GRB



def solve_stochastic_cvar(
    instance: Dict[str, Any],
    time_limit: float = None,
    mip_gap: float = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Solve the extensive-form CVaR stochastic prepositioning model.

    Args:
        instance: stochastic instance dictionary produced by stochastic_input_builder.py
        time_limit: optional Gurobi time limit in seconds
        mip_gap: optional relative MIP gap target
        verbose: if True, allow standard solver output

    Returns:
        results dictionary with selected sites, objective value, and solution details
    """

    # --- Unpack sets ---
    N: List[int] = instance["nodes"]
    A: List[Tuple[int, int]] = instance["arcs"]
    R: List[str] = instance["commodities"]
    Omega: List[int] = instance["scenarios"]

    # --- Unpack parameters ---
    prob = instance["probability"]
    demand = instance["demand"]
    inventory_if_open = instance["inventory_if_open"]
    inventory_availability = instance.get("inventory_availability", None)
    safety_stock_fraction = float(instance.get("safety_stock_fraction", 0.0))
    releasable_fraction = 1.0 - safety_stock_fraction
    # residual_arc_capacity is a shared scenario-dependent arc capacity, not a commodity-specific capacity
    residual_arc_capacity = instance["residual_arc_capacity"]
    site_cost = instance["site_cost"]
    selection_budget = instance["selection_budget"]
    arc_cost = instance["arc_cost"]
    penalty = instance["penalty"]
    P_max = instance["P_max"]
    beta = instance["beta"]

    # Convenience adjacency maps
    incoming_arcs = {j: [] for j in N}
    outgoing_arcs = {i: [] for i in N}
    for i, j in A:
        outgoing_arcs[i].append((i, j))
        incoming_arcs[j].append((i, j))

    # --- Build model ---
    model = gp.Model("stochastic_prepositioning_cvar")
    model.Params.OutputFlag = 1 if verbose else 0

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    # --- Decision variables ---
    p = model.addVars(N, vtype=GRB.BINARY, name="p")

    x = model.addVars(
        ((w, i, j, r) for w in Omega for (i, j) in A for r in R),
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    z = model.addVars(
        ((w, j, r) for w in Omega for j in N for r in R),
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="z",
    )

    release = model.addVars(
        ((w, i, r) for w in Omega for i in N for r in R),
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="release",
    )

    eta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")

    xi = model.addVars(
        Omega,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="xi",
    )

    loss = model.addVars(
        Omega,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="loss",
    )

    # --- Objective: CVaR of scenario loss (unmet-demand penalty + transportation cost) ---
    cvar_multiplier = 1.0 / (1.0 - beta)
    model.setObjective(
        eta + cvar_multiplier * gp.quicksum(prob[w] * xi[w] for w in Omega),
        GRB.MINIMIZE,
    )

    # --- Constraints ---

    # Site budget
    model.addConstr(gp.quicksum(p[i] for i in N) <= P_max, name="SiteBudget")

    # Preposition site selection budget
    model.addConstr(
        gp.quicksum(site_cost[i] * p[i] for i in N) <= selection_budget,
        name="SelectionBudget",
    )

    # Inventory release bounds
    # A node may originate flow only through its usable released inventory.
    # If inventory availability is zero in a scenario, the node cannot act as a root,
    # but it may still serve as a transshipment node through inflow/outflow balance.
    # A safety-stock fraction is also retained at each opened node and cannot be released.
    for w in Omega:
        for i in N:
            for r in R:
                availability_factor = 1.0
                if inventory_availability is not None:
                    availability_factor = inventory_availability[w, i, r]

                model.addConstr(
                    release[w, i, r] <= inventory_if_open[i, r] * releasable_fraction * availability_factor * p[i],
                    name=f"InventoryReleaseBound_w{w}_i{i}_r{r}",
                )

    # Node flow balance constraints
    # Inflow plus released inventory plus unmet demand must cover local demand and outbound flow.
    # Unmet demand records local shortfall but should not create artificial outbound supply, so an
    # additional outbound-feasibility constraint is added below to ensure only inflow and release
    # can support forwarding behavior.
    for w in Omega:
        for i in N:
            for r in R:
                model.addConstr(
                    gp.quicksum(x[w, j, i, r] for (j, _) in incoming_arcs[i])
                    + release[w, i, r]
                    + z[w, i, r]
                    == demand[i, r] + gp.quicksum(x[w, i, j, r] for (_, j) in outgoing_arcs[i]),
                    name=f"NodeBalance_w{w}_i{i}_r{r}",
                )

    # Outbound flow feasibility constraints
    # A node may forward only what it physically receives plus what it releases from usable inventory.
    # This prevents unmet demand from artificially creating outbound flow and ensures that only
    # selected prepositioning sites can act as true roots through the release variable.
    for w in Omega:
        for i in N:
            for r in R:
                model.addConstr(
                    gp.quicksum(x[w, i, j, r] for (_, j) in outgoing_arcs[i])
                    <= gp.quicksum(x[w, j, i, r] for (j, _) in incoming_arcs[i]) + release[w, i, r],
                    name=f"OutboundFeasibility_w{w}_i{i}_r{r}",
                )

    # Arc capacity constraints
    for w in Omega:
        for i, j in A:
            model.addConstr(
                gp.quicksum(x[w, i, j, r] for r in R)
                <= residual_arc_capacity[w, i, j],
                name=f"ArcCapacity_w{w}_i{i}_j{j}",
            )

    # Scenario loss definition
    for w in Omega:
        unmet_penalty_expr = gp.quicksum(
            penalty[j, r] * z[w, j, r] for j in N for r in R
        )
        transportation_cost_expr = gp.quicksum(
            arc_cost[i, j] * x[w, i, j, r] for (i, j) in A for r in R
        )
        model.addConstr(
            loss[w] == unmet_penalty_expr + transportation_cost_expr,
            name=f"LossDefinition_w{w}",
        )

    # CVaR linearization
    for w in Omega:
        model.addConstr(xi[w] >= loss[w] - eta, name=f"CVaRExcess_w{w}")

    # --- Optimize ---
    model.optimize()

    # --- Prepare results ---
    results: Dict[str, Any] = {
        "status_code": model.Status,
        "status": _status_to_string(model.Status),
        "objective_value": None,
        "eta": None,
        "selected_sites": [],
        "scenario_losses": {},
        "scenario_transport_cost": {},
        "scenario_unmet_penalty": {},
        "unmet_demand": {},
        "flows": {},
        "positive_flows": [],
        "release": {},
        "node_flow_summary": [],
        "safety_stock_fraction": safety_stock_fraction,
        "model": model,
        "variables": {
            "p": p,
            "x": x,
            "z": z,
            "release": release,
            "eta": eta,
            "xi": xi,
            "loss": loss,
        },
    }

    if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and model.SolCount > 0:
        results["objective_value"] = model.ObjVal
        results["eta"] = eta.X
        results["selected_sites"] = [i for i in N if p[i].X > 0.5]

        for w in Omega:
            results["scenario_losses"][w] = loss[w].X

        for w in Omega:
            transport_cost_val = sum(
                arc_cost[i, j] * x[w, i, j, r].X for (i, j) in A for r in R
            )
            unmet_penalty_val = sum(
                penalty[j, r] * z[w, j, r].X for j in N for r in R
            )
            results["scenario_transport_cost"][w] = transport_cost_val
            results["scenario_unmet_penalty"][w] = unmet_penalty_val

        results["release"] = {}
        for w in Omega:
            for i in N:
                for r in R:
                    val = release[w, i, r].X
                    if val > 1e-6:
                        results["release"][(w, i, r)] = val

        # Node flow summary
        results["node_flow_summary"] = []
        for w in Omega:
            for i in N:
                for r in R:
                    inflow_val = sum(x[w, j, i, r].X for (j, _) in incoming_arcs[i])
                    outflow_val = sum(x[w, i, j, r].X for (_, j) in outgoing_arcs[i])
                    release_val = release[w, i, r].X
                    demand_val = demand[i, r]
                    unmet_val = z[w, i, r].X
                    retained_val = inflow_val + release_val - outflow_val
                    satisfied_val = demand_val - unmet_val

                    if inflow_val > 1e-6 or outflow_val > 1e-6 or release_val > 1e-6 or unmet_val > 1e-6:
                        results["node_flow_summary"].append({
                            "scenario": w,
                            "node": i,
                            "commodity": r,
                            "inflow": inflow_val,
                            "release": release_val,
                            "outflow": outflow_val,
                            "demand": demand_val,
                            "unmet": unmet_val,
                            "retained": retained_val,
                            "satisfied": satisfied_val,
                        })

        for w in Omega:
            for j in N:
                for r in R:
                    val = z[w, j, r].X
                    if val > 1e-6:
                        results["unmet_demand"][(w, j, r)] = val

        for w in Omega:
            for i, j in A:
                for r in R:
                    val = x[w, i, j, r].X
                    if val > 1e-6:
                        results["flows"][(w, i, j, r)] = val
                        results["positive_flows"].append((w, i, j, r, val))

    return results



def print_solution_summary(results: Dict[str, Any], max_flows: int = 20) -> None:
    """
    Print a compact summary of the stochastic model solution.
    """
    print("\n=== STOCHASTIC MODEL SUMMARY ===")
    print(f"Status: {results['status']}")

    if results["objective_value"] is None:
        print("No feasible solution available.")
        return

    print(f"Objective value: {results['objective_value']:.4f}")
    print(f"Eta: {results['eta']:.4f}")
    print(f"Selected sites: {results['selected_sites']}")

    print("\nScenario losses:")
    for w, val in sorted(results["scenario_losses"].items()):
        print(f"  Scenario {w}: {val:.4f}")

    if results.get("scenario_transport_cost"):
        print("\nScenario transport costs:")
        for w, val in sorted(results["scenario_transport_cost"].items()):
            print(f"  Scenario {w}: {val:.4f}")

    if results.get("scenario_unmet_penalty"):
        print("\nScenario unmet-demand penalties:")
        for w, val in sorted(results["scenario_unmet_penalty"].items()):
            print(f"  Scenario {w}: {val:.4f}")

    # --- Aggregate release vs demand ---
    total_release = sum(results.get("release", {}).values())
    total_unmet = sum(results.get("unmet_demand", {}).values())
    total_demand_implied = total_release + total_unmet

    print("\nAggregate supply-demand summary:")
    print(f"  Total released from PPLs: {total_release:.4f}")
    print(f"  Total unmet demand: {total_unmet:.4f}")
    print(f"  Total demand (implied): {total_demand_implied:.4f}")

    # --- Release from non-PPL nodes (should be zero) ---
    selected = set(results.get("selected_sites", []))
    non_ppl_release = sum(
        val for (w, i, r), val in results.get("release", {}).items() if i not in selected
    )

    print("\nNon-PPL Flow Check:")
    print(f"  Total release from non-PPL nodes: {non_ppl_release:.4f}")

    print("\nNode flow balance samples:")
    for row in results.get("node_flow_summary", [])[:max_flows]:
        print(
            "  "
            f"w={row['scenario']}, node={row['node']}, {row['commodity']}: "
            f"in={row['inflow']:.4f}, rel={row['release']:.4f}, out={row['outflow']:.4f}, "
            f"dem={row['demand']:.4f}, unmet={row['unmet']:.4f}, "
            f"retained={row['retained']:.4f}, satisfied={row['satisfied']:.4f}"
        )

    if results["unmet_demand"]:
        print("\nPositive unmet demand entries:")
        for key, val in list(results["unmet_demand"].items())[:max_flows]:
            print(f"  {key}: {val:.4f}")
    else:
        print("\nNo unmet demand in the reported solution.")

    if results["positive_flows"]:
        print("\nPositive flows (first few):")
        for item in results["positive_flows"][:max_flows]:
            w, i, j, r, val = item
            print(f"  w={w}, {i}->{j}, {r}: {val:.4f}")



def _status_to_string(status_code: int) -> str:
    """
    Convert Gurobi status code to readable text.
    """
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return status_map.get(status_code, f"STATUS_{status_code}")