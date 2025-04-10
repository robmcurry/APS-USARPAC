from pyomo.environ import *
import random


from pyomo.environ import *


def build_deterministic_vrp_model():
    # Create a Concrete Model
    model = ConcreteModel()
    node_list = [1, 2, 3, 4]
    commodity_list = ["Commodity1", "Commodity2"]
    time_period_list = range(1, 4)  # Time periods 1, 2, 3
    time_period_list2 = range(0, 4)  # Time periods 1, 2, 3
    vehicle_list = range(1, 3)  # Vehicles 1, 2
    arc_list = [(1, 2), (2, 3), (3, 4), (4, 1)]  # Directed arcs

    # Initialize Sets from lists
    model.N = Set(initialize=node_list)
    model.C = Set(initialize=commodity_list)
    model.T = Set(initialize=time_period_list)
    model.Tprime = Set(initialize=time_period_list2)
    model.V = Set(initialize=vehicle_list)
    model.A = Set(dimen=2, initialize=arc_list)

    ### Parameters ###
    # Initialize Parameters
    model.delta = Param(
        model.N, model.C, model.T,
        initialize={
            (1, "Commodity1", 1): 5,
            (1, "Commodity1", 2): 10,  # Add this missing value
            (2, "Commodity2", 2): 20
        },
        default=10,
        within=NonNegativeReals
    )

    model.nu = Param(
        model.N, model.C,
        initialize={(1, "Commodity1"): 1.0, (2, "Commodity2"): 1.5},
        default=0,
        within=NonNegativeReals
    )
    model.M = Param(initialize=1000, within=NonNegativeReals)  # Big-M, example value
    model.m = Param(
        model.N,
        initialize={1: 30, 2: 50, 3: 70, 4: 20},
        default=0,
        within=NonNegativeReals
    )
    model.L = Param(
        model.C,
        initialize={"Commodity1": 1, "Commodity2": 1},
        default=0,
        within=NonNegativeReals
    )
    model.P = Param(initialize=2, mutable=True, within=NonNegativeReals)  # Maximum APS locations
    model.d = Param(
        model.N, model.C,
        initialize={(1, "Commodity1"): 10, (2, "Commodity2"): 20, (3, "Commodity1"): 15},
        default=0,
        within=NonNegativeReals
    )
    model.b = Param(
        model.C,
        initialize={"Commodity1": 0, "Commodity2": 0},
        default=0,
        within=NonNegativeReals
    )
    model.M_node = Param(
        model.N, model.T,
        initialize={(1, 1): 50, (2, 2): 75, (3, 3): 100},
        default=0,
        within=NonNegativeReals
    )
    model.mu = Param(
        model.V,
        initialize={1: 150, 2: 200},
        default=0,
        within=NonNegativeReals
    )
    model.ell = Param(
        model.N, model.C,
        initialize={(1, "Commodity1"): 2, (2, "Commodity2"): 3},
        default=0,
        within=NonNegativeReals
    )

    ### Variables ###
    model.x = Var(model.A, model.C, model.T, model.V, domain=NonNegativeReals)  # Commodity flow on arcs
    model.y = Var(model.N, model.C, model.T, domain=NonNegativeReals)  # Satisfied demand
    model.z = Var(model.N, model.C, model.T, domain=NonNegativeReals)  # Unmet demand penalty
    model.w = Var(model.N, model.C, model.Tprime, domain=NonNegativeReals)  # Inventory levels
    model.m_i = Var(model.N, domain=NonNegativeReals)  # Vehicle allocation per node
    model.q = Var(model.N, model.C, domain=NonNegativeReals)  # APS demand allocation variable
    model.r = Var(model.N, model.C, domain=Binary)  # Binary APS for commodities
    model.p = Var(model.N, domain=Binary)  # Binary APS for nodes
    model.alpha = Var(model.N, model.C, domain=NonNegativeReals)  # Safety stock variables
    model.bar_x = Var(model.A, model.T, model.V, domain=Binary)  # Vehicle flow variable
    model.bar_w = Var(model.N, model.Tprime, model.V, domain=Binary)  # Binary vehicle inventory variables

    ### Objective Function ###
    def objective_rule(model):
        return sum(
            sum(
                sum(
                    model.delta[i, c, t] * model.z[i, c, t] for t in model.T
                ) + model.nu[i, c] * model.alpha[i, c]
                for c in model.C
            )
            for i in model.N
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    ## Constraints ###
    def max_vehicles_rule(model):
        return sum(model.m_i[i] for i in model.N) <= 10

    model.MaximumVehicles = Constraint(rule=max_vehicles_rule)

    def aps_binary_constraint1_rule(model, i, c):
        return model.q[i, c] <= model.M * model.r[i, c]

    model.APSBinary1 = Constraint(model.N, model.C, rule=aps_binary_constraint1_rule)

    # def aps_binary_constraint2_rule(model, i, c):
    #     return model.r[i, c] <= model.M * model.m_i[i]
    #
    # model.APSBinary2 = Constraint(model.N, model.C, rule=aps_binary_constraint2_rule)

    def aps_binary_constraint3_rule(model, i, c):
        return model.r[i, c] <= model.M * model.p[i]

    model.APSBinary3 = Constraint(model.N, model.C, rule=aps_binary_constraint3_rule)

    def max_aps_locations_rule(model):
        return sum(model.p[i] for i in model.N) <= model.P

    model.MaxAPSLocations = Constraint(rule=max_aps_locations_rule)

    def min_aps_per_commodity_rule(model, c):
        return sum(model.r[i, c] for i in model.N) >= model.L[c]

    model.MinAPSPerCommodity = Constraint(model.C, rule=min_aps_per_commodity_rule)

    def demand_constraints_rule(model, i, c, t):
        return sum(model.y[i, c, v] for v in range(1, t + 1)) + model.z[i, c, t] == model.d[i, c]
    model.DemandConstraints = Constraint(model.N, model.C,model.T, rule=demand_constraints_rule)

    def commodity_flow_balance_rule(model, i, c, t):
        return sum(model.x[i, j, c, t, v] for j in model.N if (i, j) in model.A for v in model.V) - sum(
            model.x[j, i, c, t, v] for j in model.N if (j, i) in model.A for v in model.V) + model.w[i, c, t] - (
            model.w[i, c, t - 1] ) + model.y[i, c, t] == 0

    model.CommodityFlowBalance = Constraint(model.N, model.C, model.T, rule=commodity_flow_balance_rule)

    def starting_inventory_rule(model, i, c):
        return model.w[i, c, 0] == model.q[i, c]
    model.StartingInventory = Constraint(model.N, model.C, rule=starting_inventory_rule)

    def max_node_cap_rule(model, i, t):
        return sum(model.b[c] * model.w[i, c, t] for c in model.C) <= model.M_node[i, t]

    model.MaxNodeCapacity = Constraint(model.N, model.T, rule=max_node_cap_rule)

    def safety_stock_constraint_rule(model, i, c):
        return model.alpha[i, c] >= model.ell[i, c] - (model.q[i, c] - sum(
            model.x[i, j, c, t, v] for j in model.N if (i, j) in model.A for v in model.V for t in model.T) - sum(
            model.x[j, i, c, t, v] for j in model.N if (j, i) in model.A for v in model.V for t in model.T))

    model.SafetyStockConstraint = Constraint(model.N, model.C, rule=safety_stock_constraint_rule)

    def vehicle_flow_balance_rule(model, i, t, v):
        return sum(model.bar_x[i, j, t, v] for j in model.N if (i, j) in model.A) - sum(
            model.bar_x[j, i, t, v] for j in model.N if (j, i) in model.A) + \
            model.bar_w[i, t, v] - (model.bar_w[i, t - 1, v] if t > 0 else 0) == 0

    model.VehicleFlowBalance = Constraint(model.N, model.T, model.V, rule=vehicle_flow_balance_rule)

    def vehicles_return_rule(model, i, t, v):
        return model.bar_w[i, model.T.last(), v] == model.m_i[i]

    model.VehiclesReturn = Constraint(model.N, model.T, model.V, rule=vehicles_return_rule)

    return model

def build_stochastic_vrp_model():
    # Create a Concrete Model
    model = ConcreteModel()
    node_list = [1, 2, 3, 4]
    commodity_list = ["Commodity1", "Commodity2"]
    scenario_list = range(1, 3)  # Scenarios 1, 2
    time_period_list = range(1, 4)  # Time periods 1, 2, 3
    time_period_list2 = range(0, 4)  # Time periods 1, 2, 3
    vehicle_list = range(1, 3)  # Vehicles 1, 2
    arc_list = [(1, 2), (2, 3), (3, 4), (4, 1)]  # Directed arcs

    # Initialize Sets from lists
    model.N = Set(initialize=node_list)
    model.C = Set(initialize=commodity_list)
    model.S = Set(initialize=scenario_list)
    model.T = Set(initialize=time_period_list)
    model.Tprime = Set(initialize=time_period_list2)
    model.V = Set(initialize=vehicle_list)
    model.A = Set(dimen=2, initialize=arc_list)

    ### Parameters ###
    # Initialize Parameters
    model.phi = Param(model.S, initialize={1: 0.6, 2: 0.4},
                      within=NonNegativeReals)  # Scenario weights: e.g., 60% and 40%
    # Initialize all required values for delta
    model.delta = Param(
        model.N, model.C, model.T, model.S,
        initialize={
            (1, "Commodity1", 1, 1): 5,
            (1, "Commodity1", 2, 1): 10,  # Add this missing value
            (2, "Commodity2", 2, 1): 20
        },
    default=0,  # Default value for any missing (i, c, t, s) combination

        within=NonNegativeReals
    )

    model.nu = Param(
        model.N, model.C,
        initialize={(1, "Commodity1"): 0.0, (2, "Commodity2"): 0},     default=0,  # Default value for any missing (i, c, t, s) combination
 # Safety stock penalties
        within=NonNegativeReals
    )
    model.M = Param(initialize=1000, within=NonNegativeReals)  # Big-M, example value
    model.m = Param(
        model.N,
        initialize={1: 3, 2: 5, 3: 7, 4: 2},    default=0,  # Default value for any missing (i, c, t, s) combination
  # Maximum vehicles for node 1, 2, 3, 4
        within=NonNegativeReals
    )
    model.L = Param(
        model.C,
        initialize={"Commodity1": 2, "Commodity2": 1},    default=0,  # Default value for any missing (i, c, t, s) combination
  # Minimum APS constraint per commodity
        within=NonNegativeReals
    )
    model.P = Param(initialize=10, mutable=True, within=NonNegativeReals)  # Maximum APS locations
    model.d = Param(
        model.N, model.C, model.S,
        initialize={(1, "Commodity1", 1): 10, (2, "Commodity2", 1): 20, (3, "Commodity1", 2): 15},    default=0,  # Default value for any missing (i, c, t, s) combination
  # Demand
        within=NonNegativeReals
    )
    model.b = Param(
        model.C,
        initialize={"Commodity1": 5, "Commodity2": 3},    default=0,  # Default value for any missing (i, c, t, s) combination
  # Weight of commodities
        within=NonNegativeReals
    )
    model.M_node = Param(
        model.N, model.T,
        initialize={(1, 1): 50, (2, 2): 75, (3, 3): 100},     default=0,  # Default value for any missing (i, c, t, s) combination
 # Max capacity of node in each time period
        within=NonNegativeReals
    )
    model.mu = Param(
        model.V,
        initialize={1: 15, 2: 20},     default=0,  # Default value for any missing (i, c, t, s) combination
 # Vehicle capacity
        within=NonNegativeReals
    )
    model.ell = Param(
        model.N, model.C,
        initialize={(1, "Commodity1"): 2, (2, "Commodity2"): 3},     default=0,  # Default value for any missing (i, c, t, s) combination
 # Lower bound for safety stock
        within=NonNegativeReals
    )

    ### Variables ###
    model.x = Var(model.A, model.C, model.T, model.V, model.S, domain=NonNegativeReals)  # Commodity flow on arcs
    model.y = Var(model.N, model.C, model.T, model.S, domain=NonNegativeReals)  # Satisfied demand
    model.z = Var(model.N, model.C, model.T, model.S, domain=NonNegativeReals)  # Unmet demand penalty
    model.w = Var(model.N, model.C, model.Tprime, model.S, domain=NonNegativeReals)  # Inventory levels
    model.m_i = Var(model.N, domain=NonNegativeReals)  # Vehicle allocation per node
    model.q = Var(model.N, model.C, domain=NonNegativeReals)  # APS demand allocation variable
    model.r = Var(model.N, model.C, domain=Binary)  # Binary APS for commodities
    model.p = Var(model.N, domain=Binary)  # Binary APS for nodes
    model.alpha = Var(model.N, model.C, model.S, domain=NonNegativeReals)  # Safety stock variables
    model.bar_x = Var(model.A, model.T, model.V, model.S, domain=Binary)  # Vehicle flow variable
    model.bar_w = Var(model.N, model.Tprime, model.V, model.S, domain=Binary)  # Binary vehicle inventory variables

    ### Objective Function ###
    def objective_rule(model):
        return sum(
            model.phi[s] * (
                sum(
                    sum(
                        sum(model.delta[i, c, t, s] * model.z[i, c, t, s] for t in model.T) +
                        model.nu[i, c] * model.alpha[i, c, s]
                        for c in model.C
                    )
                    for i in model.N
                )
            )
            for s in model.S
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    ## Constraints ###
    def max_vehicles_rule(model):
        return sum(model.m_i[i] for i in model.N) <= 10
    model.MaximumVehicles = Constraint(rule=max_vehicles_rule)

    def aps_binary_constraint1_rule(model, i, c):
        return model.q[i, c] <= model.M * model.r[i, c]

    model.APSBinary1 = Constraint(model.N, model.C, rule=aps_binary_constraint1_rule)

    def aps_binary_constraint2_rule(model, i, c):
        return model.r[i, c] <= model.M * model.m_i[i]

    model.APSBinary2 = Constraint(model.N, model.C, rule=aps_binary_constraint2_rule)

    def aps_binary_constraint3_rule(model, i, c):
        return model.r[i, c] <= model.M * model.p[i]

    model.APSBinary3 = Constraint(model.N, model.C, rule=aps_binary_constraint3_rule)

    def max_aps_locations_rule(model):
        return sum(model.p[i] for i in model.N) <= model.P

    model.MaxAPSLocations = Constraint(rule=max_aps_locations_rule)

    def min_aps_per_commodity_rule(model, c):
        return sum(model.r[i, c] for i in model.N) >= model.L[c]

    model.MinAPSPerCommodity = Constraint(model.C, rule=min_aps_per_commodity_rule)

    def demand_constraints_rule(model, i, c, s):
        return sum(model.y[i, c, t, s] for t in model.T) + model.z[i, c, model.T.last(), s] == model.d[i, c, s]

    model.DemandConstraints = Constraint(model.N, model.C, model.S, rule=demand_constraints_rule)

    def commodity_flow_balance_rule(model, i, c, t, s):
        return sum(model.x[i, j, c, t, v, s] for j in model.N if (i,j) in model.A for v in model.V) - sum(
            model.x[j, i, c, t, v, s] for j in model.N if (j,i) in model.A for v in model.V) + model.w[i, c, t, s] - (
            model.w[i, c, t - 1, s] if t > 1 else 0) + model.y[i, c, t, s] == 0

    model.CommodityFlowBalance = Constraint(model.N, model.C, model.T, model.S, rule=commodity_flow_balance_rule)

    def max_node_cap_rule(model, i, t, s):
        return sum(model.b[c] * model.w[i, c,t, s]  for c in model.C) <= model.M_node[i, t]

    model.MaxNodeCapacity = Constraint(model.N, model.T, model.S, rule=max_node_cap_rule)


    def starting_inventory_rule(model, i, c,s):
        return model.w[i, c, 0,s] == model.q[i, c]
    model.StartingInventory = Constraint(model.N, model.C, model.S, rule=starting_inventory_rule)


    def safety_stock_constraint_rule(model, i, c, s):
        return model.alpha[i, c, s] >= model.ell[i, c] - (model.q[i, c] - sum(
            model.x[i, j, c, t, v, s] for j in model.N if (i,j) in model.A for v in model.V for t in model.T) - sum(model.x[j, i, c, t, v, s] for j in model.N if (j,i) in model.A for v in model.V for t in model.T))

    model.SafetyStockConstraint = Constraint(model.N, model.C, model.S, rule=safety_stock_constraint_rule)

    def vehicle_flow_balance_rule(model, i, t, v, s):
        return sum(model.bar_x[i, j, t, v, s] for j in model.N if (i,j) in model.A) - sum(model.bar_x[j, i, t, v, s] for j in model.N if (j,i) in model.A) + \
            model.bar_w[i, t, v, s] - (model.bar_w[i, t - 1, v, s] if t > 0 else 0) == 0

    model.VehicleFlowBalance = Constraint(model.N, model.T, model.V, model.S, rule=vehicle_flow_balance_rule)

    def vehicles_return_rule(model, i, t, v, s):
        return model.bar_w[i, model.T.last(), v, s] == model.m_i[i]

    model.VehiclesReturn = Constraint(model.N, model.T, model.V, model.S, rule=vehicles_return_rule)

    ### Additional Aspects ###
    # Define more constraints based on your use case.

    return model


if __name__ == "__main__":
    # Instantiate the model
    stochastic_vrp_model = build_stochastic_vrp_model()
    deterministic_vrp_model = build_deterministic_vrp_model()

    # Solve with a solver
    solver = SolverFactory("gurobi")  # Replace with a solver like CPLEX or Gurobi, if needed
    solver.solve(stochastic_vrp_model)
    solver.solve(deterministic_vrp_model)

    # Display only the variable values for both models
    # print("Stochastic Model Variable Values:")
    # stochastic_vrp_model.display(ostream=None)

    print("\nDeterministic Model Variable Values:")
    deterministic_vrp_model.display(ostream=None)

    # stochastic_vrp_model.display()
    #
    # deterministic_vrp_model.display()



