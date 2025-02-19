from pyomo.environ import *
import random
def build_min_unmet_demand_model(num_locations, num_commodities, num_time_periods, num_APS_locations, num_modes=3):
    # Create a model
    
    
    model = ConcreteModel()


    # Define sets (example sizes for each set)
    model.V = Set(initialize=range(num_locations))  # Total Nodes
    model.C = Set(initialize=range(1, num_commodities))  # Total Commodities
    model.T = Set(initialize=range(num_time_periods))  # Total Time periods
    model.Tminus = Set(initialize=range(1, num_time_periods))  # Time periods without 0
    model.A = Set(within=model.V * model.V,
                  initialize=[(i, j) for i in model.V for j in model.V if i != j])  # Fully connected DAG (Directed Acylic Graph
    model.V_s = Set(initialize=range(1, num_APS_locations))  # More special nodes

    model.M = Set(initialize=range(1, num_modes + 1))  # Modes of transportation (1 to num_modes)

    ###penalty for a unit of unmet demand at node i for commodity c at the end of time period t
    model.p = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(10, 50) for i in model.V for c in model.C for t in model.T})

    ###demand for each commodity c at node i
    model.d = Param(model.V, model.C,
                    initialize={(i, c): random.randint(30, 100) for i in model.V for c in model.C})

    ###maximum allowable allocation for commodity c at node i and time t
    model.m = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(50, 100) for i in model.V for c in model.C for t in model.T})
                    
    ###maximum capacity of arc (i,j) for commodity c during time t
    model.mu = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(0, 10) for (i, j) in model.A for c in model.C for t in model.T})

    ###Maximum allowable space for commodity c at node i
    model.M = Param(model.V, model.C,
                    initialize={(i, c): random.randint(5000, 7000) for i in model.V for c in model.C})
                    
    ###Maximum allowable space among all commodities during time t
    model.Mt = Param(model.V, model.T,
                    initialize={(i, t): random.randint(10, 20) for i in model.V for t in model.T})
                    
    ###The unit risk associated with sending commodity c on arc (i,j) during time t
    model.r1 = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(1, 5) for (i, j) in model.A for c in model.C for t in model.T})

    ###The unit risk of storing commodity c at node i at the end of time t
    model.r2 = Param(model.V, model.C, model.T,
                     initialize={(i, c, t): random.randint(1, 5) for i in model.V for c in model.C for t in model.T})

    ##The unit risk associated with pre-positioning commodity c at node i
    model.r3 = Param(model.V, model.C,
                     initialize={(i, c): random.randint(1, 5) for i in model.V for c in model.C})

    ###The minimum required supply of commodity c at location i if commodity c is pre-positioned at location i
    model.ell = Param(model.V, model.C,
                      initialize={(i, c): round(random.uniform(1, 3), 0) for i in model.V for c in model.C})

    ###Upper bound on the number of potential APS locations
    model.P = Param(initialize=5)

    # Maximum allowable risk
    model.R = Param(initialize=50000)

    ###weight of volume of a single unit of commodity c
    model.b = Param(model.C, initialize={c: round(random.uniform(1.0, 3.0), 2) for c in model.C})

    ###The minimum number of APS locations starting with commodity c
    model.L = Param(model.C, initialize={c: random.randint(1, 2) for c in model.C})

    # Define variables

    ##The amount of commodity c consumed to meet demand at node i in V
    ##at the end of time t
    model.y = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # y_ict

    ##The amount of commodity c unmet demand at node i in V at the end of time t
    model.z = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # z_ict

    ##the units of flow on arc (i,j) for commodity c during time t
    model.x = Var(model.A, model.C, model.T, within=NonNegativeReals)  # x_ijct

    ##The units of commodity c stored at node i at the end of time period t
    model.w = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # w_ict

    ##the units of units of commodity c pre-positioned at node i
    model.s_var = Var(model.V, model.C, within=NonNegativeIntegers)  # s_ic

    ##equals 1 if commodity c is pre positioned at node i
    model.p_var = Var(model.V, model.C, within=Binary)  # p_ic

    ##equals 1 if node i is pre-positioned with any commodity
    model.q_var = Var(model.V, within=Binary)  # q_i

    # Objective function
    def objective_function(model):
        return sum(model.p[i, c, t] * model.z[i, c, t] for i in model.V for c in model.C for t in model.T)
    model.obj = Objective(rule=objective_function, sense=minimize)


    ###defines the demand constraints here
    # Constraint (1)
    def commodity_demand_balance(model, i, c, t):
        return sum(model.y[i, c, t_prime] for t_prime in model.T if t_prime <= t) + model.z[i, c, t] == model.d[i, c]

    model.commodity_demand_balance = Constraint(model.V, model.C, model.T, rule=commodity_demand_balance)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs
    def constraint3(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]

    model.supply_equals_initial_inventory_constraint = Constraint(model.V_s, model.C, rule=constraint3)

    ## Constraint (4)
    ##These constraints ensure that the total commodity c leftover is equal to the total flow of c supplied minutes the amount consumed
    def constraint4(model, c):
        return sum(model.w[i, c, model.T.last()] for i in model.V) == sum(model.s_var[j, c] for j in model.V_s) - sum(model.y[i, c, t] for i in model.V for t in model.T)
    model.constraint4 = Constraint(model.C, rule=constraint4)

    ## Constraint (5)
    #Cannot exced m number of units at node i for commodity c at time t
    def constraint5(model, i, c, t):
        return model.w[i, c, t] <= model.m[i, c, t]
    model.constraint5 = Constraint(model.V, model.C, model.T, rule=constraint5)

    ## Constraint (6)
    ##cannot exceed max capacity on arc (i,j)
    def constraint6(model, i, j, c, t):
        return model.x[i, j, c, t] <= model.mu[i, j, c, t]
    model.constraint6 = Constraint(model.A, model.C, model.T, rule=constraint6)

    ## Constraint (7)
    ## cannot exceed the total budgeted cost out of i during t
    def constraint7(model, i, t):
        return sum(model.b[c] * model.w[i, c, t] for c in model.C) <= model.Mt[i, t]
    model.constraint7 = Constraint(model.V, model.T, rule=constraint7)

    ## Constraint (10)
    ## Flow balance constraints
    ## Keeps up with the flow out, the flow in, the demand consumed, and the demand that is stored there between flow periods
    def constraint10(model, i, c, t):
        return sum(model.x[i, j, c, t] for j in model.V if (i, j) in model.A) - sum(model.x[j, i, c, t] for j in model.V if (j, i) in model.A) + model.w[i, c, t] - model.w[i, c, t-1] + model.y[i, c, t] == 0
    model.constraint10 = Constraint(model.V, model.C, model.Tminus, rule=constraint10)

    ## Constraint (10.5)
    ##Ensures that if we store commodity c at i then we must have opened that node i for commodity c
    def constraint10_5(model, i, c):
        return model.s_var[i, c] <= model.M[i, c] * model.p_var[i, c]
    model.constraint10_5 = Constraint(model.V, model.C, rule=constraint10_5)

    ## Constraint (10.6)
    ##Ensures that if we store commodity c at node i then we must have opened i
    def constraint10_6(model, i, c):
        return model.p_var[i, c] <= model.q_var[i]
    model.constraint10_6 = Constraint(model.V, model.C, rule=constraint10_6)

    ## Constraint (10.7)
    ##Ensures that we don't open more than P number of APSs
    def constraint10_7(model):
        return sum(model.q_var[i] for i in model.V) <= model.P
    model.constraint10_7 = Constraint(rule=constraint10_7)

    ## Constraint (10.8)
    ## Makes sure that we have at least L units of commodity c among all starting APS locations
    def minimum_aps_requirement(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]

    model.minimum_aps_requirement_constraint = Constraint(model.C, rule=minimum_aps_requirement)

    ## Constraint (10.9)
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def constraint10_9(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]
    model.constraint10_9 = Constraint(model.V, model.C, rule=constraint10_9)

    ## Constraint (10.91)
    ##Enforces that we don't take on too much risk
    def constraint10_91(model):
        return sum( sum( sum(model.r1[i, j, c, t] * model.x[i, j, c, t] for (i, j) in model.A) +
                       sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in model.C) <= model.R
    model.constraint10_91 = Constraint(rule=constraint10_91)


    

    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model)

    # Output the values of the variables
    for c in model.C:
        for i in model.V:
            for j in model.V:
                for t in model.T:
                    if i != j and model.x[i, j, c, t].value > 0:
                        print(
                            f"Commodity {c} flow on arc ({i}, {j}) during time period {t}: {model.x[i, j, c, t].value}")

    for i in model.V:
        for c in model.C:
            for t in model.T:
                if model.y[i, c, t].value > 0:
                    print(f"Commodity {c} consumed by node {i} at end of time period {t}: {model.y[i, c, t].value}")

    for c in model.C:
        for i in model.V:
            if model.s_var[i, c].value > 0:
                print(f"Commodity {c} pre-positioned at node {i}: {model.s_var[i, c].value}")


def build_risk_model(num_locations, num_commodities, num_time_periods, num_APS_locations):
    # Create a model
    model = ConcreteModel()


       # Define sets (example sizes for each set)
    model.V = Set(initialize=range(num_locations))  # Total Nodes
    model.C = Set(initialize=range(1, num_commodities))  # Total Commodities
    model.T = Set(initialize=range(num_time_periods))  # Total Time periods
    model.Tminus = Set(initialize=range(1, num_time_periods))  # Time periods without 0
    model.A = Set(within=model.V * model.V,
                  initialize=[(i, j) for i in model.V for j in model.V if i != j])  # Fully connected DAG (Directed Acylic Graph
    model.V_s = Set(initialize=range(1, num_APS_locations))  # More special nodes
    # model.M = Set(initialize=range(1, num_modes))  # Set containing mode IDs


    ###penalty for a unit of unmet demand at node i for commodity c at the end of time period t
    model.p = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(10, 50) for i in model.V for c in model.C for t in model.T})

    ###demand for each commodity c at node i
    model.d = Param(model.V, model.C,
                    initialize={(i, c): random.randint(10, 20) for i in model.V for c in model.C})

    ###maximum allowable allocation for commodity c at node i and time t
    model.m = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(50, 100) for i in model.V for c in model.C for t in model.T})
                    
    ###maximum capacity of arc (i,j) for commodity c during time t
    model.mu = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(0, 10) for (i, j) in model.A for c in model.C for t in model.T})

    ###Maximum allowable space for commodity c at node i
    model.M = Param(model.V, model.C,
                    initialize={(i, c): random.randint(5000, 7000) for i in model.V for c in model.C})
                    
    ###Maximum allowable space among all commodities during time t
    model.Mt = Param(model.V, model.T,
                    initialize={(i, t): random.randint(10, 20) for i in model.V for t in model.T})
                    
    ###The unit risk associated with sending commodity c on arc (i,j) during time t
    model.r1 = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(1, 5) for (i, j) in model.A for c in model.C for t in model.T})

    ###The unit risk of storing commodity c at node i at the end of time t
    model.r2 = Param(model.V, model.C, model.T,
                     initialize={(i, c, t): random.randint(1, 5) for i in model.V for c in model.C for t in model.T})

    ##The unit risk associated with pre-positioning commodity c at node i
    model.r3 = Param(model.V, model.C,
                     initialize={(i, c): random.randint(1, 5) for i in model.V for c in model.C})

    ###The minimum required supply of commodity c at location i if commodity c is pre-positioned at location i
    model.ell = Param(model.V, model.C,
                      initialize={(i, c): round(random.uniform(1, 3), 0) for i in model.V for c in model.C})

    ###Upper bound on the number of potential APS locations
    model.P = Param(initialize=4)

    # Maximum allowable risk
    model.R = Param(initialize=50000)

    ###weight of volume of a single unit of commodity c
    model.b = Param(model.C, initialize={c: round(random.uniform(1.0, 3.0), 2) for c in model.C})

    ###The minimum number of APS locations starting with commodity c
    model.L = Param(model.C, initialize={c: random.randint(1, 2) for c in model.C})

    # Define variables

    ##The amount of commodity c consumed to meet demand at node i in V
    ##at the end of time t
    model.y = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # y_ict

    ##The amount of commodity c unmet demand at node i in V at the end of time t
    model.z = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # z_ict

    ##the units of flow on arc (i,j) for commodity c during time t
    model.x = Var(model.A, model.C, model.T, within=NonNegativeReals)  # x_ijct

    ##The units of commodity c stored at node i at the end of time period t
    model.w = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # w_ict

    ##the units of units of commodity c pre-positioned at node i
    model.s_var = Var(model.V, model.C, within=NonNegativeIntegers)  # s_ic

    ##equals 1 if commodity c is pre positioned at node i
    model.p_var = Var(model.V, model.C, within=Binary)  # p_ic

    ##equals 1 if node i is pre-positioned with any commodity
    model.q_var = Var(model.V, within=Binary)  # q_i

    # Objective function
    def objective_function(model):
        return sum( sum( sum(model.r1[i, j, c, t] * model.x[i, j, c, t] for (i, j) in model.A) +
                       sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in model.C)
    model.obj = Objective(rule=objective_function, sense=minimize)


    ###defines the demand constraints here
    # Constraint (1)
    def constraint1(model, i, c, t):
        return sum(model.y[i, c, t_prime] for t_prime in model.T if t_prime <= t) + model.z[i, c, t] >= model.d[i, c]

    model.demand_satisfaction_constraint = Constraint(model.V, model.C, model.T, rule=constraint1)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs
    def initial_inventory_equals_prepositioned_units(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]

    model.initial_inventory_equals_prepositioned_units_constraint = Constraint(model.V_s, model.C,
                                                                               rule=initial_inventory_equals_prepositioned_units)

    ## Constraint (4)
    ##These constraints ensure that the total commodity c leftover is equal to the total flow of c supplied minutes the amount consumed
    def constraint4(model, c):
        return sum(model.w[i, c, model.T.last()] for i in model.V) == sum(model.s_var[j, c] for j in model.V_s) - sum(model.y[i, c, t] for i in model.V for t in model.T)

    model.flow_balance_at_end_period_constraint = Constraint(model.C, rule=constraint4)

    ## Constraint (5)
    #Cannot exced m number of units at node i for commodity c at time t
    def constraint5(model, i, c, t):
        return model.w[i, c, t] <= model.m[i, c, t]
    model.constraint5 = Constraint(model.V, model.C, model.T, rule=constraint5)

    ## Constraint (6)
    ##cannot exceed max capacity on arc (i,j)
    def constraint6(model, i, j, c, t):
        return model.x[i, j, c, t] <= model.mu[i, j, c, t]
    model.constraint6 = Constraint(model.A, model.C, model.T, rule=constraint6)

    ## Constraint (7)
    ## cannot exceed the total budgeted cost out of i during t
    def constraint7(model, i, t):
        return sum(model.b[c] * model.w[i, c, t] for c in model.C) <= model.Mt[i, t]

    model.total_cost_budget_constraint = Constraint(model.V, model.T, rule=constraint7)

    ## Constraint (10)
    ## Flow balance constraints
    ## Keeps up with the flow out, the flow in, the demand consumed, and the demand that is stored there between flow periods
    def constraint10(model, i, c, t):
        return sum(model.x[i, j, c, t] for j in model.V if (i, j) in model.A) - sum(model.x[j, i, c, t] for j in model.V if (j, i) in model.A) + model.w[i, c, t] - model.w[i, c, t-1] + model.y[i, c, t] == 0
    model.constraint10 = Constraint(model.V, model.C, model.Tminus, rule=constraint10)

    ## Constraint (10.5)
    ##Ensures that if we store commodity c at i then we must have opened that node i for commodity c
    def constraint10_5(model, i, c):
        return model.s_var[i, c] <= model.M[i, c] * model.p_var[i, c]

    model.storage_opening_cost_constraint = Constraint(model.V, model.C, rule=constraint10_5)

    ## Constraint (10.6)
    ##Ensures that if we store commodity c at node i then we must have opened i
    def constraint10_6(model, i, c):
        return model.p_var[i, c] <= model.q_var[i]

    model.commodity_opening_dependency_constraint = Constraint(model.V, model.C, rule=constraint10_6)

    ## Constraint (10.7)
    ##Ensures that we don't open more than P number of APSs
    def constraint10_7(model):
        return sum(model.q_var[i] for i in model.V) <= model.P

    model.aps_location_limit_constraint = Constraint(rule=constraint10_7)

    ## Constraint (10.8)
    ## Makes sure that we have at least L units of commodity c among all starting APS locations
    def constraint10_8(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]

    model.commodity_minimum_aps_constraint = Constraint(model.C, rule=constraint10_8)

    ## Constraint (10.9)
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def constraint10_9(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]
    model.constraint10_9 = Constraint(model.V, model.C, rule=constraint10_9)

 
    ##Need to add a constraint to make sure that we meet demand at some point.
    
#    def demandconstraint(model, i, c):
#        return sum(model.y[i,c,t] for t in model.T) >= model.d[i,c]
#    model.demandconstraint = Constraint(model.V, model.C, rule=demandconstraint)



    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model)

    print(results)

    # Output the values of the variables
    for c in model.C:
        for i in model.V:
            for j in model.V:
                for t in model.T:
                    if i != j and model.x[i, j, c, t].value > 0:
                        print(
                            f"Commodity {c} flow on arc ({i}, {j}) during time period {t}: {model.x[i, j, c, t].value}")

    for i in model.V:
        for c in model.C:
            for t in model.T:
                if model.y[i, c, t].value > 0:
                    print(f"Commodity {c} consumed by node {i} at end of time period {t}: {model.y[i, c, t].value}")

    for c in model.C:
        for i in model.V:
            if model.s_var[i, c].value > 0:
                print(f"Commodity {c} pre-positioned at node {i}: {model.s_var[i, c].value}")
    print(results)

    # Output the values of the variables
    for c in model.C:
        for i in model.V:
            for j in model.V:
                for t in model.T:
                    if i != j and model.x[i, j, c, t].value > 0:
                        print(
                            f"Commodity {c} flow on arc ({i}, {j}) during time period {t}: {model.x[i, j, c, t].value}")

    for i in model.V:
        for c in model.C:
            for t in model.T:
                if model.y[i, c, t].value > 0:
                    print(f"Commodity {c} consumed by node {i} at end of time period {t}: {model.y[i, c, t].value}")

    for c in model.C:
        for i in model.V:
            if model.s_var[i, c].value > 0:
                print(f"Commodity {c} pre-positioned at node {i}: {model.s_var[i, c].value}")


def main():

    num_locations = 10
    num_commodities = 10
    num_time_periods = 10
    num_APS_locations = 6
    num_modes = 2


    #build_min_unmet_demand_model(num_locations, num_commodities, num_time_periods, num_APS_locations)

    #build_risk_model(num_locations, num_commodities, num_time_periods, num_APS_locations)

    build_multiobjective_model(num_locations, num_commodities, num_time_periods, num_APS_locations, num_modes)


def build_multiobjective_model(num_locations, num_commodities, num_time_periods, num_APS_locations,
                               num_modes):
    # Create a model
    model = ConcreteModel()


       # Define sets (example sizes for each set)
    model.V = Set(initialize=range(num_locations))  # Total Nodes
    model.C = Set(initialize=range(1, num_commodities))  # Total Commodities
    model.T = Set(initialize=range(num_time_periods))  # Total Time periods
    model.Tminus = Set(initialize=range(1, num_time_periods))  # Time periods without 0
    # Set containing mode IDs, representing different modes of transportation
    model.M = Set(initialize=range(1, num_modes))  # Modes of transportation (1 to num_modes)
    #model.IndexedSetM = Set(model.M,initialize=lambda model, m: range(m * 10, (m + 1) * 10))  # Indexed set of sets based on M
    model.A = Set(within=model.V * model.V,
                  initialize=[(i, j) for i in model.V for j in model.V if i != j])  # Fully connected DAG (Directed Acylic Graph
    model.Cm = Set(model.M, within=model.C, doc="Set of precedence commodities for each transport mode m")

    model.V_s = Set(initialize=range(1, num_APS_locations))  # More special nodes

    ###penalty for a unit of unmet demand at node i for commodity c at the end of time period t
    model.p = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(10, 50) for i in model.V for c in model.C for t in model.T})

    ###demand for each commodity c at node i
    model.d = Param(model.V, model.C,
                    initialize={(i, c): random.randint(30, 100) for i in model.V for c in model.C})

    ###maximum allowable allocation for commodity c at node i and time t
    model.m = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(50, 100) for i in model.V for c in model.C for t in model.T})

    ###maximum capacity of arc (i,j) for commodity c during time t
    model.mu = Param(model.A, model.C, model.T, model.M,
                     initialize={(i, j, c, t,m): random.randint(1, 10) for (i, j) in model.A for c in model.C for t in model.T for m in model.M})

    ###Maximum allowable space for commodity c at node i
    model.MM = Param(model.V, model.C,
                    initialize={(i, c): random.randint(5000, 7000) for i in model.V for c in model.C})

    ###Maximum allowable space among all commodities during time t
    model.Mt = Param(model.V, model.T,
                    initialize={(i, t): random.randint(10, 20) for i in model.V for t in model.T})

    ###The unit risk associated with sending commodity c on arc (i,j) during time t
    model.r1 = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(1, 5) for (i, j) in model.A for c in model.C for t in model.T})

    ###The unit risk of storing commodity c at node i at the end of time t
    model.r2 = Param(model.V, model.C, model.T,
                     initialize={(i, c, t): random.randint(1, 5) for i in model.V for c in model.C for t in model.T})

    ##The unit risk associated with pre-positioning commodity c at node i
    model.r3 = Param(model.V, model.C,
                     initialize={(i, c): random.randint(1, 5) for i in model.V for c in model.C})

    ###The minimum required supply of commodity c at location i if commodity c is pre-positioned at location i
    model.ell = Param(model.V, model.C,
                      initialize={(i, c): round(random.uniform(1, 3), 0) for i in model.V for c in model.C})

    ###Upper bound on the number of potential APS locations
    model.P = Param(initialize=4)

    # Maximum allowable risk
    model.R = Param(initialize=50000)

    ###weight of volume of a single unit of commodity c
    model.b = Param(model.C, initialize={c: round(random.uniform(1.0, 3.0), 2) for c in model.C})

    ###The minimum number of APS locations starting with commodity c
    model.L = Param(model.C, initialize={c: random.randint(1, 2) for c in model.C})


    model.Q = Param(model.M, initialize={m: random.randint(1, 2) for m in model.M})

    # Volume per unit of commodity c
    model.v = Param(model.M, initialize={m: random.randint(50, 100) for m in model.M})  # Volume capacity for mode m
    model.Q = Param(model.M,
                    initialize={m: random.randint(5, 15) for m in model.M})  # Maximum number of mode m available

    # ***** Parameters *****
    # Number of days required to use commodity c to build for mode m to arrive at node i
    model.d_bar = Param(model.V, model.C, model.M, within=NonNegativeReals,
                        doc="Days required for mode m to arrive at node i using commodity c")

    # Lower bound on commodity c consumption before transport mode m arrives
    model.l_bar = Param(model.C, model.M, within=NonNegativeReals,
                        doc="Lower bound on the amount of commodity c consumed for mode m")

    # ***** Variables *****
    # Binary variable: Equals 1 if at least l_cmi units of commodity c
    # have been consumed at node i for mode m or not
    model.phi = Var(model.C, model.M, model.V, model.T, within=Binary,
                    doc="1 if at least `l_cmi` units of commodity `c` consumed at node `i` for mode `m` at time `t`, else 0")

    # Define variables
    ##The amount of commodity c consumed to meet demand at node i in V
    ##at the end of time t
    model.y = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # y_ict

    ##The amount of commodity c unmet demand at node i in V at the end of time t
    model.z = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # z_ict

    ##the units of flow on arc (i,j) for commodity c during time t
    model.x = Var(model.A, model.C, model.T, model.M, within=NonNegativeReals)  # x_ijctm

    model.x_bar = Var(model.A, model.T, model.M, within=Binary)  # Binary: mode m is used on arc (i,j) at time t

    ##The units of commodity c stored at node i at the end of time period t
    model.w = Var(model.V, model.C, model.T, within=NonNegativeIntegers)  # w_ict

    ##The units of transport mode m stored at node i at the end of time period t
    model.wbar = Var(model.V, model.M, model.T, within=NonNegativeIntegers)  # w_ict

    ##the units of units of commodity c pre-positioned at node i
    model.s_var = Var(model.V, model.C, within=NonNegativeIntegers)  # s_ic

    ##the units of units of commodity c pre-positioned at node i
    model.s_varbar = Var(model.V, model.M, within=NonNegativeIntegers)  # s_ic

    ##equals 1 if commodity c is pre positioned at node i
    model.p_var = Var(model.V, model.C, within=Binary)  # p_ic

    ##equals 1 if mode m is pre positioned at node i
    model.p_varbar = Var(model.V, model.M, within=Binary)  # p_ic

    ##equals 1 if node i is pre-positioned with any commodity
    model.q_var = Var(model.V, within=Binary)  # q_i

    ##equals 1 if node i is pre-positioned with any transport mode
    model.q_varbar = Var(model.V, within=Binary)  # q_i

    demand_weight = 0.5
    risk_weight = 0.5

    # Objective function

    def combined_risk_and_demand_cost(model):
        return risk_weight * (
            sum(sum(sum(sum(model.r1[i, j, c, t] * model.x[i, j, c, t, m] for (i, j) in model.A) for m in model.M) +
                    sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(
                model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in
                model.C)) + demand_weight * (
            sum(model.p[i, c, t] * model.z[i, c, t] for i in model.V for c in model.C for t in model.T))

    model.combined_risk_and_demand_objective = Objective(rule=combined_risk_and_demand_cost, sense=minimize)

    ###defines the demand constraints here
    # Constraint (1)
    
    def primary_demand_satisfaction(model, i, c, t):
        return sum(model.y[i, c, t_prime] for t_prime in model.T if t_prime <= t) + model.z[i, c, t] == model.d[i, c]

    model.primary_demand_satisfaction_constraint = Constraint(model.V, model.C, model.T,
                                                              rule=primary_demand_satisfaction)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs

    def prepositioned_inventory_balance(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]

    model.prepositioned_inventory_balance = Constraint(model.V_s, model.C, rule=prepositioned_inventory_balance)

    ## Constraint (4)
    ##These constraints ensure that the total commodity c leftover is equal to the total flow of c supplied minutes the amount consumed

    def flow_balance_supply_equals_demand(model, c):
        return sum(model.w[i, c, model.T.last()] for i in model.V) == sum(model.s_var[j, c] for j in model.V_s) - sum(
            model.y[i, c, t] for i in model.V for t in model.T)

    model.flow_balance_supply_equals_demand_constraint = Constraint(model.C, rule=flow_balance_supply_equals_demand)

    ## Constraint (5)
    #Cannot exced m number of units at node i for commodity c at time t

    def max_allocation_limit_function(model, i, c, t):
        return model.w[i, c, t] <= model.m[i, c, t]

    model.max_allocation_limit_per_node_constraint = Constraint(model.V, model.C, model.T,
                                                                rule=max_allocation_limit_function)

    ## Constraint (6)
    ##cannot exceed max capacity on arc (i,j)
    
    def arc_capacity_constraint_function(model, i, j, c, t, m):
        return model.x[i, j, c, t, m] <= model.mu[i, j, c, t, m]

    model.arc_capacity_constraint = Constraint(model.A, model.C, model.T, model.M,
                                               rule=arc_capacity_constraint_function)

    ## Constraint (7)
    ## cannot exceed the total budgeted cost out of i during t
    def total_unit_storage_and_cost_limit(model, i, t):
        return sum(model.b[c] * model.w[i, c, t] for c in model.C) + sum(model.w[i, m, t] for m in model.M) <= model.Mt[
            i, t]

    model.total_unit_storage_and_cost_limit_constraint = Constraint(model.V, model.T,
                                                                    rule=total_unit_storage_and_cost_limit)

    ## Constraint (10)
    ## Flow balance constraints

    ## Keeps up with the flow out, the flow in, the demand consumed, and the demand that is stored there between flow periods
    def commodity_flow_balance(model, i, c, t):
        return sum(sum(model.x[i, j, c, t, m] for j in model.V if (i, j) in model.A) - sum(
            model.x[j, i, c, t, m] for j in model.V if (j, i) in model.A) for m in
                   model.M) + model.w[i, c, t] - model.w[i, c, t - 1] + model.y[i, c, t] == 0

    model.commodity_flow_balance_constraint = Constraint(model.V, model.C, model.Tminus, rule=commodity_flow_balance)

    def transport_mode_flow_balance(model, i, m, t):
        return sum(sum(model.x[i, j, c, t, m] for j in model.V if (i, j) in model.A) - sum(
            model.x[j, i, c, t, m] for j in model.V if (j, i) in model.A)
                   for c in model.C) + model.wbar[i, m, t] - model.wbar[i, m, t - 1] == 0

    model.transport_mode_flow_balance_constraint = Constraint(model.V, model.M, model.Tminus,
                                                              rule=transport_mode_flow_balance)

    ## Constraint (10.5)
    ##Ensures that if we store commodity c at i then we must have opened that node i for commodity c

    def commodity_storage_capacity_limit_function(model, i, c):
        return model.s_var[i, c] <= model.MM[i, c] * model.p_var[i, c]

    model.commodity_storage_capacity_limit_constraint = Constraint(model.V, model.C,
                                                                   rule=commodity_storage_capacity_limit_function)

    ##Ensures that if we store commodity c at i then we must have opened that node i for commodity c

    def transport_mode_storage_capacity_limit_function(model, i, m):
        return model.s_varbar[i, m] <= model.MM[i, 1] * model.p_varbar[i, m]

    model.transport_mode_storage_capacity_limit_constraint = Constraint(model.V, model.M,
                                                                        rule=transport_mode_storage_capacity_limit_function)

    ## Constraint (10.6)
    ##Ensures that if we store commodity c at node i then we must have opened i

    def commodity_opening_dependency(model, i, c):
        return model.p_var[i, c] <= model.q_var[i]

    model.commodity_opening_dependency_constraint = Constraint(model.V, model.C, rule=commodity_opening_dependency)

    ##Ensures that if we store commodity c at node i then we must have opened i
    
    def transport_mode_opening_dependency(model, i, m):
        return model.p_varbar[i, m] <= model.q_varbar[i]

    model.transport_mode_opening_dependency_constraint = Constraint(model.V, model.M,
                                                                    rule=transport_mode_opening_dependency)

    ## Constraint: Maximum Pre-positioned APSs
    ## Ensures that we don't open more than P number of APSs
    def maximum_prepositioned_aps(model):
        return sum(model.q_var[i] for i in model.V) <= model.P

    model.max_prepositioned_aps_constraint = Constraint(rule=maximum_prepositioned_aps)

    ## Constraint (10.8)
    ## Makes sure that we have at least L units of commodity c among all starting APS locations
    
    def minimum_APS_requirement(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]

    model.minimum_APS_requirement_constraint = Constraint(model.C, rule=minimum_APS_requirement)

    ## Constraint: Minimum Storage Requirement
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def minimum_storage_requirement(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]

    model.minimum_storage_requirement_constraint = Constraint(model.V, model.C, rule=minimum_storage_requirement)

    def constraint_max_transportation_modes(model, m):
        return sum(model.s_varbar[i, m] for i in model.V) <= model.Q[m]

    model.constraint_max_transportation_modes = Constraint(model.M, rule=constraint_max_transportation_modes)


    # Constraint: Transportation volume on arcs
    def constraint_transportation_volume(model, i, j, t, m):
        return sum(model.b[c] * model.x[i, j, c, t, m] for c in model.C) <= model.v[m] * model.x_bar[i, j, t, m]

    model.constraint_transportation_volume = Constraint(model.A, model.T, model.M,
                                                        rule=constraint_transportation_volume)

    #### **Constraint 1: Lower Bound on Commodity Consumption**


    def precedence_consumption_rule(model, t, i, c, m):
        if c in model.Cm[m]:
            return sum(model.y[i, c, v] for v in model.T if v <= t - model.d_bar[i, c, m]) >= \
                model.l_bar[c, m] * model.phi[c, m, i, t]
        return Constraint.Skip

    model.precedence_consumption_constraint = Constraint(
        model.T, model.V, model.C, model.M, rule=precedence_consumption_rule,
        doc="Precedence consumption constraint for transport mode arrival"
    )

    def activation_threshold_rule(model, t, i, j, c, m):
        if c in model.Cm[m]:
            return model.mu[i, j, c, m] * sum(model.phi[c, m, j, v] for v in model.T if v < t) >= \
                model.x[i, j, c, m, t]
        return Constraint.Skip

    model.activation_threshold_constraint = Constraint(
        model.T, model.A, model.C, model.M, rule=activation_threshold_rule,
        doc="Activation constraint for transport mode based on phi"
    )

    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model)

    print(results)

if __name__ == "__main__":
    main()


