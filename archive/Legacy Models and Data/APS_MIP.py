from pyomo.environ import *
import random


def build_min_unmet_demand_model(num_locations, num_commodities, num_time_periods, num_APS_locations):
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
        return .5*sum(model.p[i, c, t] * model.z[i, c, t] for i in model.V for c in model.C for t in model.T) + 0.5*sum( sum( sum(model.r1[i, j, c, t] * model.x[i, j, c, t] for (i, j) in model.A) +
                       sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in model.C)
    model.obj = Objective(rule=objective_function, sense=minimize)


    ###defines the demand constraints here
    # Constraint (1)
    def constraint1(model, i, c, t):
        return sum(model.y[i, c, t_prime] for t_prime in model.T if t_prime <= t) + model.z[i, c, t] == model.d[i, c]
    model.constraint1 = Constraint(model.V, model.C, model.T, rule=constraint1)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs
    def constraint3(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]
    model.constraint3 = Constraint(model.V_s, model.C, rule=constraint3)

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
    def constraint10_8(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]
    model.constraint10_8 = Constraint(model.C, rule=constraint10_8)

    ## Constraint (10.9)
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def constraint10_9(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]
    model.constraint10_9 = Constraint(model.V, model.C, rule=constraint10_9)

    ## Constraint (10.91)
    ##Enforces that we don't take on too much risk
    # def constraint10_91(model):
    #     return sum( sum( sum(model.r1[i, j, c, t] * model.x[i, j, c, t] for (i, j) in model.A) +
    #                    sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in model.C) <= model.R
    # model.constraint10_91 = Constraint(rule=constraint10_91)



    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model)

    print(results)



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
    model.constraint1 = Constraint(model.V, model.C, model.T, rule=constraint1)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs
    def constraint3(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]
    model.constraint3 = Constraint(model.V_s, model.C, rule=constraint3)

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
    def constraint10_8(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]
    model.constraint10_8 = Constraint(model.C, rule=constraint10_8)

    ## Constraint (10.9)
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def constraint10_9(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]
    model.constraint10_9 = Constraint(model.V, model.C, rule=constraint10_9)

 
    ##Need to add a constraint to make sure that we meet demand at some point.
    


    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model)

    # print(results)


# print_variable_values_with_context(model)


def build_multiobjective_model(num_locations, num_commodities, num_time_periods, num_APS_locations):
    # Create a model
    model = ConcreteModel()
       # Define sets (example sizes for each set)
    model.V = Set(initialize=range(num_locations))  # Total Nodes
    model.C = Set(initialize=range(1, num_commodities))  # Total Commodities
    model.T = Set(initialize=range(num_time_periods))  # Total Time periods
    # print("Values in T:", list(model.T))
    model.Tminus = Set(initialize=range(1, num_time_periods))  # Time periods without 0
    model.A = Set(within=model.V * model.V,
                  initialize=[(i, j) for i in model.V for j in model.V if i != j and i < j])  # Fully connected DAG (Directed Acylic Graph
    model.V_s = Set(initialize=range(1, num_APS_locations))  # More special nodes

    ###penalty for a unit of unmet demand at node i for commodity c at the end of time period t
    model.p = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(10, 50) for i in model.V for c in model.C for t in model.T})

    ###demand for each commodity c at node i
    model.d = Param(model.V, model.C,
                    initialize={(i, c): random.randint(300, 1000) for i in model.V for c in model.C})
    print("Demand (d):", {key: value for key, value in model.d.items()})

    ###maximum allowable allocation for commodity c at node i and time t
    model.m = Param(model.V, model.C, model.T,
                    initialize={(i, c, t): random.randint(500, 1000) for i in model.V for c in model.C for t in model.T})
                    
    ###maximum capacity of arc (i,j) for commodity c during time t
    model.mu = Param(model.A, model.C, model.T,
                     initialize={(i, j, c, t): random.randint(0, 10) for (i, j) in model.A for c in model.C for t in model.T})

    ###Maximum allowable space for commodity c at node i
    model.M = Param(model.V, model.C,
                    initialize={(i, c): random.randint(2000, 3500) for i in model.V for c in model.C})
                    
    ###Maximum allowable space among all commodities during time t
    model.Mt = Param(model.V, model.T,
                    initialize={(i, t): random.randint(10, 2000) for i in model.V for t in model.T})
                    
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
                      initialize={(i, c): round(random.uniform(10, 30), 0) for i in model.V for c in model.C})

    ###Upper bound on the number of potential APS locations
    model.P = Param(initialize=3)

    # Maximum allowable risk
    model.R = Param(initialize=50000)

    ###weight of volume of a single unit of commodity c
    model.b = Param(model.C, initialize={c: round(random.uniform(1.0, 3.0), 2) for c in model.C})

    ###The minimum number of APS locations starting with commodity c
    model.L = Param(model.C, initialize={c: random.randint(1, 2) for c in model.C})

    model.n = Param(model.V, model.C, initialize={(i, c): round(random.uniform(1, 3), 0) for i in model.V for c in model.C})

    model.f = Param(model.V, model.C, initialize={(i, c): round(random.uniform(1, 3), 0) for i in model.V for c in model.C})
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

    ## The units of commodity c deficit left at location j when the prepo stock at j for c drops below the vulnerable value n
    model.v = Var(model.V, model.C, within=Binary)


    demand_weight = 0
    risk_weight = 1

    # Objective function
    def objective_function(model):
        return risk_weight*(sum(model.f[j,c]*model.v[j,c] for c in model.C for j in model.V)) + demand_weight*(sum(model.p[i, c, t] * model.z[i, c, t] for i in model.V for c in model.C for t in model.T))
    model.obj = Objective(rule=objective_function, sense=minimize)


    ###defines the demand constraints here
    # Constraint (1)
    def demand_satisfaction_constraint(model, i, c, t):
        return sum(model.y[i, c, t_prime] for t_prime in model.Tminus if t_prime <= t) + model.z[i, c, t] >= model.d[
            i, c]

    model.demand_satisfaction_constraint = Constraint(model.V, model.C, model.Tminus,
                                                      rule=demand_satisfaction_constraint)

    # Constraint (3)
    #Ensures that the flow from 0 to j in Vs for commodity c is equal to the determined supply of c at j in Vs
    def starting_inventory_constraint(model, j, c):
        return model.w[j, c, 0] == model.s_var[j, c]
    model.starting_inventory_constraint = Constraint(model.V_s, model.C, rule=starting_inventory_constraint)

 

    ## Constraint (5)
    #Cannot exceed m number of units at node i for commodity c at time t
    def storage_capacity_constraint(model, i, c, t):
        return model.w[i, c, t] <= model.m[i, c, t]
    model.storage_capacity_constraint = Constraint(model.V, model.C, model.T, rule=storage_capacity_constraint)

    ## Constraint (6)
    ##cannot exceed max capacity on arc (i,j)
    def arc_capacity_constraint(model, i, j, c, t):
        return model.x[i, j, c, t] <= model.mu[i, j, c, t]
    model.arc_capacity_constraint = Constraint(model.A, model.C, model.T, rule=arc_capacity_constraint)

    ## Constraint (7)
    ## cannot exceed the total budgeted cost out of i during t
    def storage_budget_constraint(model, i, t):
        return sum(model.b[c] * model.w[i, c, t] for c in model.C) <= model.Mt[i, t]
    model.storage_budget_constraint = Constraint(model.V, model.T, rule=storage_budget_constraint)

    ## Constraint (10)
    ## Flow balance constraints
    ## Keeps up with the flow out, the flow in, the demand consumed, and the demand that is stored there between flow periods
    def flow_balance_constraint(model, i, c, t):
        return sum(model.x[i, j, c, t] for j in model.V if (i, j) in model.A) - sum(
            model.x[j, i, c, t] for j in model.V if (j, i) in model.A) + model.w[i, c, t] - model.w[i, c, t - 1] + \
            model.y[i, c, t] == 0

    model.flow_balance_constraint = Constraint(model.V, model.C, model.Tminus, rule=flow_balance_constraint)

    ## Constraint (10.5)
    ##Ensures that if we store commodity c at i then we must have opened that node i for commodity c
    def ensure_storage_if_prepositioned(model, i, c):
        return model.s_var[i, c] <= model.M[i, c] * model.p_var[i, c]
    model.ensure_storage_if_prepositioned = Constraint(model.V, model.C, rule=ensure_storage_if_prepositioned)

    ## Constraint (10.6)
    ##Ensures that if we store commodity c at node i then we must have opened i
    def ensure_prepositioning_if_opened(model, i, c):
        return model.p_var[i, c] <= model.q_var[i]
    model.ensure_prepositioning_if_opened = Constraint(model.V, model.C, rule=ensure_prepositioning_if_opened)

    ## Constraint (10.7)
    ##Ensures that we don't open more than P number of APSs
    def limit_APS_open_locations(model):
        return sum(model.q_var[i] for i in model.V) <= model.P

    model.limit_APS_open_locations = Constraint(rule=limit_APS_open_locations)

    ## Constraint (10.8)
    ## Makes sure that we have at least L units of commodity c among all starting APS locations
    def minimum_APS_commodities_constraint(model, c):
        return sum(model.p_var[i, c] for i in model.V) >= model.L[c]
    model.minimum_APS_commodities_constraint = Constraint(model.C, rule=minimum_APS_commodities_constraint)

    ## Constraint (10.9)
    ## Ensures that if we store commodity c at node i then we must store at least some minimum number
    def ensure_minimum_storage_if_prepositioning(model, i, c):
        return model.s_var[i, c] >= model.ell[i, c] * model.p_var[i, c]

    model.ensure_minimum_storage_if_prepositioning = Constraint(model.V, model.C,
                                                                rule=ensure_minimum_storage_if_prepositioning)

    def leftover_values(model, j, c):
        return model.v[j,c] >= model.p_var[j,c]*model.n[j,c] - model.w[j,c,num_time_periods-1]
    model.leftover_values = Constraint(model.V, model.C, rule=leftover_values)

    ##Enforces that we don't take on too much risk
    def total_risk_constraint(model):
        return sum(sum(sum(model.r1[i, j, c, t] * model.x[i, j, c, t] for (i, j) in model.A) +
                       sum(model.r2[i, c, t] * model.w[i, c, t] for i in model.V) + sum(
            model.r3[i, c] * model.s_var[i, c] for i in model.V) for t in model.T) for c in model.C) <= model.R

    model.total_risk_constraint = Constraint(rule=total_risk_constraint)



    # Solve the model
    solver = SolverFactory('gurobi')  # or another solver
    results = solver.solve(model, options={'MIPGap': 0.00001})

    # print(results)
    # print("\nValues of decision variables:")
    # if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    #     print("Optimization successful! Here are the results:")
    #     print_variable_values_with_context(model)
    # else:
    #     print("Solver did not find an optimal solution or encountered an issue.")


    for c in model.C:
        for i in model.V:
            for j in model.V:
                for t in model.Tminus:
                    if (i,j) in model.A:
                        if model.x[i,j,c,t].value > 0:
                            print("commodity ",c," flow on arc (",i,",",j,") during time period ", t," is ", model.x[i,j,c,t].value)

    for i in model.V:
       for c in model.C:
           for t in model.T:
               if model.y[i,c,t].value is not None:
                   if model.y[i, c, t].value > 0:
                       print("commodity ",c," consumed by node (",i,") at the end of time period ", t," is ", model.y[i,c,t].value)
               if model.z[i,c,t].value is not None:

                   if model.z[i,c,t].value > 0:
                       print("commodity ",c," unmet demand at node (",i,") at the end of time period ", t," is ", model.z[i,c,t].value)
       for c in model.C:
           for i in model.V:
               if model.s_var[i,c].value > 0:
                       print("commodity ",c," prepositioned at (",i,") is ", model.s_var[i,c].value)



       if model.v[i,c].value is not None:

           if model.v[i,c].value > 0:
              print("commodity ",c," deficit node (",i,") at the end of all time periods is ", model.v[i,c].value)

def print_variable_values_with_context(model):
    """
    Prints all variable values from the model with problem-specific context.
    """
    for v in model.component_objects(Var, active=True):  # Loop through all variables
        var_object = getattr(model, str(v))  # Get the full variable object
        for index in var_object:  # Loop over each index in the variable
            # Context-specific output
            value = var_object[index].value
            if value is not None and abs(value) > 1e-6:  # Only print non-negligible values
                if isinstance(index, tuple):  # Variables with multiple indices
                    # Customize context based on the variable
                    if v.name == "x":  # Example: flow variables
                        i, j, c, t = index
                        print(f"Flow from {i} to {j} for commodity {c} at time {t}: {value}")
                    elif v.name == "w":  # Example: unmet demand variables
                        i, c, t = index
                        print(f"Unmet demand at {i} for commodity {c} at time {t}: {value}")

                    elif v.name == "y":  # Example: unmet demand variables
                        i, c, t = index
                        print(f"demand met at {i} for commodity {c} at time {t}: {value}")
                    elif v.name == "s_var":  # Example: surplus/carry-over variables
                        i, c = index
                        print(f"Pre-positioned stock at location {i} for commodity {c}: {value}")

                    # elif v.name == "z":  # Example: penalty variables
                    #     i, c, t = index
                    #     print(f"Penalty value for {i}, commodity {c}, at time {t}: {value}")
                    # Expand with more problem-specific logic for other variables
                else:  # Variables with a single index
                    print(f"{v.name}[{index}] = {value}")



def main():

    num_locations = 10
    num_commodities = 10
    num_time_periods = 5
    num_APS_locations = 6
    
    # build_min_unmet_demand_model(num_locations, num_commodities, num_time_periods, num_APS_locations)

    # build_risk_model(num_locations, num_commodities, num_time_periods, num_APS_locations)

    build_multiobjective_model(num_locations, num_commodities, num_time_periods, num_APS_locations)

if __name__ == "__main__":
    main()


