from pyomo.environ import *
import random




def build_multiobjective_model(num_locations, num_commodities, num_time_periods, num_APS_locations):
    # Create a model
    model = ConcreteModel()
       # Define sets (example sizes for each set)
    model.V = Set(initialize=range(num_locations))  # Total Nodes
    model.vehicles = Set(initialize=range(1, 51))  # Set numbered 1 to 50
    model.C = Set(initialize=range(1, num_commodities))  # Total Commodities
    model.T = Set(initialize=range(num_time_periods))  # Total Time periods
    
    

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


