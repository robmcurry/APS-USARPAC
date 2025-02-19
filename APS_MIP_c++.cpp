#include <iostream>
#include <algorithm>
#include <fstream>
#include <list>
#include <ctime>
#include <chrono>
#include <time.h>
#include <queue>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <queue>
#include <tuple>
#include <ctime>
#include <time.h>
#include <map>
#include <cstdio>
#include <math.h>
#include <climits>
#include "gurobi_c++.h"
#include <array>
#include <numeric>


using namespace std;

int main() {
    try {
        // Initialize Gurobi environment
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        auto start = std::chrono::high_resolution_clock::now();
        // Problem dimensions
        int num_locations = 50;  // Example
        int num_commodities = 20;
        int num_time_periods = 60;
        int num_APS_locations = 20;

        // Sets
        vector<int> V(num_locations);
        iota(V.begin(), V.end(), 0);
        vector<int> C(num_commodities);
        iota(C.begin(), C.end(), 0);
        vector<int> T(num_time_periods);
        iota(T.begin(), T.end(), 0);
        vector<int> Tminus(num_time_periods - 1);
        iota(Tminus.begin(), Tminus.end(), 0);
        vector<int> V_s(num_APS_locations);
        iota(V_s.begin(), V_s.end(), 0);

        // Fully connected DAG (i, j) pairs
        vector<pair<int, int> > A;
        for (int i = 0; i < num_locations; i++) {
            for (int j = 0; j < num_locations; j++) {
                if (i != j) A.emplace_back(i, j);
            }
        }

    //output the contents of the set A


        // Randomized Parameters (Using Maps)
        map<array<int, 3>, double> p, m;
        std::map<std::array<int, 4>, double> mu, r1;  // Replaces tuple<int, int, int, int>
        std::map<std::array<int, 2>, double> M, r3, ell, Mt,d;  // Replaces tuple<int, int>
        std::map<std::array<int, 3>, double> r2;  // Replaces tuple<int, int, int>
        map<int, double> b, L;
        
        map<array<int, 3>, GRBVar> y, z, w;  // Replaces tuple<int, int, int>
        map<array<int, 4>, GRBVar> x;
        map<array<int, 2>, GRBVar>s_var, p_var;
        map<int, GRBVar> q_var;
        
//        cout << endl << endl << "Hello" << endl << endl;
        srand(static_cast<unsigned int>(time(0)));
        for (int i : V) {
            q_var[i] = model.addVar(0, 1, 0, GRB_BINARY);
            for (int c : C) {
                for (int t : T) {
                    array<int, 3> key = {i, c, t};
                    p[key] = rand() % 41 + 10;
                    m[key] = rand() % 1 + 5;
                    r2[key] = rand() % 5 + 1;
                    
                    array<int, 3> key3 = {i, c, t};
                    y[key3] = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
                    z[key3] = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
                    w[key3] = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
                }
                
                array<int, 2> key2 = {i, c};
                d[key2] = rand() % 71 + 30;
                M[key2] = rand() % 1 + 5;
                r3[key2] = rand() % 5 + 1;
                ell[key2] = rand() % 3 + 1;
                
                array<int, 2> key4 = {i, c};
                s_var[key4] = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
                p_var[key4] = model.addVar(0, 1, 0, GRB_BINARY);
            }
        }

        
//        cout << endl << endl << "Hello" << endl << endl;
        for (auto [i, j] : A) {
            for (int c : C) {
                for (int t : T) {
                    array<int, 4> key5 = {i,j,c,t};
                    x[key5] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
                }
            }
        }
        
//        cout << endl << endl << "Hello" << endl << endl;
        // Objective Function: Minimize Unmet Demand
        GRBLinExpr obj = 0;
        
        // Storage Limits (5)
            
        // Constraints
        // Demand Satisfaction (1)
        for (int i : V) {
            for (int c : C) {
                
                for (int t : T){
                    array<int, 3> key6 = {i, c, t};
                    obj += p[key6] * z[key6];
                    array<int, 3> key16 = {i, c, t};
                    model.addConstr(w[key16] <= m[key16]);
                }
                
                
                for (int t = 1; t < num_time_periods;t++ ) {
                    GRBLinExpr lhs = 0;
                    for (int t_prime = 1; t_prime <= t; t_prime++){
                        array<int, 3> key6 = {i, c, t_prime};
                        lhs += y[key6];
                    }
                    array<int, 3> key6 = {i, c, t};
                    lhs += z[key6];
                    array<int, 2> key4 = {i,c};
                    model.addConstr(lhs == d[key4]);
                }
                for (int t = 1; t < Tminus.size(); ++t) {
                    GRBLinExpr lhs = 0;
                    for (int j : V){
                        if (i != j){
                            array<int, 4> key6 = {i, j, c, t};
                            array<int, 4> key7 = {j, i, c, t};
                            lhs += x[key6] - x[key7];
                            
                        }
                    }
                    array<int, 3> key6 = {i, c, t};
                    array<int, 3> key7 = {i, c, t-1};
                    lhs += w[key6] - w[key7] + y[key6];
                    model.addConstr(lhs == 0);
                }
                
            }
        }
        
//        cout << endl << endl << "Hello" << endl << endl;
        model.setObjective(obj, GRB_MINIMIZE);
        
        
        // Initial Inventory (3)
        for (int j : V_s) {
            for (int c : C) {
                array<int, 3> key6 = {j, c, 0};
                array<int, 2> key4 = {j,c};
                model.addConstr(w[key6] == s_var[key4]);
            }
        }
        
        
//        cout << endl << endl << "Hello" << endl << endl;
        
        // Maximum Capacity on Arcs (6)
        for (auto [i, j] : A)
            for (int c : C)
                for (int t : T){
                    array<int, 4> key6 = {i, j, c, t};
                    model.addConstr(x[key6] <= mu[key6]);
                }
        // Budget Constraints (7)
        for (int i : V)
            for (int t : T) {
                GRBLinExpr lhs = 0;
                for (int c : C){
                    
                    array<int, 3> key6 = {i, c, t};
                    lhs += b[c] * w[key6];
                   
                }
                
                array<int, 2> key6 = {i, t};
                model.addConstr(lhs <= Mt[key6]);
            }

        // Risk Constraint (10.91)
        GRBLinExpr risk = 0;
        for (int c : C)
            for (int t : T)
                for (auto [i, j] : A){
                    
                    array<int, 4> key6 = {i, j, c, t};
                    risk += r1[key6] * x[key6];
                }
        for (int i : V)
            for (int c : C)
                for (int t : T){
//                    cout << endl << "hello" << endl;
                    array<int, 3> key6 = {i, c, t};
                    risk += r2[key6] * w[key6];
                }
        model.addConstr(risk <= 50000);
        

        
    
        for (int num_runs = 0; num_runs < 5; num_runs++) {
            
            
            // Solve Model
            model.optimize();

            // Print Results
            cout << "Objective Value: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            // Get the number of variables in the model
            // Example: Loop through 'x', 'y', 'z', and 'w' variables, checking for positive values
            for (int i = 0; i < num_locations; ++i) {
                for (int c = 0; c < num_commodities; ++c) {

                    array<int, 2> key6 = {i, c};


                    // Check 'y' variable (amount of commodity c consumed at node i during time t)
                    std::string sVarName = "y_" + std::to_string(i) + "_" + std::to_string(c);
                    double sVarValue = s_var[key6].get(GRB_DoubleAttr_X);  // Get the value of y_ict
    //                if (sVarValue > 0) {
    //                    std::cout << "Variable: " << sVarName << ", Value: " << sVarValue << std::endl;
    //                }
    //
    //
                    for (int t = 0; t < num_time_periods; ++t) {

                        array<int, 3> key6 = {i, c, t};


                        // Check 'y' variable (amount of commodity c consumed at node i during time t)
                        std::string yName = "y_" + std::to_string(i) + "_" + std::to_string(c) + "_" + std::to_string(t);
                        double yValue = y[key6].get(GRB_DoubleAttr_X);  // Get the value of y_ict
    //                    if (yValue > 0) {
    //                        std::cout << "Variable: " << yName << ", Value: " << yValue << std::endl;
    //                    }

                        // Check 'z' variable (amount of unmet demand at node i for commodity c at time t)
                        std::string zName = "z_" + std::to_string(i) + "_" + std::to_string(c) + "_" + std::to_string(t);
                        double zValue = z[key6].get(GRB_DoubleAttr_X);  // Get the value of z_ict
    //                    if (zValue > 0) {
    //                        std::cout << "Variable: " << zName << ", Value: " << zValue << std::endl;
    //                    }

                        // Check 'w' variable (units of commodity c stored at node i at time t)
                        std::string wName = "w_" + std::to_string(i) + "_" + std::to_string(c) + "_" + std::to_string(t);
                        double wValue = w[key6].get(GRB_DoubleAttr_X);  // Get the value of w_ict
    //                    if (wValue > 0) {
    //                        std::cout << "Variable: " << wName << ", Value: " << wValue << std::endl;
    //                    }
                    }
                }
            }

            
            
            
            model.reset();
            srand(static_cast<unsigned int>(time(0)));
            
            
            for (int i : V) {
                for (int c : C) {
                    for (int t : T) {
                        
                        array<int, 3> key = {i, c, t};
                        p[key] = rand() % 41 + 10;
                        m[key] = rand() % 1 + 5;
                        r2[key] = rand() % 5 + 1;
                        
                    }
                    
                    array<int, 2> key2 = {i, c};
                    d[key2] = rand() % 71 + 30;
                    M[key2] = rand() % 1 + 5;
                    r3[key2] = rand() % 5 + 1;
                    ell[key2] = rand() % 3 + 1;
                    
                }
            }
            
        }
        
        
        
        auto end = std::chrono::high_resolution_clock::now();
           std::chrono::duration<double> elapsed = end - start;

        
           std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    } catch (GRBException e) {
        cout << "Gurobi Error: " << e.getMessage() << endl;
    }

    return 0;
}
