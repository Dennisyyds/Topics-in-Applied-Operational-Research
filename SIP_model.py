#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import highspy
import pandas as pd

def build_sip_model(
    # Input data
    buses,                 # Set of buses
    routes,                # Set of routes
    route_loads,           # Load of each route
    route_wc_loads,        # Wheelchair load of each route
    bus_capacities,        # Capacity of each bus
    bus_wc_capacities,     # Wheelchair capacity of each bus
    route_durations,       # Duration of each route
    route_end_times,       # End time of each route
    route_start_times,     # Start time of each route (added for realism)
    reposition_scenarios,  # Dictionary of reposition times for each scenario
    scenario_probs,        # Probability of each scenario
    terminal_scenarios,    # Dictionary of terminal times for each scenario
    reposition_miles,      # Dictionary of reposition miles between routes
    terminal_miles,        # Dictionary of miles from terminal to route and back
    current_solution,      # Current bus-route assignments
    # Parameters
    c=100,                 # Penalty for using a bus
    e=1000,                # Penalty for not serving a route
    r=1,                   # Penalty for reposition miles
    v=50,                  # Penalty for deviating from current solution
    ell=100,               # Penalty for expected delay
    alpha=0,               # Buffer value for reposition times
    beta=0,                # Buffer fraction for reposition times
    w=0,                   # Lower bound on slack time (optional)
    w_bar=300,             # Upper bound on slack time (optional)
    p=None,                # Earliest departure time from terminals (optional)
):
    """
    Build the Stochastic Integer Programming model for bus-route assignment
    using the HiGHSpy optimizer.
    """
    # Create a new HiGHSpy model
    model = highspy.Highs()
    model.setOptionValue("log_to_console", False)
    
    # Get the set of scenarios
    scenarios = list(scenario_probs.keys())
    
    # Preprocessing: Determine feasible bus-route assignments
    F_ij = {}  # Capacity feasibility
    F_ijk = {} # Time and capacity feasibility
    
    # Check capacity feasibility
    for i in buses:
        for j in routes:
            F_ij[(i, j)] = (bus_capacities[i] >= route_loads[j] and 
                           bus_wc_capacities[i] >= route_wc_loads[j])
    
    # Check time feasibility for consecutive routes (at least one scenario)
    for i in buses:
        for j in routes:
            for k in routes:
                if j != k and F_ij[(i, j)] and F_ij[(i, k)]:
                    feasible_in_any_scenario = False
                    
                    for u in scenarios:
                        # Calculate buffer for reposition time
                        buffer_time = ((1 + beta) * reposition_scenarios[u][(j, k)]) + alpha
                        
                        # Check if there's enough time to travel from j to k in this scenario
                        # Using start times for more realism
                        if route_end_times[j] + buffer_time <= route_start_times[k]:
                            feasible_in_any_scenario = True
                            break
                    
                    F_ijk[(i, j, k)] = feasible_in_any_scenario
                else:
                    F_ijk[(i, j, k)] = False
    
    # Define decision variables
    # y_i0j = 1 if bus i serves route j first
    # y_ijk = 1 if bus i serves route k immediately after route j
    # y_ij0 = 1 if bus i serves route j last
    # x_j = 1 if route j is not served
    # T_j^u = start time of route j in scenario u
    # T'_j^u = end time of route j in scenario u
    # delta_j^u = delay at the end of route j in scenario u
    
    # Mapping from variable indices to their meaning
    var_mapping = {}
    var_index = {}
    var_count = 0
    
    # Variables y_i0j (bus i serves route j first)
    for i in buses:
        for j in routes:
            if F_ij[(i, j)]:
                var_mapping[var_count] = ('y_i0j', i, j)
                var_index[('y_i0j', i, j)] = var_count
                var_count += 1
    
    # Variables y_ijk (bus i serves route k after route j)
    for i in buses:
        for j in routes:
            for k in routes:
                if j != k and F_ijk[(i, j, k)]:
                    var_mapping[var_count] = ('y_ijk', i, j, k)
                    var_index[('y_ijk', i, j, k)] = var_count
                    var_count += 1
    
    # Variables y_ij0 (bus i serves route j last)
    for i in buses:
        for j in routes:
            if F_ij[(i, j)]:
                var_mapping[var_count] = ('y_ij0', i, j)
                var_index[('y_ij0', i, j)] = var_count
                var_count += 1
    
    # Variables x_j (route j is not served)
    for j in routes:
        var_mapping[var_count] = ('x_j', j)
        var_index[('x_j', j)] = var_count
        var_count += 1
    
    # Variables T_j^u (start time of route j in scenario u)
    for u in scenarios:
        for j in routes:
            var_mapping[var_count] = ('T', u, j)
            var_index[('T', u, j)] = var_count
            var_count += 1
    
    # Variables T'_j^u (end time of route j in scenario u)
    for u in scenarios:
        for j in routes:
            var_mapping[var_count] = ('T_prime', u, j)
            var_index[('T_prime', u, j)] = var_count
            var_count += 1
    
    # Variables delta_j^u (delay at the end of route j in scenario u)
    for u in scenarios:
        for j in routes:
            var_mapping[var_count] = ('delta', u, j)
            var_index[('delta', u, j)] = var_count
            var_count += 1
    
    # Create model structures
    inf = highspy.kHighsInf
    col_lower = np.zeros(var_count)
    col_upper = np.ones(var_count)  # Default upper bound
    
    # Set binary variables bounds
    for v in range(var_count):
        var_type = var_mapping[v][0]
        if var_type in ['T', 'T_prime', 'delta']:
            col_upper[v] = inf  # Continuous variables can be any positive value
    
    # Define objective function components
    obj_comp1 = np.zeros(var_count)  # Penalty for using a bus
    obj_comp2 = np.zeros(var_count)  # Penalty for not serving a route
    obj_comp3 = np.zeros(var_count)  # Penalty for reposition miles
    obj_comp4 = np.zeros(var_count)  # Penalty for deviating from current solution
    obj_comp5 = np.zeros(var_count)  # Penalty for expected delay
    
    # Component 1: Penalty for using a bus
    for v in range(var_count):
        if var_mapping[v][0] == 'y_i0j':
            i = var_mapping[v][1]
            obj_comp1[v] = c
    
    # Component 2: Penalty for not serving a route
    for v in range(var_count):
        if var_mapping[v][0] == 'x_j':
            obj_comp2[v] = e
    
    # Component 3: Penalty for reposition miles
    for v in range(var_count):
        if var_mapping[v][0] == 'y_i0j':
            i, j = var_mapping[v][1], var_mapping[v][2]
            obj_comp3[v] = r * terminal_miles[('terminal', i, j)]
        elif var_mapping[v][0] == 'y_ij0':
            i, j = var_mapping[v][1], var_mapping[v][2]
            obj_comp3[v] = r * terminal_miles[(j, 'terminal', i)]
        elif var_mapping[v][0] == 'y_ijk':
            _, i, j, k = var_mapping[v]
            obj_comp3[v] = r * reposition_miles[(j, k)]
    
    # Component 4: Penalty for deviating from current solution
    current_y_i0j = current_solution.get('y_i0j', {})
    current_y_ijk = current_solution.get('y_ijk', {})
    current_y_ij0 = current_solution.get('y_ij0', {})
    
    for v in range(var_count):
        if var_mapping[v][0] == 'y_i0j':
            i, j = var_mapping[v][1], var_mapping[v][2]
            if (i, j) in current_y_i0j and current_y_i0j[(i, j)] == 1:
                obj_comp4[v] = -v
        elif var_mapping[v][0] == 'y_ijk':
            i, j, k = var_mapping[v][1], var_mapping[v][2], var_mapping[v][3]
            if (i, j, k) in current_y_ijk and current_y_ijk[(i, j, k)] == 1:
                obj_comp4[v] = -v
        elif var_mapping[v][0] == 'y_ij0':
            i, j = var_mapping[v][1], var_mapping[v][2]
            if (i, j) in current_y_ij0 and current_y_ij0[(i, j)] == 1:
                obj_comp4[v] = -v
    
    # Component 5: Penalty for expected delay
    for v in range(var_count):
        if var_mapping[v][0] == 'delta':
            u, j = var_mapping[v][1], var_mapping[v][2]
            obj_comp5[v] = ell * scenario_probs[u]
    
    # Combine all objective components
    objective = obj_comp1 + obj_comp2 + obj_comp3 + obj_comp4 + obj_comp5
    
    # Add variables to the model
    model.addVars(var_count, col_lower, col_upper)
    
    # Set variable types
    for i in range(var_count):
        var_type = var_mapping[i][0]
        if var_type in ['y_i0j', 'y_ijk', 'y_ij0', 'x_j']:
            model.changeColIntegrality(i, highspy.HighsVarType.kInteger)
        else:
            model.changeColIntegrality(i, highspy.HighsVarType.kContinuous)
    
    # Set objective coefficients
    for i in range(var_count):
        model.changeColCost(i, objective[i])
    
    # Set the objective sense to minimize
    model.changeObjectiveSense(highspy.ObjSense.kMinimize)
    
    # Constraint (B.2): Each bus can serve at most one route first
    for i in buses:
        indices = []
        values = []
        
        for j in routes:
            if F_ij[(i, j)] and ('y_i0j', i, j) in var_index:
                indices.append(var_index[('y_i0j', i, j)])
                values.append(1.0)
        
        if indices:
            indices_np = np.array(indices, dtype=np.int32)
            values_np = np.array(values, dtype=np.float64)
            model.addRow(0.0, 1.0, len(indices_np), indices_np, values_np)
    
    # Constraint (B.3): Flow-in flow-out constraint
    for i in buses:
        for j in routes:
            if F_ij[(i, j)]:
                indices = []
                values = []
                
                # Flow in (y_i0j or y_ikj)
                if ('y_i0j', i, j) in var_index:
                    indices.append(var_index[('y_i0j', i, j)])
                    values.append(1.0)
                
                for k in routes:
                    if k != j and F_ijk[(i, k, j)]:
                        if ('y_ijk', i, k, j) in var_index:
                            indices.append(var_index[('y_ijk', i, k, j)])
                            values.append(1.0)
                
                # Flow out (y_ijk or y_ij0)
                if ('y_ij0', i, j) in var_index:
                    indices.append(var_index[('y_ij0', i, j)])
                    values.append(-1.0)
                
                for k in routes:
                    if k != j and F_ijk[(i, j, k)]:
                        if ('y_ijk', i, j, k) in var_index:
                            indices.append(var_index[('y_ijk', i, j, k)])
                            values.append(-1.0)
                
                if indices:
                    indices_np = np.array(indices, dtype=np.int32)
                    values_np = np.array(values, dtype=np.float64)
                    model.addRow(0.0, 0.0, len(indices_np), indices_np, values_np)
    
    # Constraint (B.4): Each route is served by exactly one bus or not served
    for j in routes:
        indices = []
        values = []
        
        # Add x_j (route not served)
        indices.append(var_index[('x_j', j)])
        values.append(1.0)
        
        # Add all y_i0j and y_ikj (route j served)
        for i in buses:
            if F_ij[(i, j)] and ('y_i0j', i, j) in var_index:
                indices.append(var_index[('y_i0j', i, j)])
                values.append(1.0)
            
            for k in routes:
                if k != j and F_ijk[(i, k, j)] and ('y_ijk', i, k, j) in var_index:
                    indices.append(var_index[('y_ijk', i, k, j)])
                    values.append(1.0)
        
        indices_np = np.array(indices, dtype=np.int32)
        values_np = np.array(values, dtype=np.float64)
        model.addRow(1.0, 1.0, len(indices_np), indices_np, values_np)
    
    # Large value for big-M constraints
    M = max(route_end_times.values()) * 2
    
    # Constraints (B.5) and (B.6): Start time tracking
    for u in scenarios:
        for j in routes:
            # Constraint (B.5): Start time if j is served first
            indices = [var_index[('T', u, j)]]
            values = [(route_end_times[j] - route_durations[j])]
            
            # Add indicators for buses serving route j first
            for i in buses:
                if F_ij[(i, j)] and ('y_i0j', i, j) in var_index:
                    indices.append(var_index[('y_i0j', i, j)])
                    values.append(-M)
            
            indices_np = np.array(indices, dtype=np.int32)
            values_np = np.array(values, dtype=np.float64)
            model.addRow(route_end_times[j] - route_durations[j] - M, inf, len(indices_np), indices_np, values_np)
            
            # For each potential previous route
            for k in routes:
                if k != j:
                    # Constraint (B.6): Start time if j is served after k
                    for i in buses:
                        if F_ijk[(i, k, j)] and ('y_ijk', i, k, j) in var_index:
                            indices = [
                                var_index[('T', u, j)],
                                var_index[('T', u, k)],
                                var_index[('y_ijk', i, k, j)]
                            ]
                            values = [
                                1.0,
                                -1.0,
                                M - reposition_scenarios[u][(k, j)]
                            ]
                            
                            indices_np = np.array(indices, dtype=np.int32)
                            values_np = np.array(values, dtype=np.float64)
                            model.addRow(-M, inf, len(indices_np), indices_np, values_np)
    
    # Constraint (B.7): End time is at least scheduled end time
    for u in scenarios:
        for j in routes:
            indices_np = np.array([var_index[('T_prime', u, j)]], dtype=np.int32)
            values_np = np.array([1.0], dtype=np.float64)
            model.addRow(route_end_times[j], inf, 1, indices_np, values_np)
    
    # Constraint (B.8): End time is at least start time plus duration
    for u in scenarios:
        for j in routes:
            indices_np = np.array([var_index[('T_prime', u, j)], var_index[('T', u, j)]], dtype=np.int32)
            values_np = np.array([1.0, -1.0], dtype=np.float64)
            model.addRow(route_durations[j], inf, 2, indices_np, values_np)
    
    # Constraint (B.9): Calculate delay
    for u in scenarios:
        for j in routes:
            indices_np = np.array([var_index[('delta', u, j)], var_index[('T_prime', u, j)]], dtype=np.int32)
            values_np = np.array([1.0, -1.0], dtype=np.float64)
            model.addRow(-route_end_times[j] + route_durations[j], inf, 2, indices_np, values_np)
    
    # Constraint (B.10): Non-negative delay
    for u in scenarios:
        for j in routes:
            indices_np = np.array([var_index[('delta', u, j)]], dtype=np.int32)
            values_np = np.array([1.0], dtype=np.float64)
            model.addRow(0.0, inf, 1, indices_np, values_np)
    
    return model, var_mapping, var_index
