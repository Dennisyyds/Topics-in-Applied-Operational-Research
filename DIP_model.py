#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import highspy
import pandas as pd

def build_dip_model(
    buses, routes, route_loads, route_wc_loads, bus_capacities, bus_wc_capacities,
    route_durations, route_end_times, route_start_times, reposition_miles, reposition_times,
    terminal_miles, terminal_times, current_solution,
    c=100, e=1000, r=1, v=50, s=0, b=15, alpha=0, beta=0, w=0, w_bar=300, p=None
):
    """
    Build the Deterministic Integer Programming model for bus-route assignment
    using the HiGHSpy optimizer.
    """
    # Create a new HiGHSpy model
    model = highspy.Highs()
    model.setOptionValue("log_to_console", False)

    F_ij = {}  # Capacity feasibility
    F_ijk = {} # Time and capacity feasibility
    
    # Check capacity feasibility
    for i in buses:
        for j in routes:
            F_ij[(i, j)] = (bus_capacities[i] >= route_loads[j] and 
                           bus_wc_capacities[i] >= route_wc_loads[j])
    
    # Check time feasibility for consecutive routes
    for i in buses:
        for j in routes:
            for k in routes:
                if j != k and F_ij[(i, j)] and F_ij[(i, k)]:
                    # Calculate buffer for reposition time
                    buffer_time = ((1 + beta) * reposition_times[(j, k)]) + alpha
                    
                    # Check if there's enough time to travel from j to k
                    F_ijk[(i, j, k)] = (route_end_times[j] + buffer_time + 
                                       route_durations[k] <= route_end_times[k])
                else:
                    F_ijk[(i, j, k)] = False
    
    # Define decision variables and mappings
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
    
    # Create variable bounds and objective
    inf = highspy.kHighsInf
    col_lower = np.zeros(var_count)
    col_upper = np.ones(var_count)
    
    # Set up objective components
    obj_comp1 = np.zeros(var_count)  # Penalty for using a bus
    obj_comp2 = np.zeros(var_count)  # Penalty for not serving a route
    obj_comp3 = np.zeros(var_count)  # Penalty for reposition miles
    obj_comp5 = np.zeros(var_count)  # Penalty for small slack time
    
    # Component 1: Penalty for using a bus (A.1)
    for v in range(var_count):
        if var_mapping[v][0] == 'y_i0j':
            i = var_mapping[v][1]
            obj_comp1[v] = c
    
    # Component 2: Penalty for not serving a route (A.2)
    for v in range(var_count):
        if var_mapping[v][0] == 'x_j':
            obj_comp2[v] = e
    
    # Component 3: Penalty for reposition miles (A.3)
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
    
    
    # Component 5: Penalty for small slack time (A.5)
    if s > 0:
        for v in range(var_count):
            if var_mapping[v][0] == 'y_ijk':
                i, j, k = var_mapping[v][1], var_mapping[v][2], var_mapping[v][3]
                slack = route_end_times[k] - route_end_times[j] - route_durations[k] - reposition_times[(j, k)]
                if slack < b:
                    obj_comp5[v] = s * (b - slack)
    
    # Combine all objective components
    objective = obj_comp1 + obj_comp2 + obj_comp3 +  obj_comp5
    
    # Add variables to the model
    model.addVars(var_count, col_lower, col_upper)
    
    # Set variable types to binary (integer)
    for i in range(var_count):
        model.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    
    # Set objective coefficients
    for i in range(var_count):
        model.changeColCost(i, objective[i])
    
    # Set objective sense to minimize
    model.changeObjectiveSense(highspy.ObjSense.kMinimize)
    
    # Constraint (A.6): Each bus can serve at most one route first
    for i in buses:
        indices = []
        values = []
        
        for j in routes:
            if F_ij[(i, j)] and ('y_i0j', i, j) in var_index:
                indices.append(var_index[('y_i0j', i, j)])
                values.append(1.0)
        
        if indices:
            # Convert to numpy arrays with the right types
            indices_np = np.array(indices, dtype=np.int32)
            values_np = np.array(values, dtype=np.float64)
            model.addRow(0.0, 1.0, len(indices_np), indices_np, values_np)
    
    # Constraint (A.7): Flow-in flow-out constraint
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
    
    # Constraint (A.8): Each route is served by exactly one bus or not served
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
    
    # Optional constraints
    # Constraint (A.9): Lower bound on departure time from terminal
    if p is not None:
        for i in buses:
            for j in routes:
                if F_ij[(i, j)] and ('y_i0j', i, j) in var_index:
                    indices = np.array([var_index[('y_i0j', i, j)]], dtype=np.int32)
                    values = np.array([p + terminal_times[('terminal', i, j)]], dtype=np.float64)
                    model.addRow(-inf, route_end_times[j] - route_durations[j], 1, indices, values)
    
    # Constraint (A.10): Lower bound on slack time
    if w > 0:
        for i in buses:
            for j in routes:
                for k in routes:
                    if j != k and F_ijk[(i, j, k)] and ('y_ijk', i, j, k) in var_index:
                        slack = route_end_times[k] - route_end_times[j] - route_durations[k] - reposition_times[(j, k)]
                        if slack < w:
                            indices = np.array([var_index[('y_ijk', i, j, k)]], dtype=np.int32)
                            values = np.array([1.0], dtype=np.float64)
                            model.addRow(0.0, 0.0, 1, indices, values)
    
    # Constraint (A.11): Upper bound on slack time
    if w_bar < float('inf'):
        for i in buses:
            for j in routes:
                for k in routes:
                    if j != k and F_ijk[(i, j, k)] and ('y_ijk', i, j, k) in var_index:
                        slack = route_end_times[k] - route_end_times[j] - route_durations[k] - reposition_times[(j, k)]
                        if slack > w_bar:
                            indices = np.array([var_index[('y_ijk', i, j, k)]], dtype=np.int32)
                            values = np.array([1.0], dtype=np.float64)
                            model.addRow(0.0, 0.0, 1, indices, values)
    
    F_ijk[(i, j, k)] = (route_end_times[j] + buffer_time <= route_start_times[k])
    
    for i in buses:
   
        for j in routes:
    
            for k in routes:
            
                if j != k and F_ijk[(i, j, k)]:
                
                    # Calculate end time of route j plus reposition time
                    end_plus_repos = route_end_times[j] + reposition_times[(j, k)]
                
                # This must be less than or equal to start time of route k
                    if end_plus_repos > route_start_times[k]:
                        indices = np.array([var_index[('y_ijk', i, j, k)]], dtype=np.int32)
                        values = np.array([1.0], dtype=np.float64)
                        model.addRow(0.0, 0.0, 1, indices, values)  # Forbid this assignment
                        
                        

    return model, var_mapping, var_index
