#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import highspy
import pandas as pd

def solve_sip_model(model, var_mapping, scenarios, routes, scenario_probs):
    """
    Solve the S-IP model and interpret the results.
    """
    # Run the solver
    status = model.run()
    
    # Check if the solution is optimal
    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        print(f"Model did not solve to optimality. Status: {model.modelStatus()}")
        return None
    
    # Get the solution values
    solution_values = model.getSolution().col_value
    
    # Interpret the solution
    bus_route_assignments = []
    unserved_routes = []
    delays = {}
    
    for v in range(len(solution_values)):
        if var_mapping[v][0] in ['y_i0j', 'y_ijk', 'y_ij0', 'x_j'] and solution_values[v] > 0.5:
            var_type = var_mapping[v][0]
            
            if var_type == 'y_i0j':
                i, j = var_mapping[v][1], var_mapping[v][2]
                bus_route_assignments.append((i, 'first', j))
            elif var_type == 'y_ijk':
                i, j, k = var_mapping[v][1], var_mapping[v][2], var_mapping[v][3]
                bus_route_assignments.append((i, 'after ' + str(j), k))
            elif var_type == 'y_ij0':
                i, j = var_mapping[v][1], var_mapping[v][2]
                bus_route_assignments.append((i, 'last', j))
            elif var_type == 'x_j':
                j = var_mapping[v][1]
                unserved_routes.append(j)
        
        if var_mapping[v][0] == 'delta':
            u, j = var_mapping[v][1], var_mapping[v][2]
            if u not in delays:
                delays[u] = {}
            delays[u][j] = solution_values[v]
    
    # Organize assignments by bus
    bus_assignments = {}
    for i, pos, j in bus_route_assignments:
        if i not in bus_assignments:
            bus_assignments[i] = []
        bus_assignments[i].append((pos, j))
    
    # Sort bus routes
    for i in bus_assignments:
        # Find the first route
        first_route = None
        for pos, j in bus_assignments[i]:
            if pos == 'first':
                first_route = j
                break
        
        if first_route is None:
            continue
        
        # Build the ordered route sequence
        ordered_routes = [first_route]
        current_route = first_route
        
        while True:
            next_route = None
            for pos, j in bus_assignments[i]:
                if pos == 'after ' + str(current_route):
                    next_route = j
                    break
            
            if next_route is None:
                break
                
            ordered_routes.append(next_route)
            current_route = next_route
        
        bus_assignments[i] = ordered_routes
    
    # Calculate expected delays by route
    expected_delays = {j: 0 for j in routes}
    for u in delays:
        for j in delays[u]:
            expected_delays[j] += scenario_probs[u] * delays[u][j]
    
    return {
        'bus_assignments': bus_assignments,
        'unserved_routes': unserved_routes,
        'expected_delays': expected_delays,
        'delays_by_scenario': delays,
        'objective_value': model.getObjectiveValue()
    }

