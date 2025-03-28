#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import highspy
import pandas as pd

def solve_dip_model(model, var_mapping):
    """
    Solve the D-IP model and interpret the results.
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
    total_reposition_miles = 0
    
    for v in range(len(solution_values)):
        if solution_values[v] > 0.5:  # Binary variable is 1
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
    
    return {
        'bus_assignments': bus_assignments,
        'unserved_routes': unserved_routes,
        'objective_value': model.getObjectiveValue()
    }


