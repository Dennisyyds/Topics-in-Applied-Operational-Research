#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sample data
from SIP_model import build_sip_model
from DIP_model import build_dip_model
from Solve_DIP import solve_dip_model
from Solve_SIP import solve_sip_model
import pandas as pd
import numpy as np

buses = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
         'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
         'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30',
         'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40']
routes = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
          'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20',
          'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'R30',
          'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37']
route_durations = {
    'R1': 20, 'R2': 12, 'R3': 20, 'R4': 10, 'R5': 13, 'R6': 12, 'R7': 15, 'R8': 15, 'R9': 15, 'R10': 5,
    'R11': 8, 'R12': 5, 'R13': 15, 'R14': 20, 'R15': 8, 'R16': 10, 'R17': 8, 'R18': 20, 'R19': 6, 'R20': 8,
    'R21': 6, 'R22': 8, 'R23': 5, 'R24': 5, 'R25': 8, 'R26': 5, 'R27': 10, 'R28': 15, 'R29': 10, 'R30': 15,
    'R31': 10, 'R32': 8, 'R33': 3, 'R34': 10, 'R35': 5, 'R36': 8, 'R37': 15}
route_loads = {'R1': 90, 'R2': 90, 'R3': 50, 'R4': 50, 'R5': 50, 'R6': 100, 'R7': 100, 'R8': 100, 'R9': 60, 'R10': 70,
'R11': 70, 'R12': 60, 'R13': 90, 'R14': 80, 'R15': 60, 'R16': 60, 'R17': 60, 'R18': 90, 'R19': 60, 'R20': 100,
'R21': 100, 'R22': 80, 'R23': 80, 'R24': 60, 'R25': 60, 'R26': 80, 'R27': 80, 'R28': 80, 'R29': 80, 'R30': 90,
'R31': 60, 'R32': 50, 'R33': 50, 'R34': 80, 'R35': 80, 'R36': 100, 'R37': 40}

route_wc_loads = {
    'R1': 1, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 1, 'R8': 1, 'R9': 0, 'R10': 0,
    'R11': 1, 'R12': 0, 'R13': 0, 'R14': 1, 'R15': 1, 'R16': 0, 'R17': 0, 'R18': 1, 'R19': 0, 'R20': 1,
    'R21': 1, 'R22': 0, 'R23': 0, 'R24': 1, 'R25': 1, 'R26': 0, 'R27': 0, 'R28': 1, 'R29': 0, 'R30': 1,
    'R31': 0, 'R32': 1, 'R33': 0, 'R34': 1, 'R35': 0, 'R36': 1, 'R37': 0
}
bus_capacities = {
    'B1': 100, 'B2': 100, 'B3': 100, 'B4': 100, 'B5': 100,
    'B6': 100, 'B7': 100, 'B8': 100, 'B9': 100, 'B10': 100,
    'B11': 100, 'B12': 100, 'B13': 100, 'B14': 100, 'B15': 100,
    'B16': 100, 'B17': 100, 'B18': 100, 'B19': 100, 'B20': 100,
    'B21': 100, 'B22': 100, 'B23': 100, 'B24': 100, 'B25': 100,
    'B26': 100, 'B27': 100, 'B28': 100, 'B29': 100, 'B30': 100,
    'B31': 100, 'B32': 100, 'B33': 100, 'B34': 100, 'B35': 100,
    'B36': 100, 'B37': 100, 'B38': 100, 'B39': 100, 'B40': 100}

bus_wc_capacities = {
    'B1': 2, 'B2': 2, 'B3': 2, 'B4': 2, 'B5': 2,
    'B6': 2, 'B7': 2, 'B8': 2, 'B9': 2, 'B10': 2,
    'B11': 2, 'B12': 2, 'B13': 2, 'B14': 2, 'B15': 2,
    'B16': 2, 'B17': 2, 'B18': 2, 'B19': 2, 'B20': 2,
    'B21': 2, 'B22': 2, 'B23': 2, 'B24': 2, 'B25': 2,
    'B26': 2, 'B27': 2, 'B28': 2, 'B29': 2, 'B30': 2,
    'B31': 2, 'B32': 2, 'B33': 2, 'B34': 2, 'B35': 2,
    'B36': 2, 'B37': 2, 'B38': 2, 'B39': 2, 'B40': 2}

df_s = pd.read_csv("/Users/rongzhi/Downloads/route_schedule_clear.csv")

# 2) Convert into dictionaries
route_start_times = {}
route_end_times = {}

for _, row in df_s.iterrows():
    route = row["route"]
    route_start_times[route] = row["start_time"]
    route_end_times[route] = row["end_time"]

# Reposition times between routes (minutes)

# Load the CSV
df_1 = pd.read_csv("/Users/rongzhi/Downloads/reposition_times_mean10.csv", index_col=0)

# Convert to dictionary
reposition_times = {
    (from_route, to_route): int(df_1.loc[from_route, to_route])
    for from_route in df_1.index
    for to_route in df_1.columns
    if from_route != to_route and not pd.isna(df_1.loc[from_route, to_route])
}


# Load the CSV
df_2 = pd.read_csv("/Users/rongzhi/Downloads/reposition_miles_mean2.csv", index_col=0)

# Convert to dictionary
reposition_miles = {
    (from_route, to_route): float(df_2.loc[from_route, to_route])
    for from_route in df_2.index
    for to_route in df_2.columns
    if from_route != to_route and not pd.isna(df_2.loc[from_route, to_route])
}

# Load terminal times
terminal_times_df = pd.read_csv("/Users/rongzhi/Downloads/terminal_times_mean10.csv")
terminal_times = {
    (row['from'], row['bus'], row['to']): int(row['time'])
    for _, row in terminal_times_df.iterrows()
}

# Load terminal miles
terminal_miles_df = pd.read_csv("/Users/rongzhi/Downloads/terminal_miles_mean3.csv")
terminal_miles = {
    (row['from'], row['bus'], row['to']): float(row['miles'])
    for _, row in terminal_miles_df.iterrows()
}

# Current solution (empty in this example)
current_solution = {
    'y_i0j': {},
    'y_ijk': {},
    'y_ij0': {}
}

# For the SIP model, we need reposition time scenarios
scenarios = ['S1', 'S2', 'S3']
scenario_probs = {'S1': 0.3, 'S2': 0.4, 'S3': 0.3}

# Create sample reposition time scenarios
reposition_scenarios = {
    'S1': {k: v * 0.9 for k, v in reposition_times.items()},  # Faster than expected
    'S2': {k: v for k, v in reposition_times.items()},        # As expected
    'S3': {k: v * 1.2 for k, v in reposition_times.items()}   # Slower than expected
}

# Terminal time scenarios
terminal_scenarios = {
    'S1': {k: v * 0.9 for k, v in terminal_times.items()},
    'S2': {k: v for k, v in terminal_times.items()},
    'S3': {k: v * 1.2 for k, v in terminal_times.items()}
}

# Build and solve the D-IP model
dip_model, dip_var_mapping, dip_var_index = build_dip_model(
    buses=buses,
    routes=routes,
    route_loads=route_loads,
    route_wc_loads=route_wc_loads,
    bus_capacities=bus_capacities,
    bus_wc_capacities=bus_wc_capacities,
    route_durations=route_durations,
    route_end_times=route_end_times,
    route_start_times=route_start_times,
    reposition_miles=reposition_miles,
    reposition_times=reposition_times,
    terminal_miles=terminal_miles,
    terminal_times=terminal_times,
    current_solution=current_solution,
    c=100,
    e=1000,
    r=1,
    v=50,
    s=0,
    b=15,
    alpha=0,
    beta=0
)

dip_solution = solve_dip_model(dip_model, dip_var_mapping)
print("D-IP Solution:", dip_solution)

# Build and solve the S-IP model
sip_model, sip_var_mapping, sip_var_index = build_sip_model(
    buses=buses,
    routes=routes,
    route_loads=route_loads,
    route_wc_loads=route_wc_loads,
    bus_capacities=bus_capacities,
    bus_wc_capacities=bus_wc_capacities,
    route_durations=route_durations,
    route_end_times=route_end_times,
    route_start_times=route_start_times,  # Added start times
    reposition_scenarios=reposition_scenarios,
    scenario_probs=scenario_probs,
    terminal_scenarios=terminal_scenarios,
    reposition_miles=reposition_miles,
    terminal_miles=terminal_miles,
    current_solution=current_solution,
    c=100,  # Consider reducing this to make using multiple buses more attractive
    e=1000,
    r=1,
    v=50,
    ell=100
)

sip_solution = solve_sip_model(sip_model, sip_var_mapping, scenarios, routes, scenario_probs)
print("S-IP Solution:", sip_solution)


