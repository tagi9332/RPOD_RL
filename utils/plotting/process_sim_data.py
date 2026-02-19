import numpy as np
import pandas as pd

def process_sim_data(df):
    """
    Post-processes the raw simulation log dataframe.
    Calculates magnitudes and renames pre-flattened columns for the plotting scripts.
    """
    if df is None or df.empty:
        return df

    # --- 1. General Conversions ---
    if 'sim_time' in df.columns:
        df['time_min'] = df['sim_time'] / 60.0
    
    # --- 2. Magnitudes (Position & Velocity) ---
    if 'hill_x' in df.columns:
        df['range_mag'] = np.linalg.norm(df[['hill_x', 'hill_y', 'hill_z']].values, axis=1)
    
    if 'v_DC_Hc_x' in df.columns:
        df['vel_mag'] = np.linalg.norm(df[['v_DC_Hc_x', 'v_DC_Hc_y', 'v_DC_Hc_z']].values, axis=1)

    # --- 3. Rename Flattened Columns (Attitude, Torques, Wheels) ---
    # Maps the exact names from your inference loop to the names the plotter expects
    rename_map = {
        'inspector_sigma_BN_x': 'sigma_1', 
        'inspector_sigma_BN_y': 'sigma_2', 
        'inspector_sigma_BN_z': 'sigma_3',
        'torque_cmd_x': 'torque_x', 
        'torque_cmd_y': 'torque_y', 
        'torque_cmd_z': 'torque_z',
        'wheel_speeds_x': 'ws_0',
        'wheel_speeds_y': 'ws_1',
        'wheel_speeds_z': 'ws_2'
    }
    
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    # --- 4. Pointing Error ---
    if 'pointing_error' in df.columns:
        df['pointing_error_deg'] = np.degrees(df['pointing_error'])

    return df