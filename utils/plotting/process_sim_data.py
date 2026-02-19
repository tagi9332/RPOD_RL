import numpy as np


def process_sim_data(df):
    """
    Post-processes the raw simulation log dataframe.
    Unpacks vectors, converts units, and calculates magnitudes.
    """
    if df is None or df.empty:
        return df

    # --- 1. General Conversions ---
    df['time_min'] = df['sim_time'] / 60.0
    
    # --- 2. Unpack Hill Frame State (Position & Velocity) ---
    # Assumes r_DC_Hc and v_DC_Hc are stored as lists/arrays in the cell
    df['hill_x'] = df['r_DC_Hc'].apply(lambda v: v[0])
    df['hill_y'] = df['r_DC_Hc'].apply(lambda v: v[1])
    df['hill_z'] = df['r_DC_Hc'].apply(lambda v: v[2])
    
    df['range_mag'] = df['r_DC_Hc'].apply(np.linalg.norm)
    df['vel_mag'] = df['v_DC_Hc'].apply(np.linalg.norm)

    # --- 3. Unpack Attitude (Sigma/MRPs) ---
    df['sigma_1'] = df['sigma_BN'].apply(lambda v: v[0])
    df['sigma_2'] = df['sigma_BN'].apply(lambda v: v[1])
    df['sigma_3'] = df['sigma_BN'].apply(lambda v: v[2])

    # --- 3.5. Unpack Pointing Error ---
    if 'pointing_error' in df.columns:
        df['pointing_error_deg'] = np.degrees(df['pointing_error'])

    # --- 4. Unpack Torques ---
    # Stack the list of arrays into a matrix for easy column assignment
    try:
        # Check if the first element is valid to determine size
        sample_torque = df['torque_cmd'].iloc[0]
        if len(sample_torque) == 3:
            torque_matrix = np.vstack(df['torque_cmd'].values)
            df['torque_x'] = torque_matrix[:, 0]
            df['torque_y'] = torque_matrix[:, 1]
            df['torque_z'] = torque_matrix[:, 2]
    except Exception as e:
        print(f"Warning: Could not unpack Torques. {e}")

    # --- 5. Unpack Wheel Speeds ---
    try:
        sample_wheels = df['wheel_speeds'].iloc[0]
        n_wheels = len(sample_wheels)
        wheel_matrix = np.vstack(df['wheel_speeds'].values)
        
        for i in range(n_wheels):
            df[f'ws_{i}'] = wheel_matrix[:, i]
            
        # Store num_wheels in metadata if needed, or deduce it later
        df.attrs['n_wheels'] = n_wheels 
    except Exception as e:
        print(f"Warning: Could not unpack Wheel Speeds. {e}")
        df.attrs['n_wheels'] = 0

    return df