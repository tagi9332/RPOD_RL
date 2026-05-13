import numpy as np
import pandas as pd


def interpolate_to_uniform_time(df, dt=10.0):
    """Resample a run dataframe to a uniform time grid for smooth plotting.

    Physical state columns (position, velocity, attitude, angles) are linearly
    interpolated. Per-step event columns (reward, fuel, booleans) are
    forward-filled so cumulative sums remain correct.

    Args:
        df: DataFrame with a 'sim_time' column.
        dt: Target timestep in seconds (default 10 s).
    """
    if 'sim_time' not in df.columns or len(df) < 2:
        return df

    t = df['sim_time'].values.astype(float)
    t_new = np.arange(t[0], t[-1] + dt * 0.5, dt)

    # Columns representing discrete per-step events — forward-fill, not interpolated.
    _FFILL = {'reward', 'docked_state', 'run_id', 'max_range_violation'}

    result = pd.DataFrame({'sim_time': t_new})

    for col in df.columns:
        if col == 'sim_time':
            continue

        use_ffill = (
            col in _FFILL
            or col.startswith('rew_')
            or pd.api.types.is_bool_dtype(df[col])
        )

        if not use_ffill and (
            pd.api.types.is_float_dtype(df[col])
            or pd.api.types.is_integer_dtype(df[col])
        ):
            result[col] = np.interp(t_new, t, df[col].astype(float).values)
        else:
            # Forward-fill: each interpolated point inherits the last actual value.
            idx = np.clip(np.searchsorted(t, t_new, side='right') - 1, 0, len(df) - 1)
            result[col] = df[col].values[idx]

    return result


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

    # --- 5. Resample to uniform time grid ---
    df = interpolate_to_uniform_time(df)

    return df