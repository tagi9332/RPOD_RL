import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Adjust these to test different "What-If" scenarios against your baseline CSV
ALPHA = -.10           # Log multiplier (Keep consistent with baseline if testing fuel)
FUEL_WEIGHT = 0.1      # YOUR PROPOSED TUNING (Try changing this to see effect)
DELTA_X_MAX = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0]) 

def parse_array_string(array_str):
    """Parses string '[ x y z ]' into numpy array."""
    if pd.isna(array_str): return np.zeros(3)
    clean_str = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
    values = [float(x) for x in clean_str.split() if x]
    return np.array(values)

def compute_rewards_and_plot(csv_path):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return

    # 2. Compute "Simulated" Rewards based on CONFIGURATION
    calc_fuel_rewards = []
    calc_range_rewards = []
    calc_total_rewards = []
    
    prev_fuel = None

    for index, row in df.iterrows():
        # A. Fuel Calculation
        current_fuel = row['dV_remaining']
        if prev_fuel is None:
            fuel_reward = 0.0
        else:
            delta_fuel = current_fuel - prev_fuel
            fuel_reward = FUEL_WEIGHT * delta_fuel # New Penalty
            
        prev_fuel = current_fuel
        calc_fuel_rewards.append(fuel_reward)

        # B. Range Calculation
        r_rel = parse_array_string(row['r_DC_Hc'])
        v_rel = parse_array_string(row['v_DC_Hc'])
        state_vector = np.concatenate([r_rel, v_rel])
        normalized_state = state_vector / DELTA_X_MAX
        mse = np.mean(normalized_state**2)
        
        range_reward = ALPHA * np.log(mse + 1e-8)
        calc_range_rewards.append(range_reward)
        
        # C. Total
        calc_total_rewards.append(fuel_reward + range_reward)

    # Add to DataFrame
    df['calc_fuel_rwd'] = calc_fuel_rewards
    df['calc_range_rwd'] = calc_range_rewards
    df['calc_total_rwd'] = calc_total_rewards
    
    # Compute Cumulative for both Actual (CSV) and Calculated
    # If 'cumulative_reward' exists in CSV, use it. Otherwise compute from 'reward'.
    if 'cumulative_reward' in df.columns:
        df['actual_cumulative'] = df['cumulative_reward']
    else:
        df['actual_cumulative'] = df['reward'].cumsum()
        
    df['calc_cumulative'] = df['calc_total_rwd'].cumsum()

    # --- 3. PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    
    # -- Plot 1: Cumulative Comparison (The "Bank Account") --
    # This shows if your new settings make the agent "richer" or "poorer" over time
    ax1.plot(df['sim_time'], df['actual_cumulative'], 'k-', linewidth=2, label='Baseline (Actual CSV)', alpha=0.6)
    ax1.plot(df['sim_time'], df['calc_cumulative'], 'r--', linewidth=2, label='Simulated (New Settings)')
    ax1.set_ylabel('Cumulative Score')
    ax1.set_title('1. Cumulative Reward: Baseline vs. New Tuning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Plot 2: Per-Step Comparison (The "Instant Feedback") --
    # This shows the immediate difference in reward signal at each timestep
    ax2.plot(df['sim_time'], df['reward'], 'k-', label='Baseline Total (CSV)', alpha=0.4)
    ax2.plot(df['sim_time'], df['calc_total_rwd'], 'r--', label='Simulated Total')
    ax2.set_ylabel('Reward per Step')
    ax2.set_title('2. Step-by-Step Reward Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -- Plot 3: The Breakdown (Why is it different?) --
    # This stacks your NEW fuel and range rewards to show their relative magnitude
    ax3.plot(df['sim_time'], df['calc_range_rwd'], 'g-', label='Range Component', alpha=0.8)
    ax3.plot(df['sim_time'], df['calc_fuel_rwd'], 'b-', label='Fuel Component (Penalty)', alpha=0.8)
    
    # Add a fill to make the magnitudes obvious
    ax3.fill_between(df['sim_time'], df['calc_range_rwd'], alpha=0.1, color='green')
    ax3.fill_between(df['sim_time'], df['calc_fuel_rwd'], alpha=0.1, color='blue')
    
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Component Magnitude')
    ax3.set_xlabel('Simulation Time (s)')
    ax3.set_title(f'3. Simulated Components (Fuel Weight = {FUEL_WEIGHT})')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
compute_rewards_and_plot("reference_sim_data.csv")