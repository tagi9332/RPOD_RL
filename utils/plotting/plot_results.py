import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os

def plot_trajectory_analysis(df, output_folder="results"):
    """
    Generates plots related to orbital mechanics, mission status, and RL training metrics.
    """
    if df is None or df.empty: return

    # Layout: 2 Rows, 3 Columns (Using all 6 slots now)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Mission & Trajectory Analysis (Steps: {len(df)})", fontsize=16)

    # --- 1. In-Plane Motion (Hill Frame) ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(df['hill_y'], df['hill_x'], label='Trajectory', color='blue')
    ax1.scatter(0, 0, color='red', marker='*', s=150, label='Target (RSO)', zorder=5)
    ax1.scatter(df['hill_y'].iloc[0], df['hill_x'].iloc[0], color='green', s=100, label='Start', zorder=5)
    ax1.set_xlabel("Along-Track [m] (y)")
    ax1.set_ylabel("Radial [m] (x)")
    ax1.set_title("In-Plane Motion (Hill)")
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- 2. 3D Relative Trajectory ---
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(df['hill_x'], df['hill_y'], df['hill_z'], label='Trajectory')
    ax2.scatter(0, 0, 0, color='red', marker='*')
    ax2.set_title("3D Relative Trajectory")
    ax2.set_xlabel("Radial (x)")
    ax2.set_ylabel("Along-Track (y)")
    ax2.set_zlabel("Cross-Track (z)")

    # --- 3. Approach Metrics (Range/Vel) ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(df['time_min'], df['range_mag'], label='Range [m]', color='steelblue')
    ax3.set_ylabel("Range [m]", color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['time_min'], df['vel_mag'], label='Velocity [m/s]', color='darkorange', linestyle='--')
    ax3_twin.set_ylabel("Velocity [m/s]", color='darkorange')
    ax3_twin.tick_params(axis='y', labelcolor='darkorange')
    ax3.set_xlabel("Time [min]")
    ax3.set_title("Approach Metrics")
    ax3.grid(True, alpha=0.3)

    # --- 4. Fuel Remaining ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df['time_min'], df['dV_remaining'], color='forestgreen', linewidth=2)
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Delta-V [m/s]")
    ax4.set_title("Fuel Remaining")
    ax4.grid(True)

    # --- 5. Step Reward (Instantaneous) ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(df['time_min'], df['reward'], color='purple', alpha=0.6, linewidth=1)
    ax5.fill_between(df['time_min'], df['reward'], color='purple', alpha=0.1)
    ax5.set_xlabel("Time [min]")
    ax5.set_ylabel("Reward")
    ax5.set_title("Instantaneous Step Reward")
    ax5.grid(True, alpha=0.3)

    # --- 6. Cumulative Reward (Total Return) ---
    ax6 = fig.add_subplot(2, 3, 6)
    cumulative_reward = df['reward'].cumsum()
    ax6.plot(df['time_min'], cumulative_reward, color='darkviolet', linewidth=2)
    ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("Cumulative Reward")
    ax6.set_title("Total Episode Return")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_folder, "trajectory_analysis.png")
    plt.savefig(save_path)
    print(f"Saved Trajectory Analysis to {save_path}")
    plt.close(fig)

def plot_control_analysis(df, output_folder="results"):
    """
    Generates plots related to ADCS performance: Pointing Error, Attitude, Torques, and Wheel Speeds.
    """
    if df is None or df.empty: return

    # Layout: 4 Rows, 1 Column (Vertically Aligned Time Axis)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"Attitude & Control Analysis", fontsize=16)

    # Unpack axes
    ax0, ax1, ax2, ax3 = axes

    # --- Plot 1: Pointing Error ---
    if 'pointing_error_deg' in df.columns:
        # Plot directly
        ax0.plot(df['time_min'], df['pointing_error_deg'], color='k', linewidth=2, label='Error')
        
        # Add visual guides
        ax0.axhline(5.0, color='r', linestyle='--', label='Req (5 deg)')
        ax0.fill_between(df['time_min'], 0, 5.0, color='green', alpha=0.1)
        
        ax0.set_ylabel("Error [deg]")

    # --- Plot 2: Attitude State (The "State") ---
    ax1.plot(df['time_min'], df['sigma_1'], label='$\sigma_1$', color='red')
    ax1.plot(df['time_min'], df['sigma_2'], label='$\sigma_2$', color='green')
    ax1.plot(df['time_min'], df['sigma_3'], label='$\sigma_3$', color='blue')
    ax1.set_ylabel("Inertial MRP ($\sigma$)")
    ax1.set_title("Inertial Attitude State")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='upper right')

    # --- Plot 3: Control Torques (The "Effort") ---
    if 'torque_x' in df.columns:
        ax2.plot(df['time_min'], df['torque_x'], label='$u_x$', color='red', alpha=0.8)
        ax2.plot(df['time_min'], df['torque_y'], label='$u_y$', color='green', alpha=0.8)
        ax2.plot(df['time_min'], df['torque_z'], label='$u_z$', color='blue', alpha=0.8)
    else:
        ax2.text(0.5, 0.5, "Torque Data Not Found", ha='center', transform=ax2.transAxes)
        
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_title("Reaction Wheel Motor Torques")
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='upper right')

    # --- Plot 4: Wheel Speeds (The "Limit") ---
    ws_cols = [c for c in df.columns if c.startswith('ws_')]
    if ws_cols:
        for i, col in enumerate(ws_cols):
            ax3.plot(df['time_min'], df[col], label=f'RW {i+1}')
    else:
        ax3.text(0.5, 0.5, "Wheel Speed Data Not Found", ha='center', transform=ax3.transAxes)

    ax3.set_ylabel("Speed [rad/s]")
    ax3.set_xlabel("Time [min]")
    ax3.set_title("Reaction Wheel Speeds")
    ax3.grid(True, alpha=0.5)
    if len(ws_cols) < 5: 
        ax3.legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_folder, "control_analysis.png")
    plt.savefig(save_path)
    print(f"Saved Control Analysis to {save_path}")
    plt.close(fig)