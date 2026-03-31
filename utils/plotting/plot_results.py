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

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_control_analysis(df, output_folder="results"):
    """
    Generates plots related to ADCS performance: Pointing Error, Attitude, Torques, and Wheel Speeds.
    """
    if df is None or df.empty: return

    # Layout: 4 Rows, 1 Column (Vertically Aligned Time Axis)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"Attitude & Control Analysis", fontsize=16)

    ax0, ax1, ax2, ax3 = axes

    # --- Plot 1: Pointing Error & Components ---
    # Auto-convert radians to degrees if needed
    if 'pointing_error' in df.columns and 'pointing_error_deg' not in df.columns:
        df['pointing_error_deg'] = np.degrees(df['pointing_error'])

    lines_ax0, labels_ax0 = [], []
    
    if 'pointing_error_deg' in df.columns:
        l1, = ax0.plot(df['time_min'], df['pointing_error_deg'], color='k', linewidth=2, label='Total Error [deg]')
        ax0.set_ylabel("Total Error [deg]", color='k')
        ax0.tick_params(axis='y', labelcolor='k')
        lines_ax0.append(l1)
        labels_ax0.append('Total Error [deg]')

    # Check for tracking error MRP components (sigma_BR)
    err_x = next((c for c in df.columns if c in ['sigma_BR_x', 'att_err_x']), None)
    err_y = next((c for c in df.columns if c in ['sigma_BR_y', 'att_err_y']), None)
    err_z = next((c for c in df.columns if c in ['sigma_BR_z', 'att_err_z']), None)

    if err_x and err_y and err_z:
        ax0_twin = ax0.twinx()
        l2, = ax0_twin.plot(df['time_min'], df[err_x], label='$\sigma_{BR,1}$', color='red', alpha=0.8)
        l3, = ax0_twin.plot(df['time_min'], df[err_y], label='$\sigma_{BR,2}$', color='green', alpha=0.8)
        l4, = ax0_twin.plot(df['time_min'], df[err_z], label='$\sigma_{BR,3}$', color='blue', alpha=0.8)
        ax0_twin.set_ylabel("Error MRP ($\sigma_{BR}$)")
        
        lines_ax0.extend([l2, l3, l4])
        labels_ax0.extend(['$\sigma_{BR,1}$', '$\sigma_{BR,2}$', '$\sigma_{BR,3}$'])

    if lines_ax0:
        ax0.legend(lines_ax0, labels_ax0, loc='upper right')
    else:
        ax0.text(0.5, 0.5, "Error Data Not Found", ha='center', transform=ax0.transAxes)

    ax0.set_title("Attitude Tracking Error")
    ax0.grid(True, alpha=0.5)

    # --- Plot 2: Attitude State ---
    sig_x = 'sigma_BN_x' if 'sigma_BN_x' in df.columns else 'sigma_1'
    sig_y = 'sigma_BN_y' if 'sigma_BN_y' in df.columns else 'sigma_2'
    sig_z = 'sigma_BN_z' if 'sigma_BN_z' in df.columns else 'sigma_3'
    
    if sig_x in df.columns:
        ax1.plot(df['time_min'], df[sig_x], label='$\sigma_1$', color='red')
        ax1.plot(df['time_min'], df[sig_y], label='$\sigma_2$', color='green')
        ax1.plot(df['time_min'], df[sig_z], label='$\sigma_3$', color='blue')
    else:
        ax1.text(0.5, 0.5, "Attitude Data Not Found", ha='center', transform=ax1.transAxes)
        
    ax1.set_ylabel("Inertial MRP ($\sigma_{BN}$)")
    ax1.set_title("Inertial Attitude State")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='upper right')

    # --- Plot 3: Control Torques ---
    tq_x = 'torque_cmd_x' if 'torque_cmd_x' in df.columns else 'torque_x'
    tq_y = 'torque_cmd_y' if 'torque_cmd_y' in df.columns else 'torque_y'
    tq_z = 'torque_cmd_z' if 'torque_cmd_z' in df.columns else 'torque_z'
    
    if tq_x in df.columns:
        ax2.plot(df['time_min'], df[tq_x], label='$u_x$', color='red', alpha=0.8)
        ax2.plot(df['time_min'], df[tq_y], label='$u_y$', color='green', alpha=0.8)
        ax2.plot(df['time_min'], df[tq_z], label='$u_z$', color='blue', alpha=0.8)
    else:
        ax2.text(0.5, 0.5, "Torque Data Not Found", ha='center', transform=ax2.transAxes)
        
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_title("Reaction Wheel Motor Torques")
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='upper right')

    # --- Plot 4: Wheel Speeds ---
    ws_cols = [c for c in df.columns if c.startswith('ws_') or c.startswith('wheel_speeds_')]
    
    if ws_cols:
        for i, col in enumerate(ws_cols):
            ax3.plot(df['time_min'], df[col], label=f'RW {i+1}')
    elif 'wheel_speeds' in df.columns:
        ws_array = np.vstack(df['wheel_speeds'].values)
        for i in range(ws_array.shape[1]):
            ax3.plot(df['time_min'], ws_array[:, i], label=f'RW {i+1}')
    else:
        ax3.text(0.5, 0.5, "Wheel Speed Data Not Found", ha='center', transform=ax3.transAxes)

    ax3.set_ylabel("Speed [rad/s]")
    ax3.set_xlabel("Time [min]")
    ax3.set_title("Reaction Wheel Speeds")
    ax3.grid(True, alpha=0.5)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_folder, "control_analysis.png")
    plt.savefig(save_path)
    print(f"Saved Control Analysis to {save_path}")
    plt.close(fig)