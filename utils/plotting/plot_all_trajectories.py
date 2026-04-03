import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings


def plot_all_trajectories(all_runs_data, summary_df, output_folder):
    print("Generating 3D Multi-Trajectory Plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='black', marker='*', s=300, label='Target (RSO)')

    # Added start_plotted flag here
    success_plotted, fail_plotted, start_plotted = False, False, False

    for idx, run_df in enumerate(all_runs_data):
        is_success = summary_df.loc[idx, "success"]
        color = 'mediumseagreen' if is_success else 'crimson'
        alpha = 0.8 if is_success else 0.3
        
        # Trajectory Labels
        traj_label = None
        if is_success and not success_plotted:
            traj_label = "Successful Approach"
            success_plotted = True
        elif not is_success and not fail_plotted:
            traj_label = "Failed/Timeout"
            fail_plotted = True

        # Plot the trajectory line
        ax.plot(run_df["hill_x"], run_df["hill_y"], run_df["hill_z"], 
                color=color, alpha=alpha, linewidth=1.5, label=traj_label)

        # --- Legend logic for Initial Position ---
        start_label = "Initial Position" if not start_plotted else None
        
        ax.scatter(run_df["hill_x"].iloc[0], run_df["hill_y"].iloc[0], run_df["hill_z"].iloc[0], 
                   color='blue', s=15, alpha=0.5, label=start_label)
        
        start_plotted = True 
        # -----------------------------------------

    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.set_zlabel('Relative Z (m)')
    ax.set_title(f'Monte Carlo Trajectories ({len(all_runs_data)} Runs)')
    ax.legend()
    
    plot_path = os.path.join(output_folder, 'mc_all_trajectories.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_last_100m_views(all_runs_data, summary_df, output_folder):
    print("Generating 2D Last 100m Ingress Views...")
    
    # Set up a 1x3 grid for the three planar projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_xy, ax_yz, ax_xz = axes
    
    # Plot the Target (RSO) and a 100m boundary circle in all views
    for ax in axes:
        ax.scatter(0, 0, color='black', marker='*', s=200, label='Target (RSO)', zorder=5)
        # Optional: Add a faint dashed circle to visualize the 100m sphere boundary
        circle = plt.Circle((0, 0), 100, fill=False, linestyle='--', color='gray', alpha=0.5)
        ax.add_patch(circle)

    success_plotted, fail_plotted = False, False

    for idx, run_df in enumerate(all_runs_data):
        is_success = summary_df.loc[idx, "success"]
        color = 'mediumseagreen' if is_success else 'crimson'
        alpha = 0.8 if is_success else 0.3
        
        # Calculate the 3D distance from the origin at each time step
        distances = np.sqrt(run_df["hill_x"]**2 + run_df["hill_y"]**2 + run_df["hill_z"]**2)
        
        # Filter the dataframe to only keep the last 100m
        mask = distances <= 100.0
        filtered_df = run_df[mask]
        
        # Skip this trajectory if it never got within 100m
        if filtered_df.empty:
            continue

        # Setup Labels for the legend (only add them on the first occurrence)
        traj_label = None
        if is_success and not success_plotted:
            traj_label = "Successful Approach"
            success_plotted = True
        elif not is_success and not fail_plotted:
            traj_label = "Failed/Timeout"
            fail_plotted = True

        # View 1: X-Y (Radial vs In-Track)
        ax_xy.plot(filtered_df["hill_x"], filtered_df["hill_y"], 
                   color=color, alpha=alpha, linewidth=1.5, label=traj_label)
        
        # View 2: Y-Z (In-Track vs Cross-Track)
        ax_yz.plot(filtered_df["hill_y"], filtered_df["hill_z"], 
                   color=color, alpha=alpha, linewidth=1.5)
        
        # View 3: X-Z (Radial vs Cross-Track)
        ax_xz.plot(filtered_df["hill_x"], filtered_df["hill_z"], 
                   color=color, alpha=alpha, linewidth=1.5)

    # --- Formatting View 1 (X-Y) ---
    ax_xy.set_xlabel('Relative X / Radial (m)')
    ax_xy.set_ylabel('Relative Y / In-Track (m)')
    ax_xy.set_title('R-I Plane (X-Y)')
    ax_xy.legend(loc='upper right')

    # --- Formatting View 2 (Y-Z) ---
    ax_yz.set_xlabel('Relative Y / In-Track (m)')
    ax_yz.set_ylabel('Relative Z / Cross-Track (m)')
    ax_yz.set_title('I-C Plane (Y-Z)')

    # --- Formatting View 3 (X-Z) ---
    ax_xz.set_xlabel('Relative X / Radial (m)')
    ax_xz.set_ylabel('Relative Z / Cross-Track (m)')
    ax_xz.set_title('R-C Plane (X-Z)')

    # --- Apply uniform settings to all subplots ---
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal')  # Ensure geometry is not distorted
        ax.set_xlim(-105, 105)  # Lock the view to just outside the 100m range
        ax.set_ylim(-105, 105)

    plt.suptitle(f'Last 100m Ingress - Orthographic Views', fontsize=16, y=1.02)
    
    # Save the plot
    plot_path = os.path.join(output_folder, 'mc_last_100m_views.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()