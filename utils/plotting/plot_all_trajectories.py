import os
import matplotlib.pyplot as plt


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