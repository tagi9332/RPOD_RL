import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_mc_distributions(all_runs_data, summary_df, output_folder):
    print("Generating Scatter + Histogram Distributions...")
    
    # Calculate derived metrics for each run
    dv_usages = []
    final_vels = []
    
    for run_df in all_runs_data:
        # 1. dV Usage
        if "dV_remaining" in run_df.columns:
            # Assuming dV_remaining decreases over time
            dv_used = run_df["dV_remaining"].iloc[0] - run_df["dV_remaining"].iloc[-1]
        else:
            dv_used = 0.0
        dv_usages.append(dv_used)
        
        # 2. Final relative velocity magnitude
        # Check if v_DC_Hc exists (typical BSK-RL key), else estimate from position delta
        if "v_DC_Hc_x" in run_df.columns:
            vx = run_df["v_DC_Hc_x"].iloc[-1]
            vy = run_df["v_DC_Hc_y"].iloc[-1]
            vz = run_df["v_DC_Hc_z"].iloc[-1]
            vf = np.linalg.norm([vx, vy, vz])
        else:
            # Fallback backward difference estimate
            if len(run_df) > 1:
                # Extract sim rate from elapsed time / steps
                dt = (run_df["sim_time"].iloc[-1] - run_df["sim_time"].iloc[-2]) if "sim_time" in run_df.columns else 1.0
                dx = run_df["hill_x"].iloc[-1] - run_df["hill_x"].iloc[-2]
                dy = run_df["hill_y"].iloc[-1] - run_df["hill_y"].iloc[-2]
                dz = run_df["hill_z"].iloc[-1] - run_df["hill_z"].iloc[-2]
                vf = np.linalg.norm([dx, dy, dz]) / dt if dt > 0 else 0.0
            else:
                vf = 0.0
        final_vels.append(vf)
        
    # Create a local dataframe for plotting to map colors nicely
    plot_df = summary_df.copy()
    plot_df["dv_usage"] = dv_usages
    plot_df["final_vel"] = final_vels
    
    # Define the 4 metrics we are plotting
    metrics = [
        ("dv_usage", "Total dV Usage (m/s)"),
        ("total_sim_time", "Simulation Duration (s)"),
        ("final_vel", "Final Rel. Velocity (m/s)"),
        ("total_reward", "Total Reward")
    ]
    
    fig = plt.figure(figsize=(10, 14))
    # 4 rows, 2 columns (scatter, histogram)
    gs = gridspec.GridSpec(4, 2, width_ratios=(4, 1), wspace=0.05, hspace=0.3)
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'] # Seaborn muted colors
    
    for i, (col, ylabel) in enumerate(metrics):
            ax_scatter = fig.add_subplot(gs[i, 0])
            ax_hist = fig.add_subplot(gs[i, 1], sharey=ax_scatter)
            
            x = plot_df["run_id"]
            y = plot_df[col]

            # --- Calculate Statistics ---
            mean_val = y.mean()
            std_val = y.std()
            upper_bound = mean_val + 3 * std_val
            lower_bound = mean_val - 3 * std_val
            
            # Map scatter colors based on pass/fail
            scatter_colors = plot_df["success"].apply(lambda s: colors[i] if s else "gray")
            
            # Plot Scatter
            ax_scatter.scatter(x, y, color=scatter_colors, alpha=0.8, edgecolor='k', s=60, zorder=3)
            
            # Plot Histogram
            ax_hist.hist(y, bins=15, orientation='horizontal', color=colors[i], alpha=0.7, edgecolor='k', zorder=3)

            # --- Add Mean and 3-Sigma Lines ---
            for ax in [ax_scatter, ax_hist]:
                # Mean line
                ax.axhline(mean_val, color='black', linestyle='-', linewidth=1.5, label='Mean', zorder=4)
                # 3rd Sigma bounds
                ax.axhline(upper_bound, color='red', linestyle=':', linewidth=1.2, label='3σ', zorder=4)
                ax.axhline(lower_bound, color='red', linestyle=':', linewidth=1.2, zorder=4)

            # Formatting
            ax_scatter.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            if i == 0: # Add legend to the top plot only to keep it clean
                ax_scatter.legend(loc='upper right', fontsize='small', frameon=True).set_zorder(5)
                
            if i == 3:
                ax_scatter.set_xlabel("Monte Carlo Run ID", fontsize=11)
                ax_hist.set_xlabel("Count", fontsize=11)
                
            ax_scatter.grid(True, linestyle='--', alpha=0.6)
            ax_hist.grid(True, linestyle='--', alpha=0.6)
            ax_hist.tick_params(labelleft=False)
        
    fig.suptitle("Monte Carlo Run Distributions", fontsize=16, y=0.93, fontweight='bold')
    
    plot_path = os.path.join(output_folder, 'mc_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()