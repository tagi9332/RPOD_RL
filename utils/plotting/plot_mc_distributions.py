import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_mc_distributions(all_runs_data, summary_df, output_folder):
    print("Generating Scatter + Histogram Distributions...")

    dv_usages = []
    final_vels = []

    for run_df in all_runs_data:
        if "dV_remaining" in run_df.columns:
            dv_used = run_df["dV_remaining"].iloc[0] - run_df["dV_remaining"].iloc[-1]
        else:
            dv_used = 0.0
        dv_usages.append(dv_used)

        if "v_DC_Hc_x" in run_df.columns:
            vx = run_df["v_DC_Hc_x"].iloc[-1]
            vy = run_df["v_DC_Hc_y"].iloc[-1]
            vz = run_df["v_DC_Hc_z"].iloc[-1]
            vf = np.linalg.norm([vx, vy, vz])
        else:
            if len(run_df) > 1:
                dt = (run_df["sim_time"].iloc[-1] - run_df["sim_time"].iloc[-2]) if "sim_time" in run_df.columns else 1.0
                dx = run_df["hill_x"].iloc[-1] - run_df["hill_x"].iloc[-2]
                dy = run_df["hill_y"].iloc[-1] - run_df["hill_y"].iloc[-2]
                dz = run_df["hill_z"].iloc[-1] - run_df["hill_z"].iloc[-2]
                vf = np.linalg.norm([dx, dy, dz]) / dt if dt > 0 else 0.0
            else:
                vf = 0.0
        final_vels.append(vf)

    plot_df = summary_df.copy()
    plot_df["dv_usage"] = dv_usages
    plot_df["final_vel"] = final_vels

    metrics = [
        ("dv_usage",       "Total dV Usage (m/s)",        "dv_usage"),
        ("total_sim_time", "Simulation Duration (s)",     "sim_duration"),
        ("final_vel",      "Final Rel. Velocity (m/s)",   "final_vel"),
        ("total_reward",   "Total Reward",                "total_reward"),
    ]

    colors = ['#D95319', '#0072BD', '#77AC30', '#A2142F']

    for i, (col, ylabel, file_tag) in enumerate(metrics):
        color = colors[i]

        # Filter to successful runs only
        success_mask = plot_df["success"] == True
        df_success = plot_df[success_mask]

        x = df_success["run_id"].values
        y = df_success[col].values

        mean_val  = y.mean() if len(y) > 0 else 0.0
        std_val   = y.std()  if len(y) > 0 else 0.0
        min_val   = y.min()  if len(y) > 0 else 0.0
        max_val   = y.max()  if len(y) > 0 else 0.0

        fig = plt.figure(figsize=(11, 4.5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=(3, 1), wspace=0.08)

        ax_scatter = fig.add_subplot(gs[0])
        ax_hist    = fig.add_subplot(gs[1], sharey=ax_scatter)

        # Scatter
        ax_scatter.scatter(x, y, color=color, alpha=0.7, edgecolor='none', s=18, zorder=3)
        ax_scatter.axhline(mean_val,           color='black',  linestyle='-',  linewidth=1.5, label='Mean', zorder=4)
        ax_scatter.axhline(mean_val + std_val, color='#FFD700', linestyle='--', linewidth=1.2, label='±1σ',  zorder=4)
        ax_scatter.axhline(mean_val - std_val, color='#FFD700', linestyle='--', linewidth=1.2,               zorder=4)

        # Stats annotation box
        stats_text = (
            f"n = {len(y)}\n"
            f"mean = {mean_val:.3g}\n"
            f"std  = {std_val:.3g}\n"
            f"min  = {min_val:.3g}\n"
            f"max  = {max_val:.3g}"
        )
        ax_scatter.text(
            0.02, 0.97, stats_text,
            transform=ax_scatter.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85)
        )

        ax_scatter.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
                          fontsize='medium', frameon=True, ncol=2)
        ax_scatter.set_xlabel("Monte Carlo Run ID", fontsize=13)
        ax_scatter.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax_scatter.grid(True, linestyle='--', alpha=0.6)

        # Horizontal histogram
        ax_hist.hist(y, bins=15, orientation='horizontal', color=color, alpha=0.7, edgecolor='k', zorder=3)
        ax_hist.axhline(mean_val,           color='black',  linestyle='-',  linewidth=1.5, zorder=4)
        ax_hist.axhline(mean_val + std_val, color='#FFD700', linestyle='--', linewidth=1.2, zorder=4)
        ax_hist.axhline(mean_val - std_val, color='#FFD700', linestyle='--', linewidth=1.2, zorder=4)
        ax_hist.set_xlabel("Count", fontsize=13)
        ax_hist.tick_params(labelleft=False)
        ax_hist.grid(True, linestyle='--', alpha=0.6)

        success_pct = 100.0 * success_mask.sum() / len(plot_df) if len(plot_df) > 0 else 0.0
        fig.suptitle(
            f"{ylabel}  —  Successful Runs ({success_mask.sum()}/{len(plot_df)})",
            fontsize=15, fontweight='bold', y=1.01
        )

        plot_path = os.path.join(output_folder, f'mc_dist_{file_tag}.png')
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")
