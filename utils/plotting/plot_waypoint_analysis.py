import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_waypoint_analysis(all_runs_data, summary_df, output_folder):
    print("Generating Waypoint Analysis Plots...")

    n_total = len(summary_df)
    n_waypoint = int(summary_df["waypoint_captured"].sum()) if "waypoint_captured" in summary_df.columns else 0
    n_success = int(summary_df["success"].sum())

    capture_dists = np.array([])
    if "waypoint_capture_dist" in summary_df.columns:
        vals = summary_df.loc[summary_df["waypoint_captured"] == True, "waypoint_capture_dist"].dropna()
        capture_dists = vals[np.isfinite(vals)].values

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.40)

    # --- Panel 1: Run Outcomes Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["Total\nRuns", "Waypoint\nCaptured", "Docking\nSuccess"]
    counts = [n_total, n_waypoint, n_success]
    colors = ['#2E86C1', '#1565A7', '#0A2342']
    bars = ax1.bar(categories, counts, color=colors, edgecolor='k', zorder=3)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / n_total
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{count} ({pct:.0f}%)", ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=15)
    ax1.set_ylim(0, n_total * 1.22)
    ax1.set_title("Run Outcomes", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')

    # --- Panel 2: Waypoint Capture Accuracy Histogram ---
    ax2 = fig.add_subplot(gs[0, 1])
    if len(capture_dists) > 0:
        ax2.hist(capture_dists, bins=min(25, len(capture_dists)),
                 color='#2E86C1', edgecolor='k', alpha=0.85, zorder=3)
        mean_dist = np.mean(capture_dists)
        ax2.axvline(mean_dist, color='black', linewidth=2, label=f'Mean: {mean_dist:.2f} m')
        ax2.set_xlabel("Distance to Waypoint Center at Capture (m)", fontsize=15)
        ax2.set_ylabel("Count", fontsize=15)
        ax2.set_title(f"Waypoint Capture Accuracy\n(n={len(capture_dists)} captures)", fontsize=16, fontweight='bold')
        ax2.legend(fontsize=13)
    else:
        ax2.text(0.5, 0.5, "No waypoint captures recorded", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=15)
        ax2.set_title("Waypoint Capture Accuracy", fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("Waypoint & Docking Analysis", fontsize=20, fontweight='bold', y=1.04)
    plot_path = os.path.join(output_folder, "waypoint_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_angular_velocity_analysis(all_runs_data, summary_df, output_folder):
    print("Generating Angular & Velocity Analysis Plots...")

    # Angular offset at docking — successful runs only
    success_mask = summary_df["success"] == True
    docking_angles = []
    for _, row in summary_df[success_mask].iterrows():
        run_df = all_runs_data[int(row["run_id"]) - 1]
        if "approach_angle_deg" in run_df.columns:
            docking_angles.append(float(run_df.iloc[-1]["approach_angle_deg"]))

    # Final step body-frame velocity decomposition — successful runs only
    # docking_port_boresight = [0, 0, 1], so axial = v_z, lateral = ||(v_x, v_y)||
    axial_vels = []
    lateral_vels = []
    for _, row in summary_df[success_mask].iterrows():
        run_df = all_runs_data[int(row["run_id"]) - 1]
        has_vz = "v_DC_C_z" in run_df.columns
        has_vxy = "v_DC_C_x" in run_df.columns and "v_DC_C_y" in run_df.columns
        if has_vz:
            axial_vels.append(float(run_df.iloc[-1]["v_DC_C_z"]) * 100.0)
        if has_vxy:
            vx = float(run_df.iloc[-1]["v_DC_C_x"])
            vy = float(run_df.iloc[-1]["v_DC_C_y"])
            lateral_vels.append(float(np.sqrt(vx**2 + vy**2)) * 100.0)

    axial_vels = np.array(axial_vels)
    lateral_vels = np.array(lateral_vels)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.40)

    # --- Panel 1: Angular Offset at Docking ---
    ax1 = fig.add_subplot(gs[0, 0])
    if docking_angles:
        ax1.hist(docking_angles, bins=min(25, len(docking_angles)),
                 color='#1565A7', edgecolor='k', alpha=0.85, zorder=3)
        mean_ang = np.mean(docking_angles)
        p95_ang = np.percentile(docking_angles, 95)
        ax1.axvline(mean_ang, color='black', linewidth=2, label=f'Mean: {mean_ang:.1f}°')
        ax1.axvline(p95_ang, color='#0A2342', linestyle=':', linewidth=1.5, label=f'95th pct: {p95_ang:.1f}°')
        ax1.set_xlabel("Approach Angle at Docking (deg)", fontsize=15)
        ax1.set_ylabel("Count", fontsize=15)
        ax1.set_title(f"Angular Offset at Docking\n(n={len(docking_angles)} successful runs)", fontsize=16, fontweight='bold')
        ax1.legend(fontsize=13, loc='upper right')
    else:
        ax1.text(0.5, 0.5, "No successful runs", ha='center', va='center',
                 transform=ax1.transAxes, fontsize=15)
        ax1.set_title("Angular Offset at Docking", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 2: Final Axial Velocity (inward = positive) ---
    axial_vels_inward = np.abs(axial_vels)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(axial_vels_inward) > 0:
        ax2.hist(axial_vels_inward, bins=min(25, len(axial_vels_inward)),
                 color='#2E86C1', edgecolor='k', alpha=0.85, zorder=3)
        mean_axial = np.mean(axial_vels_inward)
        p95_axial = np.percentile(axial_vels_inward, 95)
        ax2.axvline(mean_axial, color='black', linewidth=2, label=f'Mean: {mean_axial:.2f} cm/s')
        ax2.axvline(p95_axial, color='#0A2342', linestyle=':', linewidth=1.5, label=f'95th pct: {p95_axial:.2f} cm/s')
        ax2.set_xlabel("Axial Velocity (cm/s)", fontsize=15)
        ax2.set_ylabel("Count", fontsize=15)
        ax2.set_title(f"Final Axial Velocity (Inward)\n(n={len(axial_vels_inward)} successful runs)", fontsize=16, fontweight='bold')
        ax2.legend(fontsize=13, loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No v_DC_C data logged", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=15)
        ax2.set_title("Final Axial Velocity (Inward)", fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 3: Final Lateral Velocity (perpendicular to docking-port boresight) ---
    ax3 = fig.add_subplot(gs[0, 2])
    if len(lateral_vels) > 0:
        ax3.hist(lateral_vels, bins=min(25, len(lateral_vels)),
                 color='#0D5C8B', edgecolor='k', alpha=0.85, zorder=3)
        mean_lat = np.mean(lateral_vels)
        p95_lat = np.percentile(lateral_vels, 95)
        ax3.axvline(mean_lat, color='black', linewidth=2, label=f'Mean: {mean_lat:.2f} cm/s')
        ax3.axvline(p95_lat, color='#0A2342', linestyle=':', linewidth=1.5, label=f'95th pct: {p95_lat:.2f} cm/s')
        ax3.set_xlabel("Lateral Velocity (cm/s)", fontsize=15)
        ax3.set_ylabel("Count", fontsize=15)
        ax3.set_title(f"Final Lateral Velocity\n(perp. to docking axis, n={len(lateral_vels)} runs)", fontsize=16, fontweight='bold')
        ax3.legend(fontsize=13, loc='upper right')
    else:
        ax3.text(0.5, 0.5, "No v_DC_C data logged", ha='center', va='center',
                 transform=ax3.transAxes, fontsize=15)
        ax3.set_title("Final Lateral Velocity", fontsize=16, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("Approach Angle & Final Velocity (Monte Carlo)", fontsize=20, fontweight='bold', y=1.04)
    plot_path = os.path.join(output_folder, "angular_velocity_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
