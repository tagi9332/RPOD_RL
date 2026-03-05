import os
import matplotlib.pyplot as plt

def plot_pareto_front(all_runs_data, summary_df, output_folder):
    """
    Plots a Pareto front of Total Simulation Time vs. Delta V Used
    for all successful Monte Carlo runs.
    """
    # 1. Filter for successful runs only
    success_df = summary_df[summary_df["success"] == True].copy()
    
    if success_df.empty:
        print("No successful runs available to generate a Pareto front.")
        return

    # 2. Extract Delta V used for each successful run
    times = []
    dvs_used = []
    run_ids = []

    for _, row in success_df.iterrows():
        run_id = int(row["run_id"])
        run_df = all_runs_data[run_id - 1] # Adjust for 0-indexing
        
        # Safely get initial and final fuel, assuming 50.0 is your max
        initial_dv = run_df.iloc[0].get("dV_remaining", 50.0)
        final_dv = run_df.iloc[-1].get("dV_remaining", 0.0)
        dv_used = initial_dv - final_dv
        
        times.append(row["total_sim_time"])
        dvs_used.append(dv_used)
        run_ids.append(run_id)

    # 3. Add dV data to our working dataframe and sort by Time (X-axis)
    success_df["dv_used"] = dvs_used
    success_df = success_df.sort_values(by="total_sim_time")

    # 4. Calculate the Pareto Front (Non-dominated sorting)
    pareto_x = []
    pareto_y = []
    min_dv = float('inf')

    for _, row in success_df.iterrows():
        # A point is on the Pareto front if its dV is strictly lower than 
        # the lowest dV seen so far (since we are already sorted by ascending time)
        if row["dv_used"] < min_dv:
            pareto_x.append(row["total_sim_time"])
            pareto_y.append(row["dv_used"])
            min_dv = row["dv_used"]

    # 5. Build the Plot
    plt.figure(figsize=(10, 6))
    
    # Scatter all successful runs
    plt.scatter(
        success_df["total_sim_time"], 
        success_df["dv_used"], 
        color='lightgrey', 
        label='Successful Runs', 
        edgecolors='darkgrey',
        alpha=0.8,
        s=60,
        zorder=2
    )
    
    # Plot the Pareto frontier line and points
    plt.plot(
        pareto_x, 
        pareto_y, 
        color='red', 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        markersize=8, 
        label='Pareto Front (Optimal)',
        zorder=3
    )

    # Annotate points with Run IDs so you know which trajectories to investigate
    for _, row in success_df.iterrows():
        is_pareto = (row["total_sim_time"] in pareto_x) and (row["dv_used"] in pareto_y)
        color = 'darkred' if is_pareto else 'dimgrey'
        fontweight = 'bold' if is_pareto else 'normal'
        
        plt.annotate(
            f"Run {int(row['run_id'])}", 
            (row["total_sim_time"], row["dv_used"]),
            textcoords="offset points", 
            xytext=(6, 4), 
            ha='left',
            fontsize=9,
            color=color,
            weight=fontweight
        )

    # 6. Formatting
    plt.title("Pareto Front: Docking Time vs. Fuel Consumption", fontsize=14, fontweight='bold')
    plt.xlabel("Total Simulation Time (s)", fontsize=12)
    plt.ylabel("Delta V Used (m/s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
    plt.legend()
    plt.tight_layout()

    # 7. Save out
    save_path = os.path.join(output_folder, "pareto_front_time_vs_dv.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Pareto front plot saved to: {save_path}")