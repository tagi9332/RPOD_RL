import os
import matplotlib.pyplot as plt

def plot_summary_table(summary_df, output_folder):
    print("Generating Results Table Image...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    display_df = summary_df[["run_id", "total_reward", "total_sim_time", "end_status", "success"]].copy()
    display_df["total_reward"] = display_df["total_reward"].round(2)
    display_df["total_sim_time"] = display_df["total_sim_time"].astype(str) + " s"
    display_df["success"] = display_df["success"].apply(lambda x: "Pass" if x else "Fail")
    
    table = ax.table(cellText=display_df.values, colLabels=["Run ID", "Final Reward", "Total Sim Time", "End Condition", "Success"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5) 
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0')

    plt.title("Monte Carlo Results Summary", fontsize=16, fontweight='bold', pad=20)
    plot_path = os.path.join(output_folder, 'mc_results_table.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
