import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

def plot_single_run_rewards(run_df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """Plots step-by-step and cumulative rewards for a single run."""
    reward_cols = [c for c in run_df.columns if c.startswith("rew_")]
    if not reward_cols: return

    time = run_df["sim_time"]
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])
    
    # 1. Step-by-Step Rewards
    ax0 = plt.subplot(gs[0])
    ax0.set_title(f"{prefix.replace('_', ' ').title()}Reward Signal per Timestep", fontsize=14, fontweight="bold")
    for col in reward_cols:
        clean_name = col.replace("rew_", "")
        ax0.plot(time, run_df[col], alpha=0.7, label=clean_name)
    ax0.plot(time, run_df["reward"], color="black", linewidth=2, linestyle="--", label="Total Step Reward")
    ax0.set_ylabel("Step Reward")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # 2. Cumulative Return
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_title("Cumulative Return (Episode Tally)", fontsize=14, fontweight="bold")
    for col in reward_cols:
        clean_name = col.replace("rew_", "")
        cum_rew = run_df[col].cumsum()
        ax1.plot(time, cum_rew, linewidth=2, label=f"{clean_name} (Final: {cum_rew.iloc[-1]:.1f})")
    
    total_return = run_df["reward"].cumsum()
    ax1.plot(time, total_return, color="black", linewidth=3, linestyle="--", 
             label=f"Total Return (Final: {total_return.iloc[-1]:.1f})")
    
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Cumulative Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}reward_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_mc_reward_summary(all_runs_data: list, output_folder: str):
    """Generates a Bar Chart and Boxplot for cumulative rewards across all Monte Carlo runs."""
    reward_cols = [c for c in all_runs_data[0].columns if c.startswith("rew_")]
    if not reward_cols: return

    # Extract final cumulative sums for each run
    summary_data = []
    for run_id, df in enumerate(all_runs_data):
        run_totals = {"Run": run_id + 1}
        for col in reward_cols:
            run_totals[col.replace("rew_", "")] = df[col].sum()
        run_totals["Total Return"] = df["reward"].sum()
        summary_data.append(run_totals)
    
    sum_df = pd.DataFrame(summary_data)
    melted_df = sum_df.drop(columns=["Run"]).melt(var_name="Reward Component", value_name="Cumulative Total")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Bar Chart (Average Contribution)
    sns.barplot(data=melted_df, x="Cumulative Total", y="Reward Component", 
                ax=axes[0], estimator=np.mean, errorbar='sd', palette="viridis")
    axes[0].set_title("Average Reward Contribution per Component", fontweight="bold")
    axes[0].set_xlabel("Average Cumulative Reward (with Std Dev)")
    axes[0].axvline(0, color='black', linewidth=1)

    # 2. Box Plot (Distribution across MC runs)
    sns.boxplot(data=melted_df, x="Cumulative Total", y="Reward Component", 
                ax=axes[1], palette="viridis")
    sns.stripplot(data=melted_df, x="Cumulative Total", y="Reward Component", 
                  ax=axes[1], color="black", alpha=0.5, size=4)
    axes[1].set_title("Reward Distribution Across All Runs", fontweight="bold")
    axes[1].axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "mc_reward_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Monte Carlo reward summary plots to {output_folder}/mc_reward_distributions.png")