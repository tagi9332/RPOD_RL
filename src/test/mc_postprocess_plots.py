"""
Standalone post-processing script: loads pre-saved MC CSV files and regenerates all plots.

Usage:
    python -m src.test.mc_postprocess_plots
    python -m src.test.mc_postprocess_plots --data results/mc_all_runs_data.csv --summary results/mc_summary_stats.csv --output results
"""
import argparse
import os
import numpy as np
import pandas as pd

from utils.plotting import (
    plot_control_analysis,
    plot_trajectory_analysis,
    process_sim_data,
    interpolate_to_uniform_time,
    plot_interactive_trajectories,
    plot_mc_distributions,
    plot_all_trajectories,
    plot_summary_table,
    plot_pareto_front,
    plot_single_run_rewards,
    plot_last_30m_views,
    plot_waypoint_analysis,
    plot_angular_velocity_analysis,
    plot_vr_diagram,
    plot_approach_angle_history,
    plot_initial_condition_scatter,
    plot_failure_mode_breakdown,
    plot_distance_history,
    plot_dv_vs_distance,
    plot_reward_heatmap,
    plot_sun_angle_vs_distance,
    plot_impulse_histogram,
    plot_impulse_timeseries,
    plot_mean_impulse_history,
)


def load_mc_data(data_csv: str, summary_csv: str) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    all_data = pd.read_csv(data_csv)
    summary_df = pd.read_csv(summary_csv)

    run_ids = sorted(all_data["run_id"].unique())
    all_runs_data = [all_data[all_data["run_id"] == rid].reset_index(drop=True) for rid in run_ids]

    print(f"Loaded {len(all_runs_data)} runs from {data_csv}")
    print(f"Success rate: {(summary_df['success'].sum() / len(summary_df)) * 100:.1f}%")
    print(f"Mean reward: {summary_df['total_reward'].mean():.2f} +/- {summary_df['total_reward'].std():.2f}")

    return all_runs_data, summary_df


def run_postprocess(data_csv: str, summary_csv: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    raw_runs_data, summary_df = load_mc_data(data_csv, summary_csv)

    all_runs_data = [interpolate_to_uniform_time(df) for df in raw_runs_data]

    # MC-level plots
    plot_all_trajectories(all_runs_data, summary_df, output_folder)
    plot_interactive_trajectories(all_runs_data, summary_df, output_folder)
    plot_summary_table(summary_df, output_folder)
    plot_mc_distributions(all_runs_data, summary_df, output_folder)
    plot_pareto_front(all_runs_data, summary_df, output_folder)
    plot_last_30m_views(all_runs_data, summary_df, output_folder)
    plot_waypoint_analysis(all_runs_data, summary_df, output_folder)
    plot_angular_velocity_analysis(all_runs_data, summary_df, output_folder)
    plot_vr_diagram(all_runs_data, summary_df, output_folder)
    plot_approach_angle_history(all_runs_data, summary_df, output_folder)
    plot_initial_condition_scatter(all_runs_data, summary_df, output_folder)
    plot_failure_mode_breakdown(summary_df, output_folder)
    plot_distance_history(all_runs_data, summary_df, output_folder)
    plot_dv_vs_distance(all_runs_data, summary_df, output_folder)
    plot_reward_heatmap(all_runs_data, summary_df, output_folder)
    plot_sun_angle_vs_distance(all_runs_data, summary_df, output_folder)
    plot_impulse_histogram(raw_runs_data, summary_df, output_folder)
    plot_mean_impulse_history(raw_runs_data, summary_df, output_folder)

    # Best / worst single-run plots
    worst_run_id = int(summary_df.sort_values("total_reward", ascending=True).iloc[0]["run_id"])
    best_run_id = int(summary_df.sort_values("total_reward", ascending=False).iloc[0]["run_id"])

    worst_run_df = all_runs_data[worst_run_id - 1]
    best_run_df = all_runs_data[best_run_id - 1]

    processed_best = process_sim_data(best_run_df)
    processed_worst = process_sim_data(worst_run_df)

    best_folder = os.path.join(output_folder, "best_run")
    worst_folder = os.path.join(output_folder, "worst_run")
    os.makedirs(best_folder, exist_ok=True)
    os.makedirs(worst_folder, exist_ok=True)

    plot_control_analysis(processed_best, best_folder)
    plot_trajectory_analysis(processed_best, best_folder)
    plot_single_run_rewards(best_run_df, best_folder, prefix="best_")

    plot_control_analysis(processed_worst, worst_folder)
    plot_trajectory_analysis(processed_worst, worst_folder)
    plot_single_run_rewards(worst_run_df, worst_folder, prefix="worst_")

    raw_best = raw_runs_data[best_run_id - 1]
    raw_worst = raw_runs_data[worst_run_id - 1]
    plot_impulse_timeseries(raw_best, raw_worst, output_folder)

    print("\nAll plots complete!")


if __name__ == "__main__":
    # --- Configure paths here ---
    OUTPUT_FOLDER = r"D:\tanne\Desktop\results"
    DATA_CSV      = os.path.join(OUTPUT_FOLDER, "mc_all_runs_data.csv")
    SUMMARY_CSV   = os.path.join(OUTPUT_FOLDER, "mc_summary_stats.csv")
    # ----------------------------

    parser = argparse.ArgumentParser(description="Regenerate MC plots from saved CSV files.")
    parser.add_argument("--data",    default=DATA_CSV,      help="Path to mc_all_runs_data.csv")
    parser.add_argument("--summary", default=SUMMARY_CSV,   help="Path to mc_summary_stats.csv")
    parser.add_argument("--output",  default=OUTPUT_FOLDER, help="Output folder for plots")
    args = parser.parse_args()

    run_postprocess(args.data, args.summary, args.output)
