# Standard Imports
import os
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib import gridspec

# BSK-RL and Basilisk imports
from bsk_rl import ConstellationTasking, scene, data
from bsk_rl.sim import fsw
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
    sat_arg_randomizer
)

# Import custom rewarders
from src.rewarders import (
    RelativeRangeLogReward
)

# Local plotting scripts
from utils.plotting import (
    animate_results,
    plot_control_analysis,
    plot_trajectory_analysis,
    process_sim_data,
    plot_interactive_trajectories
)
# Import weights
from resources import (
    dv_reward_weight,
    rel_range_log_weight
)

# Set BSK logging level
bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

# Inference environment wrapper
class InferenceEnv(Sb3BksEnv):
    """
    Inherits the exact step/reset logic from the training environment, 
    but tacks on extra telemetry for the plotting scripts.
    """
    def __init__(self, env, agent_name="Inspector"):
        super().__init__(env, agent_name) 
        self.sim_rate = getattr(env, 'sim_rate', 1.0) 
        self.current_sim_time = 0.0

    def reset(self, **kwargs):
        self.current_sim_time = 0.0 
        return super().reset(**kwargs)

    def step(self, action):
        # 1. Run normal training step logic
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_sim_time += self.sim_rate

        # 2. Extract extra telemetry just for inference plots
        rso = self.env.satellites[0]
        inspector = self.env.satellites[1]
        
        rso_r_N = np.array(rso.dynamics.r_BN_N)
        insp_r_N = np.array(inspector.dynamics.r_BN_N)
        sigma_BN = np.array(inspector.dynamics.sigma_BN)
        dcm_BN = rbk.MRP2C(sigma_BN)

        # Compute pointing error
        r_Rel_N = rso_r_N - insp_r_N
        dist = np.linalg.norm(r_Rel_N)
        u_Target_N = r_Rel_N / dist if dist > 1e-6 else np.array([1., 0., 0.])
        u_Target_B = np.dot(dcm_BN, u_Target_N)
        boresight_B = np.array([0.0, 0.0, 1.0]) 
        pointing_error_rad = np.arccos(np.clip(np.dot(u_Target_B, boresight_B), -1.0, 1.0))

        # Hardware metrics
        insp_torque_cmd = inspector.dynamics.satellite.data_store.satellite.dynamics.satellite.fsw.rwMotorTorque.rwMotorTorqueOutMsg.payloadPointer.motorTorque[0:3]
        wheel_speeds = inspector.data_store.satellite.fsw.satellite.dynamics.wheel_speeds

        # 3. Append to existing metrics dictionary
        if "metrics" in info:
            info["metrics"]["sim_time"] = self.current_sim_time
            info["metrics"]["pointing_error"] = pointing_error_rad
            info["metrics"]["torque_cmd"] = insp_torque_cmd
            info["metrics"]["wheel_speeds"] = wheel_speeds
            # Extract Fuel Remaining
            info["metrics"]["dV_remaining"] = inspector.fsw.dv_available if hasattr(inspector.fsw, 'dv_available') else 0.0
        
        return obs, reward, terminated, truncated, info

# Plotting functions

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

# --- 3. Inference Loop ---
def run_monte_carlo_inference(model_path, output_folder, num_runs=30):
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))
    
    rewarders = (
        data.ResourceReward(resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0, reward_weight=dv_reward_weight),
        RelativeRangeLogReward(alpha=rel_range_log_weight, delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])),
    )

    print("Initializing Environment...")
    env = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=inspector_sat_args)],
        sat_arg_randomizer=sat_arg_randomizer, scenario=scenario, rewarder=rewarders, time_limit=6000, sim_rate=1.0
    )

    env_sb3 = InferenceEnv(env)
    env_sb3 = FlattenObservation(env_sb3)

    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return None, None

    all_runs_data = [] 
    summary_stats = [] 

    for run_idx in range(num_runs):
        print(f"--- Executing Run {run_idx + 1}/{num_runs} ---")
        obs, _ = env_sb3.reset()
        if isinstance(obs, tuple): obs = obs[0]
            
        done = False
        run_data_log = [] 
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env_sb3.step(action)
            
            if len(step_result) == 4:
                obs, reward, done_array, info_array = step_result
                done = done_array[0]
                info = info_array[0]
                reward = reward[0] 
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += reward

            if "metrics" in info:
                metrics = info["metrics"]
                flat_metrics = {"run_id": run_idx + 1} 
                flat_metrics["reward"] = reward  # Ensure step reward is logged
                
                for k, v in metrics.items():
                    key_name = "hill" if k == "r_DC_Hc" else k
                    if isinstance(v, (np.ndarray, list)) and len(v) == 3:
                        flat_metrics[f"{key_name}_x"] = v[0]
                        flat_metrics[f"{key_name}_y"] = v[1]
                        flat_metrics[f"{key_name}_z"] = v[2]
                    else:
                        flat_metrics[key_name] = v
                        
                run_data_log.append(flat_metrics)

        run_df = pd.DataFrame(run_data_log)
        all_runs_data.append(run_df)
        
        sim_rate = env_sb3.unwrapped.sim_rate #type: ignore
        total_sim_time = len(run_df) * sim_rate
        final_dist = np.linalg.norm([run_df.iloc[-1]["hill_x"], run_df.iloc[-1]["hill_y"], run_df.iloc[-1]["hill_z"]])
        success = bool(run_df.iloc[-1].get("docked_state", False))
        
        if success: end_status = "Conjunction (Docked)"
        elif len(run_df) >= (6000 / sim_rate): end_status = "Timeout"
        else: end_status = "Fuel Exhausted / Boundary Viol."

        summary_stats.append({
            "run_id": run_idx + 1, "total_reward": total_reward, "episode_length": len(run_df),
            "total_sim_time": total_sim_time, "end_status": end_status, "success": success, "final_distance": final_dist
        })

    print("\n=== Monte Carlo Summary ===")
    summary_df = pd.DataFrame(summary_stats)
    print(f"Mean Reward: {summary_df['total_reward'].mean():.2f} +/- {summary_df['total_reward'].std():.2f}")
    print(f"Success Rate: {(summary_df['success'].sum() / num_runs) * 100:.1f}%")
    print(f"Average Final Distance: {summary_df['final_distance'].mean():.2f}m")
    
    summary_df.to_csv(os.path.join(output_folder, "mc_summary_stats.csv"), index=False)
    pd.concat(all_runs_data, ignore_index=True).to_csv(os.path.join(output_folder, "mc_all_runs_data.csv"), index=False)

    return all_runs_data, summary_df

if __name__ == "__main__":
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    # --------------------------- Model Path Configuration ---------------------------
    model_path = "models\\ppo_inspector_crawl.zip"
    #---------------------------------------------------------------------------------

    all_runs_data, summary_df = run_monte_carlo_inference(model_path, output_folder, num_runs=200)  

    if all_runs_data:
        plot_all_trajectories(all_runs_data, summary_df, output_folder)
        plot_interactive_trajectories(all_runs_data, summary_df, output_folder)
        plot_summary_table(summary_df, output_folder)
        
        # ---> ADD THE NEW FUNCTION CALL HERE <---
        plot_mc_distributions(all_runs_data, summary_df, output_folder)

    # Trim down to worst run
    worst_run_id = summary_df.sort_values(by="total_reward", ascending=True).iloc[0]["run_id"] #type: ignore
    worst_rin_df = all_runs_data[worst_run_id - 1]  # Adjust for 0-indexing #type: ignore
    best_run_id = summary_df.sort_values(by="total_reward", ascending=False).iloc[0]["run_id"] #type: ignore
    best_run_df = all_runs_data[best_run_id - 1]  # Adjust for 0-indexing #type: ignore


    # Process data and save
    processed_data = process_sim_data(worst_rin_df)

    # Note: Removed 'run_id=best_run_id' to fix TypeError
    animate_results(processed_data, output_folder)
    plot_control_analysis(processed_data, output_folder)
    plot_trajectory_analysis(processed_data, output_folder)
    
    print("\nAll inferences and plots complete!")