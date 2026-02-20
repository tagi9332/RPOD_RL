import os
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

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
    SB3_BKS_env,
    sat_arg_randomizer
)

# Import custom rewarders
from src.rewarders import (
    RelativeRangeLogReward
)

# Local plotting scripts
from utils.plotting.plot_results import plot_control_analysis, plot_trajectory_analysis
from utils.plotting.animate_results import animate_results
from utils.plotting.process_sim_data import process_sim_data
from utils.plotting.plot_interactive_trajectories import plot_interactive_trajectories

# Import weights
from resources import (
    dv_reward_weight,
    rel_range_log_weight
)

# Set BSK logging level
bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

# --- 1. Custom Inference Wrapper ---
class InferenceEnv(SB3_BKS_env):
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

# --- 2. Plotting Functions ---
def plot_all_trajectories(all_runs_data, summary_df, output_folder):
    print("Generating 3D Multi-Trajectory Plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='black', marker='*', s=300, label='Target (RSO)') #type: ignore
    
    success_plotted, fail_plotted = False, False

    for idx, run_df in enumerate(all_runs_data):
        is_success = summary_df.loc[idx, "success"]
        color = 'mediumseagreen' if is_success else 'crimson'
        alpha = 0.8 if is_success else 0.3
        
        label = None
        if is_success and not success_plotted:
            label = "Successful Approach"
            success_plotted = True
        elif not is_success and not fail_plotted:
            label = "Failed/Timeout"
            fail_plotted = True

        ax.plot(run_df["hill_x"], run_df["hill_y"], run_df["hill_z"], color=color, alpha=alpha, linewidth=1.5, label=label)
        ax.scatter(run_df["hill_x"].iloc[0], run_df["hill_y"].iloc[0], run_df["hill_z"].iloc[0], color='blue', s=15, alpha=0.5) #type: ignore

    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.set_zlabel('Relative Z (m)') #type: ignore
    ax.set_title('Monte Carlo Trajectories (30 Runs)')
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

    all_runs_data, summary_df = run_monte_carlo_inference(model_path, output_folder, num_runs=20)  

    if all_runs_data:
        plot_all_trajectories(all_runs_data, summary_df, output_folder)
        plot_interactive_trajectories(all_runs_data, summary_df, output_folder)
        plot_summary_table(summary_df, output_folder)

    # Trim down to worst run
    worst_run_id = summary_df.sort_values(by="total_reward", ascending=True).iloc[0]["run_id"] #type: ignore
    worst_rin_df = all_runs_data[worst_run_id - 1]  # Adjust for 0-indexing #type: ignore
    best_run_id = summary_df.sort_values(by="total_reward", ascending=False).iloc[0]["run_id"] #type: ignore
    best_run_df = all_runs_data[best_run_id - 1]  # Adjust for 0-indexing #type: ignore


    # Process data and save
    processed_data = process_sim_data(worst_rin_df)
    processed_data.to_csv(os.path.join(output_folder, f"processed_run_{worst_run_id}.csv"), index=False)

    # Note: Removed 'run_id=best_run_id' to fix TypeError
    animate_results(processed_data, output_folder)
    plot_control_analysis(processed_data, output_folder)
    plot_trajectory_analysis(processed_data, output_folder)
    
    print("\nAll inferences and plots complete!")