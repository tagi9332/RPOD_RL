# Standard Imports
import os
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

# BSK-RL and Basilisk imports
from bsk_rl import ConstellationTasking, scene
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

from stable_baselines3.common.vec_env import DummyVecEnv

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
)

# Import custom satellite argument randomizer
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer as sat_arg_randomizer

# Import custom rewarders
from src.rewarders import get_rewarders

# Local plotting scripts
from utils.plotting import (
    animate_results,
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
    vizard_output
)
# Import weights
from resources import (
    approach_corridor_angle_deg,
    inspector_boresight,
    docking_port_boresight,
    STANDOFF_DISTANCE,
    WAYPOINT_CAPTURE_RADIUS,
)

# Import sim parameters
from resources import (
    SIM_TIME,
    SIM_DT,
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
        self._waypoint_ever_captured = False
        self._waypoint_capture_dist = float('inf')

    def reset(self, **kwargs):
        self.current_sim_time = 0.0
        self._waypoint_ever_captured = False
        self._waypoint_capture_dist = float('inf')
        return super().reset(**kwargs)

    def step(self, action):
        # 1. Run normal training step logic
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_sim_time = self.env.simulator.sim_time

        # 2. Extract extra telemetry just for inference plots
        rso = self.env.satellites[0]
        inspector = self.env.satellites[1]

        rso_r_N = np.array(rso.dynamics.r_BN_N)
        rso_v_N = np.array(rso.dynamics.v_BN_N)
        insp_r_N = np.array(inspector.dynamics.r_BN_N)
        insp_v_N = np.array(inspector.dynamics.v_BN_N)
        dist = np.linalg.norm(insp_r_N - rso_r_N)

        rso_sigma_BN = np.array(rso.dynamics.sigma_BN)
        dcm_BN_rso = rbk.MRP2C(rso_sigma_BN)

        # Body-frame relative position and velocity
        r_DC_C = dcm_BN_rso @ (insp_r_N - rso_r_N)
        v_DC_C = dcm_BN_rso @ (insp_v_N - rso_v_N)

        # ----------------------------------------------------------------------------
        # METRIC 1: Waypoint capture — direct computation, same logic as DataStore
        # Bypasses the patch mechanism so it works regardless of DataStore lifecycle.
        # ----------------------------------------------------------------------------
        waypoint_B = docking_port_boresight * STANDOFF_DISTANCE
        dist_to_waypoint = float(np.linalg.norm(r_DC_C - waypoint_B))
        if dist_to_waypoint < WAYPOINT_CAPTURE_RADIUS and not self._waypoint_ever_captured:
            self._waypoint_ever_captured = True
            self._waypoint_capture_dist = dist_to_waypoint

        # ----------------------------------------------------------------------------
        # METRIC 2: Approach Angle (Inspector's position relative to RSO's docking port)
        # ----------------------------------------------------------------------------
        # Standoff waypoint in Hill frame (body-z boresight * 30m, rotated to Hill)
        dcm_HN = np.array(rso.dynamics.HN)
        r_waypoint_H = dcm_HN @ (dcm_BN_rso.T @ (docking_port_boresight * STANDOFF_DISTANCE))

        if dist > 1e-6:
            r_rel_B_hat = dcm_BN_rso @ ((insp_r_N - rso_r_N) / dist)
            approach_angle_rad = np.arccos(np.clip(np.dot(r_rel_B_hat, docking_port_boresight), -1.0, 1.0))
            approach_angle_deg = np.degrees(approach_angle_rad)
        else:
            approach_angle_deg = 0.0

        # ----------------------------------------------------------------------------
        # METRIC 3: Pointing Error (Inspector's camera aiming at the RSO)
        # ----------------------------------------------------------------------------
        sigma_BN_insp = np.array(inspector.dynamics.sigma_BN)
        dcm_BN_insp = rbk.MRP2C(sigma_BN_insp)

        if dist > 1e-6:
            u_Target_B = dcm_BN_insp @ ((rso_r_N - insp_r_N) / dist)
            pointing_error_rad = np.arccos(np.clip(np.dot(u_Target_B, inspector_boresight), -1.0, 1.0))
        else:
            pointing_error_rad = 0.0

        # ----------------------------------------------------------------------------
        # Save to Info Dictionary
        # ----------------------------------------------------------------------------
        if "metrics" in info:
            info["metrics"]["sim_time"] = self.current_sim_time

            # Attitude Metrics
            info["metrics"]["approach_angle_deg"] = approach_angle_deg
            info["metrics"]["pointing_error"] = pointing_error_rad
            info["metrics"]["rso_sigma_BN"] = rso_sigma_BN
            info["metrics"]["r_waypoint_H"] = r_waypoint_H

            # Body-frame velocity (for axial/lateral docking velocity analysis)
            info["metrics"]["v_DC_C"] = v_DC_C

            # Waypoint capture (sticky flag, reset each episode in reset())
            info["metrics"]["waypoint_captured"] = self._waypoint_ever_captured
            info["metrics"]["waypoint_capture_dist"] = self._waypoint_capture_dist

            # Hardware Metrics
            insp_torque_cmd = inspector.dynamics.satellite.data_store.satellite.dynamics.satellite.fsw.rwMotorTorque.rwMotorTorqueOutMsg.payloadPointer.motorTorque[0:3]
            wheel_speeds = inspector.data_store.satellite.fsw.satellite.dynamics.wheel_speeds
            info["metrics"]["torque_cmd"] = insp_torque_cmd
            info["metrics"]["wheel_speeds"] = wheel_speeds

            # Consumables
            info["metrics"]["dV_remaining"] = inspector.fsw.dv_available if hasattr(inspector.fsw, 'dv_available') else 0.0

            # Reward component telemetry
            if hasattr(self.env, 'rewarder'):
                rewarder_iterable = self.env.rewarder if isinstance(self.env.rewarder, (list, tuple)) else [self.env.rewarder]
                for rew_obj in rewarder_iterable:
                    name = rew_obj.__class__.__name__
                    if name == "ResourceReward":
                        val = rew_obj.reward.get("Inspector", 0.0) if hasattr(rew_obj, 'reward') else 0.0
                    else:
                        val = getattr(rew_obj, 'last_reward', 0.0)
                    info["metrics"][f"rew_{name}"] = val

        return obs, reward, terminated, truncated, info

# --- Inference Loop ---
def enable_eval_reward_telemetry(env):
    """
    Patches each rewarder's calculate_reward to store the last per-step component
    value in .last_reward so InferenceEnv.step() can log reward breakdowns.
    Handles rewarders nested inside a ComposedReward one level deep.
    """
    raw_rewarders = env.rewarder if isinstance(env.rewarder, (list, tuple)) else [env.rewarder]
    flat_rewarders = []
    for r in raw_rewarders:
        if r.__class__.__name__ == "ComposedReward":
            flat_rewarders.extend(r.rewarders)
        else:
            flat_rewarders.append(r)

    for rew_obj in flat_rewarders:
        if hasattr(rew_obj, 'calculate_reward'):
            def create_patched_method(obj, orig_fn):
                def patched_calculate_reward(new_data_dict):
                    reward_dict = orig_fn(new_data_dict)
                    obj.last_reward = reward_dict.get("Inspector", 0.0)
                    return reward_dict
                return patched_calculate_reward

            rew_obj.calculate_reward = create_patched_method(rew_obj, rew_obj.calculate_reward)

def run_monte_carlo_inference(model_path, output_folder, num_runs=30):
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))
    
    rewarders = get_rewarders()

    print("Initializing Environment...")
    env = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=inspector_sat_args)],
        sat_arg_randomizer=sat_arg_randomizer(mode="train", rso_att_type="random"), 
        scenario=scenario, 
        rewarder=rewarders, 
        time_limit=SIM_TIME, 
        sim_rate=SIM_DT, 
        log_level="WARNING"
    )

    enable_eval_reward_telemetry(env)

    # 1. Create a specific helper function to instantiate the wrapped environment
    def make_env():
        base_env = InferenceEnv(env)
        return FlattenObservation(base_env)

    # 2. Pass the function pointer directly to DummyVecEnv
    env_sb3 = DummyVecEnv([make_env])

    try:
        model = PPO.load(model_path, device="cpu")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return None, None

    all_runs_data = [] 
    summary_stats = [] 

    for run_idx in range(num_runs):
        print(f"--- Executing Run {run_idx + 1}/{num_runs} ---")
        # DummyVecEnv only returns the observation, no info dict!
        obs = env_sb3.reset()

       # Create Vizard output
        vizard_output(env, output_folder, run_idx)
            
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
        
        total_sim_time = run_df["sim_time"].max() if "sim_time" in run_df.columns else 0.0
        final_dist = np.linalg.norm([run_df.iloc[-1]["hill_x"], run_df.iloc[-1]["hill_y"], run_df.iloc[-1]["hill_z"]])
        
        # --- NEW SUCCESS LOGIC ---
        conjunction = bool(run_df.iloc[-1].get("docked_state", False))
        final_angle = run_df.iloc[-1].get("approach_angle_deg", 180.0)
        
        # Success is strictly a conjunction WITHIN the cone limit
        success = conjunction and (final_angle <= approach_corridor_angle_deg)
        
        if success: 
            end_status = f"Docked ({final_angle:.1f}°)"
        elif conjunction: 
            end_status = f"Collision ({final_angle:.1f}°)"
        elif total_sim_time >= SIM_TIME * 0.99:
            end_status = "Timeout"
        else: 
            end_status = "Fuel Exhausted / Bounds Viol."
        # --- WAYPOINT CAPTURE STATS ---
        if "waypoint_captured" in run_df.columns:
            wp_captured = bool(run_df["waypoint_captured"].max())
            if wp_captured and "waypoint_capture_dist" in run_df.columns:
                finite_dists = run_df["waypoint_capture_dist"].replace([np.inf, -np.inf], np.nan).dropna()
                wp_capture_dist = float(finite_dists.min()) if len(finite_dists) > 0 else np.nan
            else:
                wp_capture_dist = np.nan
        else:
            wp_captured = False
            wp_capture_dist = np.nan

        summary_stats.append({
            "run_id": run_idx + 1, "total_reward": total_reward, "episode_length": len(run_df),
            "total_sim_time": total_sim_time, "end_status": end_status, "success": success,
            "final_distance": final_dist, "waypoint_captured": wp_captured,
            "waypoint_capture_dist": wp_capture_dist,
        })

    print("\n=== Monte Carlo Summary ===")
    summary_df = pd.DataFrame(summary_stats)
    print(f"Mean Reward: {summary_df['total_reward'].mean():.2f} +/- {summary_df['total_reward'].std():.2f}")
    print(f"Success Rate: {(summary_df['success'].sum() / num_runs) * 100:.1f}%")
    print(f"Average Final Distance: {summary_df['final_distance'].mean():.2f}m")
    n_wp = int(summary_df['waypoint_captured'].sum())
    print(f"Waypoint Capture Rate: {(n_wp / num_runs) * 100:.1f}%  ({n_wp}/{num_runs} runs)")
    mean_wp_dist = summary_df['waypoint_capture_dist'].dropna().mean()
    if not np.isnan(mean_wp_dist):
        print(f"Mean Waypoint Capture Distance from Center: {mean_wp_dist:.2f}m")
    
    summary_df.to_csv(os.path.join(output_folder, "mc_summary_stats.csv"), index=False)
    pd.concat(all_runs_data, ignore_index=True).to_csv(os.path.join(output_folder, "mc_all_runs_data.csv"), index=False)

    return all_runs_data, summary_df

if __name__ == "__main__":
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    # --------------------------- Model Path Configuration ---------------------------
    model_path = r"models\training_run_2026-05-17_09-16-57\ppo_inspector_multicore_checkpoint_599928_steps.zip"
    #---------------------------------------------------------------------------------

    all_runs_data, summary_df = run_monte_carlo_inference(model_path, output_folder, num_runs=100)

    if all_runs_data:
        all_runs_data = [interpolate_to_uniform_time(df) for df in all_runs_data]

    if all_runs_data:
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

    # Trim down to worst and best runs
    worst_run_id = summary_df.sort_values(by="total_reward", ascending=True).iloc[0]["run_id"] #type: ignore
    worst_run_df = all_runs_data[worst_run_id - 1]  # Adjust for 0-indexing #type: ignore
    best_run_id = summary_df.sort_values(by="total_reward", ascending=False).iloc[0]["run_id"] #type: ignore
    best_run_df = all_runs_data[best_run_id - 1]  # Adjust for 0-indexing #type: ignore


    # Process data and save
    processed_data_best = process_sim_data(best_run_df)
    processed_data_worst = process_sim_data(worst_run_df)

    # Create best/worst run folders
    if not os.path.exists(os.path.join(output_folder, "best_run")):
        os.makedirs(os.path.join(output_folder, "best_run"))
    if not os.path.exists(os.path.join(output_folder, "worst_run")):
        os.makedirs(os.path.join(output_folder, "worst_run"))

    # Save processed data for best/worst runs

    plot_control_analysis(processed_data_best, os.path.join(output_folder, "best_run"))
    plot_trajectory_analysis(processed_data_best, os.path.join(output_folder, "best_run"))
    plot_single_run_rewards(best_run_df, os.path.join(output_folder, "best_run"), prefix="best_")

    plot_control_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_trajectory_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_single_run_rewards(worst_run_df, os.path.join(output_folder, "worst_run"), prefix="worst_")

    print("\nAll inferences and plots complete!")