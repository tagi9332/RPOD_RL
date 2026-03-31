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
from Basilisk.utilities import vizSupport

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
    plot_interactive_trajectories,
    plot_mc_distributions,
    plot_all_trajectories,
    plot_summary_table,
    plot_pareto_front
)
# Import weights
from resources import (
    approach_corridor_angle_deg,
    inspector_boresight,
    docking_port_boresight
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
        dist = np.linalg.norm(insp_r_N - rso_r_N)

        # ----------------------------------------------------------------------------
        # METRIC 1: Approach Angle (Inspector's position relative to RSO's docking port)
        # ----------------------------------------------------------------------------
        rso_sigma_BN = np.array(rso.dynamics.sigma_BN)
        dcm_BN_rso = rbk.MRP2C(rso_sigma_BN)
        
        # Vector FROM RSO TO Inspector
        r_rel_rso_to_insp_N = insp_r_N - rso_r_N

        if dist > 1e-6:
            r_rel_B_hat = np.dot(dcm_BN_rso, r_rel_rso_to_insp_N / dist)
            approach_angle_rad = np.arccos(np.clip(np.dot(r_rel_B_hat, docking_port_boresight), -1.0, 1.0))
            approach_angle_deg = np.degrees(approach_angle_rad)
        else:
            approach_angle_deg = 0.0

        # ----------------------------------------------------------------------------
        # METRIC 2: Pointing Error (Inspector's camera aiming at the RSO)
        # ----------------------------------------------------------------------------
        sigma_BN_insp = np.array(inspector.dynamics.sigma_BN)
        dcm_BN_insp = rbk.MRP2C(sigma_BN_insp) 

        # Vector FROM Inspector TO RSO (Notice the subtraction order is flipped!)
        r_rel_insp_to_rso_N = rso_r_N - insp_r_N 
        
        if dist > 1e-6:
            u_Target_N = r_rel_insp_to_rso_N / dist 
            u_Target_B = np.dot(dcm_BN_insp, u_Target_N) 
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
            
            # Hardware Metrics
            insp_torque_cmd = inspector.dynamics.satellite.data_store.satellite.dynamics.satellite.fsw.rwMotorTorque.rwMotorTorqueOutMsg.payloadPointer.motorTorque[0:3]
            wheel_speeds = inspector.data_store.satellite.fsw.satellite.dynamics.wheel_speeds
            info["metrics"]["torque_cmd"] = insp_torque_cmd
            info["metrics"]["wheel_speeds"] = wheel_speeds
            
            # Consumables
            info["metrics"]["dV_remaining"] = inspector.fsw.dv_available if hasattr(inspector.fsw, 'dv_available') else 0.0
        
        return obs, reward, terminated, truncated, info

# --- Inference Loop ---
def run_monte_carlo_inference(model_path, output_folder, num_runs=30):
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))
    
    rewarders = get_rewarders()

    print("Initializing Environment...")
    env = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=inspector_sat_args)],
        sat_arg_randomizer=sat_arg_randomizer(mode="test"), scenario=scenario, rewarder=rewarders, time_limit=SIM_TIME, sim_rate=SIM_DT, log_level="WARNING"
    )

    env_sb3 = InferenceEnv(env)
    env_sb3 = FlattenObservation(env_sb3)

    try:
        model = PPO.load(model_path, device="cpu")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return None, None

    all_runs_data = [] 
    summary_stats = [] 

    for run_idx in range(num_runs):
        print(f"--- Executing Run {run_idx + 1}/{num_runs} ---")
        obs, _ = env_sb3.reset()
        if isinstance(obs, tuple): obs = obs[0]

        # ====================================================================
        # --- VIZARD INTEGRATION ---
        # 1. Grab the raw Basilisk simulator directly from the base env
        sim = env.simulator

        # 2. Extract the C++ spacecraft objects so Vizard knows what to draw
        rso_sc = env.satellites[0].dynamics.scObject
        insp_sc = env.satellites[1].dynamics.scObject
        sc_objects = [rso_sc, insp_sc]

        # 3. Create a unique save path for this run's Vizard data
        viz_save_dir = os.path.abspath(os.path.join(output_folder, "vizard_data"))
        os.makedirs(viz_save_dir, exist_ok=True)
        viz_filepath = os.path.join(viz_save_dir, f"run_{run_idx + 1}_vizard.bin") 

        # 4. Find a valid task name dynamically from the C++ core
        task_names = [task.TaskPtr.TaskName for proc in sim.TotalSim.processList for task in proc.processTasks]
        viz_task_name = next((name for name in task_names if 'dyn' in name.lower()), task_names[0])

        # 5. Enable the recorder and capture the viz module object
        viz = vizSupport.enableUnityVisualization(
            sim, 
            viz_task_name, 
            sc_objects, 
            saveFile=viz_filepath
        )

        # --- IMPROVED: Sensor Boresight Visualization ---
        
        # Define the boresight direction in the Inspector's Body Frame (B)
        # If your sensor is on the +X axis, use [1, 0, 0]
        boresight_body = [1.0, 0.0, 0.0] 
        
        # # Create a custom sensor to visualize the boresight
        # # This will draw a cone projecting from the satellite
        # vizSupport.createCustomSensor(viz, 
        #     scName=insp_sc.ModelTag,      # Attach to Inspector
        #     location=[0.0, 0.0, 0.0],     # Origin of sensor (Body frame meters)
        #     normalVector=boresight_body,  # Direction sensor is pointing (Body frame)
        #     fieldOfView=20.0 * (np.pi/180.0), # 20 degree cone (helpful for visual alignment)
        #     range=500.0,                  # How far the visual beam extends (meters)
        #     color=vizSupport.toRGBA255("cyan")
        # )

        # # Draw a line specifically to the RSO as well? 
        # # (Optional: keep this if you want to see the error between boresight and target)
        # vizSupport.createPointLine(viz, 
        #     fromBodyName=insp_sc.ModelTag, 
        #     toBodyName=rso_sc.ModelTag, 
        #     lineColor=[255, 255, 255, 50] # Faint white line for "Ideal" vector
        # )
        
        # Keep these enabled—they are your best friends for troubleshooting!
        viz.settings.showSpacecraftLabels = 1
        viz.settings.showSpacecraftLocalFrames = 1
        # Turn on labels and local body frames (X, Y, Z axes) so you can see 
        # exactly how the Inspector's docking port aligns with the RSO
        viz.settings.showSpacecraftLabels = 1
        viz.settings.showSpacecraftLocalFrames = 1 

        # 6. CRITICAL FIX: Do NOT call sim.InitializeSimulation() here!
        # env.reset() already set up your randomized initial states. 
        # Instead, just initialize the specific Vizard module we just added to the task
        # so it allocates memory without wiping your RL environment states.
        viz.Reset(0)
        # ====================================================================
            
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
        
        total_sim_time = len(run_df) * SIM_DT
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
        elif len(run_df) >= (SIM_TIME / SIM_DT) * 0.99: 
            end_status = "Timeout"
        else: 
            end_status = "Fuel Exhausted / Bounds Viol."
        # -------------------------

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
    model_path = r"models/training_run_2026-03-31_09-16-08/rpo_min_dv_spec.zip"
    #---------------------------------------------------------------------------------

    all_runs_data, summary_df = run_monte_carlo_inference(model_path, output_folder, num_runs=10)  

    if all_runs_data:
        plot_all_trajectories(all_runs_data, summary_df, output_folder)
        plot_interactive_trajectories(all_runs_data, summary_df, output_folder)
        plot_summary_table(summary_df, output_folder)
        plot_mc_distributions(all_runs_data, summary_df, output_folder)
        plot_pareto_front(all_runs_data, summary_df, output_folder)

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
    animate_results(processed_data_best, os.path.join(output_folder, "best_run"))

    plot_control_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_trajectory_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    animate_results(processed_data_worst, os.path.join(output_folder, "worst_run"))

    print("\nAll inferences and plots complete!")