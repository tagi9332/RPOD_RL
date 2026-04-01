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
from Basilisk.utilities import macros

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
    plot_interactive_trajectories,
    plot_mc_distributions,
    plot_all_trajectories,
    plot_summary_table,
    plot_pareto_front,
    plot_single_run_rewards
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
            
            # --- NEW: Extract Reward Components ---
            if hasattr(self.env, 'rewarder'):
                rewarder_iterable = self.env.rewarder if isinstance(self.env.rewarder, (list, tuple)) else [self.env.rewarder]
                for rew_obj in rewarder_iterable:
                    name = rew_obj.__class__.__name__
                    if name == "ResourceReward":
                        # BSK-RL's built-in ResourceReward stores the last computed dict in .reward
                        val = rew_obj.reward.get("Inspector", 0.0) if hasattr(rew_obj, 'reward') else 0.0
                    else:
                        # Our custom telemetry hook saves it to .last_reward
                        val = getattr(rew_obj, 'last_reward', 0.0)
                    
                    info["metrics"][f"rew_{name}"] = val        
        return obs, reward, terminated, truncated, info

# --- Inference Loop ---
def enable_eval_reward_telemetry(env):
    """
    Patches rewarders, including those nested inside a ComposedReward.
    """
    # 1. Get a flat list of all rewarders
    raw_rewarders = env.rewarder if isinstance(env.rewarder, (list, tuple)) else [env.rewarder]
    flat_rewarders = []
    
    for r in raw_rewarders:
        if r.__class__.__name__ == "ComposedReward":
            # If it's a container, grab everything inside it
            flat_rewarders.extend(r.rewarders)
        else:
            flat_rewarders.append(r)

    # 2. Patch each individual rewarder
    for rew_obj in flat_rewarders:
        if hasattr(rew_obj, 'calculate_reward'):
            orig_method = rew_obj.calculate_reward
            
            def create_patched_method(obj, orig_fn):
                def patched_calculate_reward(new_data_dict):
                    reward_dict = orig_fn(new_data_dict)
                    # Store the component value
                    obj.last_reward = reward_dict.get("Inspector", 0.0)
                    return reward_dict
                return patched_calculate_reward
            
            rew_obj.calculate_reward = create_patched_method(rew_obj, orig_method)

def run_monte_carlo_inference(model_path, output_folder, num_runs=30):
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))
    
    rewarders = get_rewarders()

    print("Initializing Environment...")
    env = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=inspector_sat_args)],
        sat_arg_randomizer=sat_arg_randomizer(mode="test", rso_att_type="velocity"), 
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

        # ====================================================================
        # --- VIZARD INTEGRATION OVERHAUL ---
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

        # --- THE OVERHAUL: RPO & DOCKING VISUALS ---
        
        # A. Trajectory & Relative Motion (CRITICAL)
        # Instead of dizzying global orbits, draw the Inspector's path relative to the RSO
        viz.settings.orbitLinesOn = 2              # 2 = Relative to chief spacecraft
        viz.settings.trueTrajectoryLinesOn = 2     # 2 = True path relative to chief
        viz.settings.relativeOrbitFrame = 1        # 1 = Use Hill Frame (Standard for RPO)
        viz.settings.showHillFrame = 1             # Draw the Hill frame axes (Radial, Along-track, Cross-track)
        viz.liveSettings.relativeOrbitChief = rso_sc.ModelTag # Set RSO as the center of the universe

        # B. Spacecraft Frames & Labels
        viz.settings.showSpacecraftLabels = 1
        viz.settings.spacecraftCSon = 1            # Show local X, Y, Z axes of the spacecraft
        viz.settings.showCSLabels = 1              # Label those axes (helps align docking ports)

        # C. Camera & HUD Settings
        viz.settings.mainCameraTarget = insp_sc.ModelTag # Focus the camera on your RL agent
        viz.settings.viewCameraBoresightHUD = 1    # Draws a line out the front of your camera
        viz.settings.viewCameraFrustumHUD = 1      # Draws the sensor cone
        viz.settings.showDataRateDisplay = -1      # Hide clutter
        viz.settings.ambient = 0.5                 # Brighten the dark side of the spacecraft a bit

        # D. Actuator Visibility (Debug the Suicide Burn)
        # This will show a UI panel of thruster forces AND draw plumes coming out of the ship!
        vizSupport.setActuatorGuiSetting(viz, 
            spacecraftName=insp_sc.ModelTag,
            viewThrusterPanel=1, 
            viewThrusterHUD=1, 
            showThrusterLabels=1
        )

        # ====================================================================
        # --- ADVANCED GEOMETRY OVERLAY (CORRECTED) ---
        
        # 1. The Docking Keep-In Corridor
        # createConeInOut draws a physical cone and checks the angle between the 
        # normal vector and the vector to the target body.
        vizSupport.createConeInOut(viz,
            fromBodyName=rso_sc.ModelTag,      # Cone originates from the RSO
            toBodyName=insp_sc.ModelTag,       # Evaluates if the Inspector is inside it
            normalVector_B=docking_port_boresight,    # Direction of the RSO's docking port (+X axis)
            incidenceAngle=approach_corridor_angle_deg * macros.D2R,  # 15-degree half-angle safe approach corridor
            coneHeight=200.0,                  # Draw the cone out to 200 meters
            coneColor="green", 
            isKeepIn=True,                     # True = Keep-In cone
            coneName="dockingCorridor"
        )

        # 2. Inspector Sensor Boresight
        vizSupport.createConeInOut(viz,
            fromBodyName=insp_sc.ModelTag,     # Originates from Inspector
            toBodyName=rso_sc.ModelTag,        # Evaluates if RSO is within the FOV
            normalVector_B=inspector_boresight,    # Assuming Inspector sensor points along its +X
            incidenceAngle=20.0 * macros.D2R,  # 20-degree half-angle FOV
            coneHeight=30,                  # Max range of the sensor
            coneColor="cyan",
            isKeepIn=True,                     
            coneName="sensorBoresight"
        )

        # 3. Dynamic Sun Vector
        # Draws a line constantly pointing from the Inspector to the Sun.
        vizSupport.createPointLine(viz,
            fromBodyName=insp_sc.ModelTag,
            toBodyName="sun",                  # Vizard recognizes "sun" natively
            lineColor="yellow" 
        )
        
        # 4. Target Vector 
        # Draws a line from the Inspector directly to the RSO.
        vizSupport.createPointLine(viz,
            fromBodyName=insp_sc.ModelTag,
            toBodyName=rso_sc.ModelTag,
            lineColor="white"     
        )
        # ====================================================================

        # 6. Initialize memory without wiping RL states
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
    model_path = r"models/training_run_2026-03-31_20-08-21/rpo_min_dv_spec.zip"
    #---------------------------------------------------------------------------------

    all_runs_data, summary_df = run_monte_carlo_inference(model_path, output_folder, num_runs=50)  

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
    plot_single_run_rewards(best_run_df, os.path.join(output_folder, "best_run"), prefix="best_")

    plot_control_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_trajectory_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_single_run_rewards(worst_run_df, os.path.join(output_folder, "worst_run"), prefix="worst_")

    print("\nAll inferences and plots complete!")