# Standard Libraries
import types
import numpy as np
from functools import partial
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import os

# Basilisk Core
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from Basilisk.architecture import bskLogging

# BSK-RL Framework
from bsk_rl import sats, obs, act, ConstellationTasking, scene, data
from bsk_rl.obs.relative_observations import rso_imaged_regions
from bsk_rl.utils.orbital import fibonacci_sphere
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief
from bsk_rl.utils.orbital import cd2hill

# RL Libraries (Stable Baselines3 and Gymnasium)
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Reduce logging noise
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# -----------------------------------------------------------------------
# 1. SATELLITE & PHYSICS DEFINITIONS
# -----------------------------------------------------------------------

class RSOSat(sats.Satellite):
    observation_spec = [obs.SatProperties(dict(prop="one", fn=lambda _: 1.0))]
    action_spec = [act.Drift(duration=60.0)]
    dyn_type = types.new_class("Dyn", (dyn.ImagingDynModel, dyn.ConjunctionDynModel, dyn.RSODynModel))
    fsw_type = fsw.ContinuousImagingFSWModel

def sun_hat_chief(self, other):
    r_SN_N = (
        self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.simulator.world.sun_index
        ].read().PositionVector
    )
    r_BN_N = self.dynamics.r_BN_N
    r_SN_N = np.array(r_SN_N)
    r_SB_N = r_SN_N - r_BN_N
    r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)
    HN = other.dynamics.HN
    return HN @ r_SB_N_hat

class InspectorSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="dv_available", norm=50),
            dict(prop="inclination", norm=np.pi),
            dict(prop="eccentricity", norm=0.1),
            dict(prop="semi_major_axis", norm=7000),
            dict(prop="ascending_node", norm=2 * np.pi),
            dict(prop="argument_of_periapsis", norm=2 * np.pi),
            dict(prop="true_anomaly", norm=2 * np.pi),
            dict(prop="beta_angle", norm=np.pi),
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500),
            dict(prop="v_DC_Hc", norm=5),
            dict(
                prop="rso_imaged_regions",
                fn=partial(
                    rso_imaged_regions,
                    region_centers=fibonacci_sphere(15),
                    frame="chief_hill",
                ),
            ),
            dict(prop="sun_hat_Hc", fn=sun_hat_chief),
            chief_name="RSO",
        ),
        obs.Eclipse(norm=5700),
        obs.Time(),
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=2.0,
            max_drift_duration=5700.0 * 2,
            fsw_action="action_inspect_rso",
        )
    ]
    dyn_type = types.new_class("Dyn", (dyn.MaxRangeDynModel, dyn.ConjunctionDynModel, dyn.RSOInspectorDynModel))
    fsw_type = types.new_class("FSW", (fsw.SteeringFSWModel, fsw.MagicOrbitalManeuverFSWModel, fsw.RSOInspectorFSWModel))

def sat_arg_randomizer(satellites):
    R_E = 6371.0 
    a = R_E + np.random.uniform(500, 1100)
    e = np.random.uniform(0.0, min(1 - (R_E + 500) / a, (R_E + 1100) / a - 1))
    chief_orbit = random_orbit(a=a, e=e)

    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]

    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO",
            chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate(
                    (random_unit_vector() * np.random.uniform(250, 750), random_unit_vector() * np.random.uniform(0, 1.0))
                ),
            },
        )
        args.update(relative_randomizer([rso, inspector]))

    mu = rso.sat_args_generator["mu"]
    r_N, v_N = elem2rv(mu, args[rso]["oe"])
    r_hat = r_N / np.linalg.norm(r_N)
    v_hat = v_N / np.linalg.norm(v_N)
    x = r_hat
    z = np.cross(r_hat, v_hat); z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    HN = np.array([x, y, z])
    BH = np.eye(3)
    a = chief_orbit.a
    T = np.sqrt(a**3 / mu) * 2 * np.pi 
    omega_BN_N = z * 2 * np.pi / T
    args[rso]["sigma_init"] = C2MRP(BH @ HN)
    args[rso]["omega_init"] = BH @ HN @ omega_BN_N
    return args

# -----------------------------------------------------------------------
# 2. GYM WRAPPER & LOGGING
# -----------------------------------------------------------------------

class SB3CompatibleEnv(gym.Env):
    def __init__(self, env, agent_name="Inspector"):
        self.env = env
        self.agent_name = agent_name
        self.observation_space = env.observation_space(agent_name)
        self.action_space = env.action_space(agent_name)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict[self.agent_name], info

    def step(self, action):
        obs_dict, reward_dict, terminated_dict, truncated_dict, info = self.env.step({self.agent_name: action})
        
        # Extract Telemetry for Logging
        rso_sat = self.env.satellites[0]
        inspector_sat = self.env.satellites[1]
        
        # Coordinate transforms
        rso_r_N = np.array(rso_sat.dynamics.r_BN_N)
        rso_v_N = np.array(rso_sat.dynamics.v_BN_N)
        inspector_r_N = np.array(inspector_sat.dynamics.r_BN_N)
        inspector_v_N = np.array(inspector_sat.dynamics.v_BN_N)
        hill_state = cd2hill(rso_r_N, rso_v_N, inspector_r_N, inspector_v_N)
        
        dv_remaining = 0.0
        if hasattr(inspector_sat.fsw, 'dv_available'):
            dv_remaining = inspector_sat.fsw.dv_available

        info = {
            "metrics": {
                "r_DC_Hc": hill_state[0],
                "v_DC_Hc": hill_state[1],
                "inspector_sigma_BN": np.array(inspector_sat.dynamics.sigma_BN),
                "reward": reward_dict[self.agent_name],
                "dV_remaining": dv_remaining
            }
        }
        return obs_dict[self.agent_name], reward_dict[self.agent_name], terminated_dict[self.agent_name], truncated_dict[self.agent_name], info

class SimulationLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SimulationLoggerCallback, self).__init__(verbose)
        self.data_log = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'metrics' in info:
                self.data_log.append(info['metrics'])
        return True

    def get_dataframe(self):
        return pd.DataFrame(self.data_log)

# -----------------------------------------------------------------------
# 3. ENVIRONMENT FACTORY (Required for Multiprocessing)
# -----------------------------------------------------------------------

def make_bsk_env():
    """
    Creates a NEW instance of the Basilisk environment.
    This function is called by every CPU core.
    """
    # Re-define arguments inside to ensure they are local to the process
    rso_sat_args = dict(
        conjunction_radius=2.0, K=7.0/20, P=35.0/20, Ki=1e-6, dragCoeff=0.0,
        batteryStorageCapacity=1e9, storedCharge_Init=1e9, wheelSpeeds=[0.0, 0.0, 0.0], u_max=1.0,
    )
    
    inspector_sat_args = dict(
        imageAttErrorRequirement=1.0, imageRateErrorRequirement=None, instrumentBaudRate=1,
        dataStorageCapacity=1e6, batteryStorageCapacity=1e9, storedCharge_Init=1e9,
        conjunction_radius=2.0, dv_available_init=50.0, max_range_radius=10000,
        chief_name="RSO", u_max=2.0,
    )

    scenario = scene.SphericalRSO(
        n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60),
    )

    rewarders = (
        data.RSOInspectionReward(completion_bonus=1.0, completion_threshold=0.90),
        data.ResourceReward(
            resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0,
            reward_weight=np.random.uniform(0.0, 0.5),
        ),
        data.RelativeRangeLogReward(alpha=-0.1, delta_x_max=np.array([1000, 1000, 1000, 1, 1, 1])),
    )

    rso = RSOSat("RSO", sat_args=rso_sat_args)
    inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

    env = ConstellationTasking(
        satellites=[rso, inspector],
        sat_arg_randomizer=sat_arg_randomizer,
        scenario=scenario,
        rewarder=rewarders,
        time_limit=60000,
        sim_rate=5.0,
        log_level="WARNING", # Reduced log level for speed
    )

    env_sb3 = SB3CompatibleEnv(env)
    return FlattenObservation(env_sb3)

# -----------------------------------------------------------------------
# 4. PLOTTING FUNCTION
# -----------------------------------------------------------------------

def process_and_plot(df):
    df['hill_x'] = df['r_DC_Hc'].apply(lambda v: v[0])
    df['hill_y'] = df['r_DC_Hc'].apply(lambda v: v[1])
    df['hill_z'] = df['r_DC_Hc'].apply(lambda v: v[2])
    df['range_mag'] = df['r_DC_Hc'].apply(np.linalg.norm)
    df['vel_mag'] = df['v_DC_Hc'].apply(np.linalg.norm)
    df['cumulative_reward'] = df['reward'].cumsum()
    
    time_step = 5.0
    df['time_sec'] = df.index * time_step
    df['time_min'] = df['time_sec'] / 60.0

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle(f"Inspector Agent Analysis (Steps: {len(df)})", fontsize=16)

    # Hill Frame Trajectory (2D)
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(df['hill_y'], df['hill_x'], label='Trajectory', color='blue', marker='.', markersize=2)
    ax1.scatter(0, 0, color='red', marker='*', s=100, label='Target (RSO)')
    ax1.set_title("In-Plane Motion (Hill Frame)")
    ax1.set_xlabel("Along-Track [m] (y)")
    ax1.set_ylabel("Radial [m] (x)")
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()

    # Hill Frame Trajectory (3D)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(df['hill_x'], df['hill_y'], df['hill_z'], label='Path')
    ax2.scatter(0, 0, 0, color='red', marker='*', s=100, label='RSO')
    ax2.set_title("3D Relative Trajectory")
    ax2.set_xlabel("Radial (x)")
    ax2.set_ylabel("Along-Track (y)")
    ax2.set_zlabel("Cross-Track (z)")

    # Approach Metrics
    ax3 = fig.add_subplot(2, 3, 3)
    color = 'tab:blue'
    ax3.set_xlabel('Time [min]')
    ax3.set_ylabel('Range [m]', color=color)
    ax3.plot(df['time_min'], df['range_mag'], color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3_twin = ax3.twinx()  
    color = 'tab:orange'
    ax3_twin.set_ylabel('Velocity [m/s]', color=color)  
    ax3_twin.plot(df['time_min'], df['vel_mag'], color=color, linestyle='--')
    ax3_twin.tick_params(axis='y', labelcolor=color)
    ax3.set_title("Approach Metrics")
    ax3.grid(True, alpha=0.3)

    # Fuel
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df['time_min'], df['dV_remaining'], color='green')
    ax4.set_title("Fuel Remaining")
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Delta-V [m/s]")
    ax4.grid(True)

    # Reward
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(df['time_min'], df['reward'], alpha=0.5, color='gray', label='Step Reward')
    ax5.plot(df['time_min'], df['reward'].rolling(window=10).mean(), color='purple', label='Smoothed')
    ax5.set_title("Reward Function")
    ax5.set_xlabel("Time [min]")
    ax5.set_ylabel("Reward")
    ax5.legend()
    ax5.grid(True)
    
    # Attitude
    df['sigma_mag'] = df['inspector_sigma_BN'].apply(np.linalg.norm)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(df['time_min'], df['sigma_mag'], color='brown')
    ax6.set_title("Attitude Magnitude")
    ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("|sigma|")
    ax6.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # --- A. PARALLEL TRAINING PHASE ---
    # Reserve 2 cores for OS, use rest for Training
    num_cpu = max(1, multiprocessing.cpu_count() - 2)
    print(f"Launching Training on {num_cpu} cores via SubprocVecEnv...")
    
    # Create parallel environment
    vec_env = make_vec_env(make_bsk_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    # Initialize PPO
    # Adjusted batch/steps for parallel execution
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        device="cpu", 
        n_steps=2048 // max(1, num_cpu), 
        batch_size=64
    )
    
    # Train
    total_timesteps = 50_000 # Increased so you actually see CPU usage
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training Complete.")
    
    # Save Model
    model.save("ppo_bsk_docking")
    vec_env.close() # Close the parallel processes

    # --- B. VERIFICATION & PLOTTING PHASE ---
    print("\nStarting Verification Episode (Single Thread)...")
    
    # Create a SINGLE environment for plotting (avoids messy interleaved data)
    eval_env = DummyVecEnv([make_bsk_env])
    
    # Create a fresh logger just for this episode
    eval_logger = SimulationLoggerCallback()
    
    # Run one episode manually
    obs = eval_env.reset()
    done = False
    
    while not done:
        # Use the trained model to predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, rewards, dones, infos = eval_env.step(action)
        
        # Manually trigger logger (since we aren't using model.learn here)
        # We construct a fake 'locals' dict to satisfy the callback signature
        eval_logger.locals = {'infos': infos}
        eval_logger._on_step()
        
        done = dones[0]

    # Get data and Plot
    print("Processing Data...")
    df = eval_logger.get_dataframe()
    df.to_csv("simulation_log_verification.csv", index=False)
    
    if len(df) > 1:
        process_and_plot(df)
    else:
        print("Verification episode ended too quickly to plot.")