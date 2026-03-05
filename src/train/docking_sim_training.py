# Standard libraries
import types
import numpy as np
import pandas as pd
import os

# Basilisk core
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

# BSK-RL framework
from bsk_rl import sats, obs, act, ConstellationTasking, scene, data
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief, cd2hill
from bsk_rl.sim import dyn, fsw

# RL libraries
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor\

# Import custom observations
from utils.observations import (
    custom_sigma_DC,
    custom_r_DC_C,
)

# Import custom rewarder function
from utils.rewarders import get_rewarders

# Import weights
from resources import (
    docking_reward,
    max_range_penalty,
    conjunction_penalty,
    R_EARTH,
    learning_rate,
    entropy_coeff,
    max_grad_norm,
    docking_corridor_angle_deg,
    clip_range
)

# Set BSK logging level
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Import simulation parameters
from resources import (
    SIM_TIME,
    SIM_DT,
    MAX_REL_POS,
    MAX_REL_VEL,
    MIN_REL_POS,
    MIN_REL_VEL,
    MAX_DV,
    MAX_DRIFT_DURATION,
    rso_sat_args,
    inspector_sat_args,
)

# --- CLASS DEFINITIONS ---

class RSOSat(sats.Satellite):
    observation_spec = [obs.SatProperties(dict(prop="one", fn=lambda _: 1.0))]
    action_spec = [act.Drift(duration=60.0)]
    dyn_type = types.new_class("Dyn", (dyn.ImagingDynModel, dyn.ConjunctionDynModel, dyn.RSODynModel))
    fsw_type = fsw.ContinuousImagingFSWModel

    @property
    def requires_retasking(self) -> bool:
        """The RSO never needs a new task from the RL agent."""
        return False
    
    @requires_retasking.setter
    def requires_retasking(self, value): # type: ignore
        """Ignore attempts by the environment to set this to True."""
        pass

def sun_hat_chief(self, other):
    r_SN_N = self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[self.simulator.world.sun_index].read().PositionVector
    r_BN_N = self.dynamics.r_BN_N
    r_SN_N = np.array(r_SN_N)
    r_SB_N = r_SN_N - r_BN_N
    r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)
    HN = other.dynamics.HN
    return HN @ r_SB_N_hat

class InspectorSat(sats.Satellite):
    # observation_spec = [
    #     # Trimmed observation spec
    #     obs.SatProperties(
    #         dict(prop="dv_available", norm=50),
    #     ),
    #     obs.ResourceRewardWeight(),
    #     obs.RelativeProperties(
    #         dict(prop="r_DC_Hc", norm=500), 
    #         dict(prop="v_DC_Hc", norm=5), 
    #         chief_name="RSO",
    #     ),
    # ]
    observation_spec = [
        # Full observation spec
        obs.SatProperties(
            dict(prop="dv_available", norm=50),
            # dict(prop="eccentricity", norm=0.1),
            # dict(prop="semi_major_axis", norm=35786.0*1000),
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500),
            dict(prop="v_DC_Hc", norm=5), 
            dict(prop="r_DC_C", fn=custom_r_DC_C, norm=500),
            dict(prop="sun_hat_Hc", fn=sun_hat_chief),
            chief_name="RSO",
        ),
        # obs.Eclipse(norm=1.0), 
        obs.Time(norm=SIM_TIME), # Normalize time to episode length
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO", max_dv=MAX_DV, max_drift_duration=MAX_DRIFT_DURATION,
        )
    ]
    dyn_type = types.new_class("Dyn", (dyn.MaxRangeDynModel, dyn.ConjunctionDynModel, dyn.RSOInspectorDynModel))
    # fsw_type = types.new_class("FSW", (fsw.SteeringFSWModel, fsw.MagicOrbitalManeuverFSWModel, fsw.RSOInspectorFSWModel))
    fsw_type = types.new_class("FSW", (fsw.MagicOrbitalManeuverFSWModel, fsw.RSOInspectorFSWModel))


class Sb3BksEnv(gym.Env):
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

        # Extract Metrics
        rso_sat = self.env.satellites[0]
        inspector_sat = self.env.satellites[1]

        rso_r_N = np.array(rso_sat.dynamics.r_BN_N)
        rso_v_N = np.array(rso_sat.dynamics.v_BN_N)
        inspector_r_N = np.array(inspector_sat.dynamics.r_BN_N)
        inspector_v_N = np.array(inspector_sat.dynamics.v_BN_N)
        
        inspector_sigma_BN = np.array(inspector_sat.dynamics.sigma_BN)
        inspector_omega_BN_B = np.array(inspector_sat.dynamics.omega_BN_B)

        hill_state = cd2hill(rso_r_N, rso_v_N, inspector_r_N, inspector_v_N)
        dv_remaining = inspector_sat.fsw.dv_available if hasattr(inspector_sat.fsw, 'dv_available') else 0.0

        # Check conjunction status
        if hasattr(inspector_sat.dynamics, 'conjunctions') and inspector_sat.dynamics.conjunctions:
            info["conjunction"] = True
            info["conjunction_with"] = [sat.name for sat in inspector_sat.dynamics.conjunctions]
            
            # Get RSO attitude and convert to Direction Cosine Matrix (Inertial -> Body)
            rso_sigma_BN = np.array(rso_sat.dynamics.sigma_BN)
            dcm_BN = rbk.MRP2C(rso_sigma_BN) 
            
            # Calculate the relative position vector (pointing FROM RSO TO Inspector)
            r_rel_N = inspector_r_N - rso_r_N 
            dist = np.linalg.norm(r_rel_N)
            
            if dist > 1e-6:
                # Rotate the normalized relative position into the RSO Body Frame
                r_rel_N_hat = r_rel_N / dist
                r_rel_B_hat = np.dot(dcm_BN, r_rel_N_hat)
                
                # Define your docking port boresight (assuming Z-axis here)
                boresight_B = np.array([0.0, 0.0, 1.0]) 
                
                # Calculate the angle between the inspector's position and the boresight
                cos_theta = np.dot(r_rel_B_hat, boresight_B)
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                
                # Apply Sparse Reward based on the docking angle cone
                if angle_deg <= docking_corridor_angle_deg:
                    reward_dict[self.agent_name] += docking_reward
                    inspector_sat.logger.info(f"SUCCESSFUL DOCKING! Angle: {angle_deg:.2f} deg at sim time {self.env.simulator.sim_time:.2f}s")
                else:
                    # Optional: Apply a crash penalty if it hits the wrong side of the RSO
                    reward_dict[self.agent_name] += conjunction_penalty 
                    inspector_sat.logger.info(f"FAILED DOCKING (Collision). Angle: {angle_deg:.2f} deg at sim time {self.env.simulator.sim_time:.2f}s")

            inspector_sat.logger.info(f"final episode reward: {reward_dict[self.agent_name]:.4f}")

        # Check max range violation
        max_range = inspector_sat.sat_args.get("max_range_radius", 10000)
        rho, rho_d = cd2hill(rso_r_N, rso_v_N, inspector_r_N, inspector_v_N)
        r_rel_mag = np.linalg.norm(rho)
        if r_rel_mag > max_range:
            info["max_range_violation"] = True
            reward_dict[self.agent_name] += max_range_penalty  # Large negative reward for max range violation
        else:
            info["max_range_violation"] = False

        info["metrics"] = {
            "rso_r_N": rso_r_N,
            "rso_v_N": rso_v_N,
            "inspector_r_N": inspector_r_N,
            "inspector_v_N": inspector_v_N,
            "r_DC_Hc": hill_state[0],
            "v_DC_Hc": hill_state[1],
            "inspector_sigma_BN": inspector_sigma_BN,
            "inspector_omega_BN_B": inspector_omega_BN_B,
            "reward": reward_dict[self.agent_name],
            "docked_state": info.get("conjunction", False),
            "dV_remaining": dv_remaining,
            "max_range_violation": info.get("max_range_violation", False)
        }

        return obs_dict[self.agent_name], reward_dict[self.agent_name], terminated_dict[self.agent_name], truncated_dict[self.agent_name], info

def sat_arg_randomizer(satellites):
    a = (R_EARTH*1000) + np.random.uniform(35776.0*1000, 35796.0*1000) # Near GEO orbit
    e = np.random.uniform(0.0, 0.0005)
    chief_orbit = random_orbit(a=a, e=e)
    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]
    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO", chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate((random_unit_vector() * np.random.uniform(MIN_REL_POS, MAX_REL_POS), random_unit_vector() * np.random.uniform(MIN_REL_VEL, MAX_REL_VEL))),
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
    HN = np.array([x, y, z]); BH = np.eye(3)
    a = chief_orbit.a; T = np.sqrt(a**3 / mu) * 2 * np.pi # type: ignore
    omega_BN_N = z * 2 * np.pi / T
    args[rso]["sigma_init"] = C2MRP(BH @ HN)
    args[rso]["omega_init"] = BH @ HN @ omega_BN_N
    return args

# Logger callback with disk flushing for RAM management
class SimulationLoggerCallback(BaseCallback):
    def __init__(self, save_freq: int = 1000, save_path: str = "training_log_singlecore.csv", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.data_log = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'metrics' in info:
                self.data_log.append(info['metrics'])
        
        if self.n_calls % self.save_freq == 0:
            self.save_to_csv()
        return True

    def save_to_csv(self):
        if not self.data_log:
            return
        df = pd.DataFrame(self.data_log)
        write_header = not os.path.exists(self.save_path)
        df.to_csv(self.save_path, mode='a', header=write_header, index=False)
        self.data_log = [] # Clear RAM

    def _on_training_end(self) -> None:
        self.save_to_csv()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))

    # Rewarder function (returns the tuple of rewarders to use in the environment)
    rewarders = get_rewarders()

    rso = RSOSat("RSO", sat_args=rso_sat_args)
    inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

    env = ConstellationTasking(
        satellites=[rso, inspector],
        sat_arg_randomizer=sat_arg_randomizer,
        scenario=scenario,
        rewarder=rewarders,
        time_limit=SIM_TIME,
        sim_rate=SIM_DT, 
        log_level="ERROR",
    )

    env_sb3 = Sb3BksEnv(env)
    env_sb3 = Monitor(env_sb3) # Aligned with multi-core (Crucial for SB3 logging)
    env_sb3 = FlattenObservation(env_sb3)
    env_sb3_vec = DummyVecEnv([lambda: env_sb3])

    model = PPO(
        "MlpPolicy", 
        env_sb3_vec, 
        verbose=1, 
        device="cpu",
        n_steps=2048,
        batch_size=64,
        clip_range=clip_range,
        learning_rate=learning_rate,
        ent_coef=entropy_coeff,
        max_grad_norm=max_grad_norm,
    )

    # sim_logger = SimulationLoggerCallback(save_freq=50) # Flushes CSV every 50 updates
    model.learn(total_timesteps=1000, callback=None)

    # Model saving
    output_dir = "./models/"
    os.makedirs(output_dir, exist_ok=True)    
    model.save(os.path.join(output_dir, "ppo_inspector_singlecore"))
    print("Model saved to models/ppo_inspector_singlecore.zip")