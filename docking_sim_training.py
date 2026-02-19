# Standard Libraries
import types
import numpy as np
from functools import partial
import pandas as pd
import os

# Basilisk Core
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from Basilisk.architecture import bskLogging

# BSK-RL Framework
from bsk_rl import sats, obs, act, ConstellationTasking, scene, data
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief, cd2hill
from bsk_rl.sim import dyn, fsw

# RL Libraries
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Import weights
from weights import reward_weight,alpha,weight

# --- CONFIGURATION DICTIONARIES ---
rso_sat_args = dict(
    conjunction_radius=2.0,
    K=7.0 / 20,
    P=35.0 / 20,
    Ki=1e-6,
    dragCoeff=0.0,
    batteryStorageCapacity=1e9, 
    storedCharge_Init=1e9,
    wheelSpeeds=[0.0, 0.0, 0.0],
    u_max=2.0,
)

inspector_sat_args = dict(
    imageAttErrorRequirement=1.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e12,
    storedCharge_Init=1e12,
    conjunction_radius=50.0,
    dv_available_init=500.0,
    max_range_radius=5000,
    chief_name="RSO",
    u_max=2.0
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
    def requires_retasking(self, value):
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
    observation_spec = [
        # Trimmed observation spec from previous discussion
        obs.SatProperties(
            dict(prop="dv_available", norm=50),
            # dict(prop="eccentricity", norm=0.1),
            # dict(prop="semi_major_axis", norm=7000),
            # dict(prop="true_anomaly", norm=2 * np.pi),
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500), 
            dict(prop="v_DC_Hc", norm=5), 
            # dict(prop="sun_hat_Hc", fn=sun_hat_chief), 
            chief_name="RSO",
        ),
        # obs.Eclipse(norm=5700), 
        # obs.Time(norm=3000), # Normalize time to episode length
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO", max_dv=2, max_drift_duration=20,
        )
    ]

    # action_spec = [
    #     act.ImpulsiveThrustSphericalLOS(
    #         chief_name="RSO", 
    #         max_dv=0.5, 
    #         max_drift_duration=30, 
    #         fsw_action="action_inspect_rso"
    #     )
    # ]

    dyn_type = types.new_class("Dyn", (dyn.MaxRangeDynModel, dyn.ConjunctionDynModel, dyn.RSOInspectorDynModel))
    # fsw_type = types.new_class("FSW", (fsw.SteeringFSWModel, fsw.MagicOrbitalManeuverFSWModel, fsw.RSOInspectorFSWModel))
    fsw_type = types.new_class("FSW", (fsw.MagicOrbitalManeuverFSWModel, fsw.RSOInspectorFSWModel))


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
            reward_dict[self.agent_name] += 100.0  # Large positive reward for conjunction (for testing)

            # Debug print
            inspector_sat.logger.info(f"Conjunction occurred with {[sat.name for sat in inspector_sat.dynamics.conjunctions]} at sim time {self.env.simulator.sim_time:.2f} seconds")
            inspector_sat.logger.info(f"final episode reward: {reward_dict[self.agent_name]:.4f}")

        # Check max range violation
        max_range = inspector_sat.sat_args.get("max_range_radius", 10000)
        rho, rho_d = cd2hill(rso_r_N, rso_v_N, inspector_r_N, inspector_v_N)
        r_rel_mag = np.linalg.norm(rho)
        if r_rel_mag > max_range:
            info["max_range_violation"] = True
            reward_dict[self.agent_name] -= 100.0  # Large negative reward for max range violation
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
    R_E = 6371.0 * 1000
    a = R_E + np.random.uniform(35776.0*1000, 35796.0*1000) # Near GEO orbit
    e = np.random.uniform(0.0, 0.0005)
    chief_orbit = random_orbit(a=a, e=e)
    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]
    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO", chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate((random_unit_vector() * np.random.uniform(5000, 100), random_unit_vector() * np.random.uniform(0, 0.01))),
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

# Callback with disk flushing to save RAM
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

    rewarders = (
        data.ResourceReward(
            resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0,
            reward_weight=reward_weight, 
        ),
        data.RelativeRangeLogReward(alpha=alpha, delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])),

        data.RelativeCosineReward(weight=weight)
    )

    rso = RSOSat("RSO", sat_args=rso_sat_args)
    inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

    env = ConstellationTasking(
        satellites=[rso, inspector],
        sat_arg_randomizer=sat_arg_randomizer,
        scenario=scenario,
        rewarder=rewarders,
        time_limit=3000,
        sim_rate=1.0, 
        log_level="ERROR",
    )

    env_sb3 = SB3CompatibleEnv(env)
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
        learning_rate=8e-4,
        max_grad_norm=1,
    )

    # sim_logger = SimulationLoggerCallback(save_freq=50) # Flushes CSV every 50 updates
    model.learn(total_timesteps=1000, callback=None)

    model.save("ppo_inspector_singlecore")
    print("Model saved as ppo_inspector_singlecore.zip")