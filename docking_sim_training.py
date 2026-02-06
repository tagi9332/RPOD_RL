# Standard Libraries
import types
import numpy as np
from functools import partial
import pandas as pd
from matplotlib import pyplot as plt

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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)



# Define Cheif (RSO) Satellite Class
class RSOSat(sats.Satellite):
    observation_spec = [
        obs.SatProperties(dict(prop="one", fn=lambda _: 1.0)),
    ]
    action_spec = [act.Drift(duration=60.0)]
    dyn_type = types.new_class(
        "Dyn", (dyn.ImagingDynModel, dyn.ConjunctionDynModel, dyn.RSODynModel)
    )
    fsw_type = fsw.ContinuousImagingFSWModel

rso_sat_args = dict(
    conjunction_radius=2.0,
    K=7.0 / 20,
    P=35.0 / 20,
    Ki=1e-6,
    dragCoeff=0.0,
    batteryStorageCapacity=1e9, # Very large to avoid running out of power
    storedCharge_Init=1e9,
    wheelSpeeds=[0.0, 0.0, 0.0],
    u_max=2.0,
)

def sun_hat_chief(self, other):
    r_SN_N = (
        self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = self.dynamics.r_BN_N
    r_SN_N = np.array(r_SN_N)
    r_SB_N = r_SN_N - r_BN_N
    r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)
    HN = other.dynamics.HN
    return HN @ r_SB_N_hat

# Define Inspector Satellite Class
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
            dict(prop="beta_angle", norm=np.pi), # angle between sun vector and orbital plane
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            dict(prop="r_DC_Hc", norm=500), # relative position from deputy to chief in chief hill frame (rho vector)
            dict(prop="v_DC_Hc", norm=5), # relative velocity from deputy to chief in chief hill frame (rho dot vector)
            dict(
                prop="rso_imaged_regions",
                fn=partial(
                    rso_imaged_regions,
                    region_centers=fibonacci_sphere(15),
                    frame="chief_hill",
                ),
            ),
            dict(prop="sun_hat_Hc", fn=sun_hat_chief), # sun direction in chief hill frame
            chief_name="RSO",
        ),
        obs.Eclipse(norm=5700), # eclipse observation
        obs.Time(), # simulation time
    ]
    action_spec = [
        act.ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=10.0,
            max_drift_duration=5700.0 * 2,
            fsw_action="action_inspect_rso",
        )
    ]
    dyn_type = types.new_class(
        "Dyn",
        (
            dyn.MaxRangeDynModel,
            dyn.ConjunctionDynModel,
            dyn.RSOInspectorDynModel,
        ),
    )
    fsw_type = types.new_class(
        "FSW",
        (
            fsw.SteeringFSWModel,
            fsw.MagicOrbitalManeuverFSWModel,
            fsw.RSOInspectorFSWModel,
        ),
    )

inspector_sat_args = dict(
    imageAttErrorRequirement=1.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e12,
    storedCharge_Init=1e12,
    conjunction_radius=2.0,
    dv_available_init=50.0,
    max_range_radius=10000,
    chief_name="RSO",
    u_max=0.01
)

# Define satellite argument randomizer (Initializes random simulation parameters at each env.reset)
def sat_arg_randomizer(satellites):
    # Generate the RSO orbit
    R_E = 6371.0  # km
    a = R_E + np.random.uniform(500, 1100)
    e = np.random.uniform(0.0, min(1 - (R_E + 500) / a, (R_E + 1100) / a - 1))
    chief_orbit = random_orbit(a=a, e=e)

    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]

    # Generate the inspector initial states.
    args = {}
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO",
            chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate(
                    (
                        random_unit_vector() * np.random.uniform(250, 750),
                        random_unit_vector() * np.random.uniform(0, 0.1),
                    )
                ),
            },
        )
        args.update(relative_randomizer([rso, inspector]))

    # Align RSO Hill frame for initial nadir pointing
    mu = rso.sat_args_generator["mu"]
    r_N, v_N = elem2rv(mu, args[rso]["oe"])

    r_hat = r_N / np.linalg.norm(r_N)
    v_hat = v_N / np.linalg.norm(v_N)
    x = r_hat
    z = np.cross(r_hat, v_hat)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    HN = np.array([x, y, z])
    BH = np.eye(3)

    a = chief_orbit.a
    T = np.sqrt(a**3 / mu) * 2 * np.pi # type: ignore
    omega_BN_N = z * 2 * np.pi / T

    args[rso]["sigma_init"] = C2MRP(BH @ HN)
    args[rso]["omega_init"] = BH @ HN @ omega_BN_N

    return args

scenario = scene.SphericalRSO(
    n_points=100,
    radius=1.0,
    theta_max=np.radians(30),
    range_max=250,
    theta_solar_max=np.radians(60),
)

# Initialize reward functions
rewarders = (
    data.RSOInspectionReward(
        completion_bonus=1.0,
        completion_threshold=0.90,
    ),
    data.ResourceReward(
        resource_fn=lambda sat: sat.fsw.dv_available
        if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel)
        else 0.0,
        # reward_weight=np.random.uniform(0.0, 0.5),
        reward_weight=0.001,
    ),
    # Custom relative range log reward
    data.RelativeRangeLogReward(
        alpha=-1,
        delta_x_max=np.array([1000, 1000, 1000, 1, 1, 1]),
    ),
)

# Initialize Chief (RSO) and Deputy (Inspector) satellites
rso = RSOSat("RSO", sat_args=rso_sat_args)
inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

# Create the constellation tasking environment
env = ConstellationTasking(
    satellites=[rso, inspector],
    sat_arg_randomizer=sat_arg_randomizer,
    scenario=scenario,
    rewarder=rewarders,
    time_limit=60000,
    sim_rate=5.0, # Somehow 2x multiplier on this is needed to get the expected time steps. Not sure why.
    log_level="INFO",
)

# SB3 Compatibility Wrapper
class SB3CompatibleEnv(gym.Env):
    def __init__(self, env, agent_name="Inspector"):
        self.env = env
        self.agent_name = agent_name

        self.observation_space = env.observation_space(agent_name)
        self.action_space = env.action_space(agent_name)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)  # Gymnasium style
        return obs_dict[self.agent_name], info

    def step(self, action):
            # Basilisk simulation step
            obs_dict, reward_dict, terminated_dict, truncated_dict, info = self.env.step({self.agent_name: action})

            # Extract BSK sim data
            rso_sat = self.env.satellites[0]
            inspector_sat = self.env.satellites[1]

            # Extract inertial position and velocity
            rso_r_N = np.array(rso_sat.dynamics.r_BN_N)
            rso_v_N = np.array(rso_sat.dynamics.v_BN_N)
            inspector_r_N = np.array(inspector_sat.dynamics.r_BN_N)
            inspector_v_N = np.array(inspector_sat.dynamics.v_BN_N)

            # Extract inspector attitude (MRPs) and rates
            inspector_sigma_BN = np.array(inspector_sat.dynamics.sigma_BN)
            inspector_omega_BN_B = np.array(inspector_sat.dynamics.omega_BN_B)

            # Compute relative Hill frame state
            hill_state = cd2hill(rso_r_N, rso_v_N, inspector_r_N, inspector_v_N)
            r_DC_Hc = hill_state[0]
            v_DC_Hc = hill_state[1]

            # Get fuel
            dv_remaining = 0.0
            if hasattr(inspector_sat.fsw, 'dv_available'):
                dv_remaining = inspector_sat.fsw.dv_available

            # Store in info Dictionary
            info = {
                "metrics": {
                    "rso_r_N": rso_r_N,
                    "rso_v_N": rso_v_N,
                    "inspector_r_N": inspector_r_N,
                    "inspector_v_N": inspector_v_N,
                    "r_DC_Hc": r_DC_Hc,
                    "v_DC_Hc": v_DC_Hc,
                    "inspector_sigma_BN": inspector_sigma_BN,
                    "inspector_omega_BN_B": inspector_omega_BN_B,
                    "reward": reward_dict[self.agent_name],
                    "dV_remaining": dv_remaining
                }
            }

            obs = obs_dict[self.agent_name]
            reward = reward_dict[self.agent_name]
            terminated = terminated_dict[self.agent_name]
            truncated = truncated_dict[self.agent_name]
            
            return obs, reward, terminated, truncated, info
    

class SimulationLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SimulationLoggerCallback, self).__init__(verbose)
        self.data_log = [] # List to store our data

    def _on_step(self) -> bool:
        # 'self.locals' gives us access to local variables in the model.learn() function
        # 'infos' is a list of info dicts (one for each environment in the VecEnv)
        infos = self.locals['infos']
        
        # Loop through environments (you likely only have 1)
        for info in infos:
            if 'metrics' in info:
                # Append the metrics to our log list
                self.data_log.append(info['metrics'])
        
        return True # Returning False would stop training early

    def get_dataframe(self):
        # Helper to retrieve data as a Pandas DataFrame
        return pd.DataFrame(self.data_log)


if __name__ == "__main__":
    # Wrap the environment to be SB3 compatible    
    env_sb3 = SB3CompatibleEnv(env)
    env_sb3 = FlattenObservation(env_sb3)
    env_sb3_vec = DummyVecEnv([lambda: env_sb3])
    # model = PPO("MlpPolicy", env_sb3_vec, verbose=1, device="cuda")

    model = PPO(
        "MlpPolicy", 
        env_sb3_vec, 
        verbose=1, 
        device="cpu",
        n_steps=2048,
        batch_size=64, # Must be a factor of n_steps
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )

    # Set up the simulation logger callback
    sim_logger = SimulationLoggerCallback()

    # Initiate model learning
    model.learn(total_timesteps=100000, callback=sim_logger)

    # # Save model
    model.save("ppo_inspector_v1")
    print("Model saved as ppo_inspector_v1.zip")
