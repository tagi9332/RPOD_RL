# Standard libraries
import types
import numpy as np
import pandas as pd
import os

# Basilisk core
from Basilisk.architecture import bskLogging

# BSK-RL framework
from bsk_rl import sats, obs, act, ConstellationTasking, scene
from bsk_rl.utils.orbital import cd2hill
from bsk_rl.sim import dyn, fsw

# RL libraries
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import custom observations
from utils.observations import (
    custom_sigma_DC,
    custom_r_DC_C,
    custom_v_DC_C,
    make_dist_to_waypoint_fn,
)

# Import custom FSW model for continuous pointing
from src.fsw_modules.pointing_fsw import RSOInspectorFSWModel

# Import custom action with drift capability
from src.actions.impulsive_thrust_hill import ImpulsiveThrustHill

# Import custom rewarder function
from src.rewarders import get_rewarders

# Import custom satellite argument randomizer
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer as sat_arg_randomizer

# Import weights
from resources import (
    learning_rate,
    entropy_coeff,
    max_grad_norm,
    clip_range,
)

# Set BSK logging level
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Import simulation parameters
from resources import (
    SIM_TIME,
    SIM_DT,
    MAX_DV,
    MAX_DRIFT_DURATION,
    MAX_REL_POS,
    rso_sat_args,
    inspector_sat_args,
    STANDOFF_DISTANCE,
    docking_port_boresight,
    approach_corridor_angle_deg,
)

# Import rewarder types for isinstance checks in Sb3BksEnv
from src.rewarders.sparse_event_rewarder import SparseEventReward
from src.rewarders.docking_corridor_rewarder import DockingCorridorReward

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
    observation_spec = [
        obs.SatProperties(
            dict(prop="dv_available", norm=50),
        ),
        obs.ResourceRewardWeight(),
        obs.RelativeProperties(
            # Hill-frame state — global navigation context
            dict(prop="r_DC_Hc", norm=500),
            dict(prop="v_DC_Hc", norm=5),
            # Body-frame state — docking alignment context
            dict(prop="r_DC_C",          fn=custom_r_DC_C, norm=500),
            dict(prop="v_DC_C",          fn=custom_v_DC_C, norm=1.0),
            # Phase indicator: distance to the 30 m standoff waypoint (≈0 when captured)
            dict(prop="dist_to_waypoint", fn=make_dist_to_waypoint_fn(STANDOFF_DISTANCE, docking_port_boresight), norm=MAX_REL_POS),
            dict(prop="sun_hat_Hc",      fn=sun_hat_chief),
            chief_name="RSO",
        ),
        obs.Time(norm=SIM_TIME),
    ]
    action_spec = [
        ImpulsiveThrustHill(
            chief_name="RSO",
            max_dv=MAX_DV,
            max_drift_duration=MAX_DRIFT_DURATION
        ),
    ]
    dyn_type = types.new_class("Dyn", (dyn.MaxRangeDynModel, dyn.ConjunctionDynModel, dyn.RSOInspectorDynModel))
    fsw_type = types.new_class("FSW", (fsw.MagicOrbitalManeuverFSWModel, RSOInspectorFSWModel))


class Sb3BksEnv(gym.Env):
    def __init__(self, env, agent_name="Inspector", randomizer=None):
        self.env = env
        self.agent_name = agent_name
        self.observation_space = env.observation_space(agent_name)
        self.action_space = env.action_space(agent_name)
        self.randomizer = randomizer

        # Scheduled parameters — updated by curriculum callbacks via set_attr
        self.scheduled_conjunction_radius = inspector_sat_args.get("conjunction_radius", 30)
        self.scheduled_corridor_angle_deg = approach_corridor_angle_deg
        self.scheduled_max_error_deg: float | None = None  # None → no override

        # Cache references to schedulable rewarders
        self._sparse_event_rewarder: SparseEventReward | None = None
        self._corridor_rewarder: DockingCorridorReward | None = None
        rewarder_seq = getattr(env, "rewarder", ()) or ()
        if not hasattr(rewarder_seq, "__iter__"):
            rewarder_seq = (rewarder_seq,)
        for r in rewarder_seq:
            if isinstance(r, SparseEventReward):
                self._sparse_event_rewarder = r
            elif isinstance(r, DockingCorridorReward):
                self._corridor_rewarder = r

    def set_scheduled_parameters(
        self,
        conjunction_radius=None,
        corridor_angle_deg=None,
        max_error_deg=None,
    ):
        if conjunction_radius is not None:
            self.scheduled_conjunction_radius = conjunction_radius
            for sat in self.env.satellites:
                if sat.name == "Inspector":
                    sat.sat_args["conjunction_radius"] = conjunction_radius

        if corridor_angle_deg is not None:
            self.scheduled_corridor_angle_deg = corridor_angle_deg
            if self._sparse_event_rewarder is not None:
                self._sparse_event_rewarder.corridor_angle_deg = corridor_angle_deg
            if self._corridor_rewarder is not None:
                self._corridor_rewarder.corridor_angle_deg = corridor_angle_deg

        if max_error_deg is not None:
            self.scheduled_max_error_deg = max_error_deg
            if self.randomizer is not None:
                self.randomizer.max_error_deg = max_error_deg

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        self.set_scheduled_parameters(
            conjunction_radius=self.scheduled_conjunction_radius,
            corridor_angle_deg=self.scheduled_corridor_angle_deg,
            max_error_deg=self.scheduled_max_error_deg,
        )
        return obs_dict[self.agent_name], info

    def step(self, action):
        obs_dict, reward_dict, terminated_dict, truncated_dict, info = self.env.step({self.agent_name: action})

        # --- State extraction (metrics only; all reward math lives in rewarder classes) ---
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

        # Conjunction flag (for info/logging; reward handled by SparseEventReward)
        if getattr(inspector_sat.dynamics, 'conjunctions', None):
            info["conjunction"] = True
            info["conjunction_with"] = [sat.name for sat in inspector_sat.dynamics.conjunctions]

        # Max-range flag (for info/logging; reward handled by SparseEventReward)
        max_range = inspector_sat.sat_args.get("max_range_radius", 10_000)
        info["max_range_violation"] = bool(np.linalg.norm(hill_state[0]) > max_range)

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
            "max_range_violation": info.get("max_range_violation", False),
        }

        return obs_dict[self.agent_name], reward_dict[self.agent_name], terminated_dict[self.agent_name], truncated_dict[self.agent_name], info

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
        sat_arg_randomizer=sat_arg_randomizer(mode="train", rso_att_type="velocity"),
        scenario=scenario,
        rewarder=rewarders,
        time_limit=SIM_TIME,
        sim_rate=SIM_DT, 
        log_level="ERROR",
    )

    env_sb3 = Sb3BksEnv(env)
    env_sb3 = FlattenObservation(env_sb3)
    env_sb3 = Monitor(env_sb3) # Aligned with multi-core (Crucial for SB3 logging)
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
    model.learn(total_timesteps=100, callback=None)

    # Model saving
    output_dir = "./models/"
    os.makedirs(output_dir, exist_ok=True)    
    model.save(os.path.join(output_dir, "ppo_inspector_singlecore"))
    print("Model saved to models/ppo_inspector_singlecore.zip")