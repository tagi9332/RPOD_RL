import numpy as np

# Basilisk & BSK-RL imports
from bsk_rl import data
from bsk_rl.sim import fsw

# Import custom classes
from src.rewarders.docking_corridor_rewarder import DockingCorridorReward
from src.rewarders.quadratic_time_penalty import QuadraticTimePenalty
from src.rewarders.rel_range_rewarder import RelativeRangeLogReward
from src.rewarders.target_illumination_rewarder import IlluminationReward

# Import weights and constants
from resources import (
    dv_reward_weight,
    rel_range_log_weight,
    approach_corridor_weight,
    time_penalty_weight,
    SIM_TIME,
    docking_port_boresight,
    illumination_weight,
    sun_illumination_cone_angle_deg,
    illumination_cutoff_range,

)

def get_rewarders():
    """Builds and returns the tuple of rewarders for the RL environment."""
    return (
        data.ResourceReward(
            resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0,
            reward_weight=dv_reward_weight, 
        ),
        RelativeRangeLogReward(
            alpha=rel_range_log_weight, 
            delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])
        ),
        DockingCorridorReward(
            weight=approach_corridor_weight, 
            docking_port_boresight=docking_port_boresight, 
            cutoff_range=1000
        ),
        QuadraticTimePenalty(
            max_penalty=time_penalty_weight, 
            max_sim_time=SIM_TIME, 
            power=2.0
        ),
        IlluminationReward(
            weight=illumination_weight,
            cutoff_range=illumination_cutoff_range,
            cone_angle_deg=sun_illumination_cone_angle_deg
        )

    )