import numpy as np

# Basilisk & BSK-RL imports
from bsk_rl import data
from bsk_rl.sim import fsw

# Import custom classes
from utils.rewarders.docking_corridor_rewarder import DockingCorridorReward
from utils.rewarders.quadratic_time_penalty import QuadraticTimePenalty
from utils.rewarders.rel_range_rewarder import RelativeRangeLogReward

# Import weights and constants
from resources import (
    dv_reward_weight,
    rel_range_log_weight,
    docking_corridor_weight,
    time_penalty_weight,
    SIM_TIME
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
            weight=docking_corridor_weight, 
            docking_port_boresight=np.array([0.0, 0.0, 1.0]), 
            cutoff_range=1000
        ),
        QuadraticTimePenalty(
            max_penalty=time_penalty_weight, 
            max_sim_time=SIM_TIME, 
            power=2.0
        ),
    )