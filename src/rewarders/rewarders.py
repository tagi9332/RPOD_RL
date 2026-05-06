import numpy as np

from bsk_rl.sim import fsw

from src.rewarders.docking_corridor_rewarder import DockingCorridorReward
from src.rewarders.quadratic_time_penalty import QuadraticTimePenalty
from src.rewarders.rel_range_rewarder import RelativeRangeLogReward
from src.rewarders.target_illumination_rewarder import IlluminationReward
from src.rewarders.dv_rewarder import DeltaVReward
from src.rewarders.sparse_event_rewarder import SparseEventReward

from resources import (
    dv_reward_weight,
    rel_range_log_weight,
    approach_corridor_weight,
    time_penalty_weight,
    illumination_weight,
    sun_illumination_cone_angle_deg,
    illumination_cutoff_range,
    docking_port_boresight,
    SIM_TIME,
)

# ============================================================
# REWARD BUDGET  (expected cumulative contribution per episode)
# ============================================================
# Component              | Typical range       | Notes
# -----------------------|---------------------|---------------------------
# DeltaVReward           | -0.5  …  0          | -w * |Δv|^2 per burn
# RelativeRangeLogReward | -1.0  … +1.0        | log-MSE, peaks near target
# DockingCorridorReward  | -0.5  … +0.5        | alignment shaping ≤100 m
# QuadraticTimePenalty   |  0    … -65         | integral form, negligible
#                        |                     |   early, severe at t→T
# IlluminationReward     | -0.1  … +0.1        | solar lighting guidance
# SparseEventReward      |  -10  or  +8…+10    | terminal: dock/collision/range
# ============================================================


def get_rewarders():
    """Builds and returns the tuple of rewarders for the RL environment."""
    return (
        DeltaVReward(
            reward_weight=dv_reward_weight,
            exponent=2.0,
        ),
        RelativeRangeLogReward(
            alpha=rel_range_log_weight,
            delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0]),
        ),
        DockingCorridorReward(
            weight=approach_corridor_weight,
            docking_port_boresight=docking_port_boresight,
            cutoff_range=1000,
        ),
        QuadraticTimePenalty(
            max_episode_penalty=time_penalty_weight,
            max_sim_time=SIM_TIME,
            power=2.0,
        ),
        IlluminationReward(
            weight=illumination_weight,
            cutoff_range=illumination_cutoff_range,
            cone_angle_deg=sun_illumination_cone_angle_deg,
        ),
        SparseEventReward(),
    )
