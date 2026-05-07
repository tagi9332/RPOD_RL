import numpy as np

from src.rewarders.docking_corridor_rewarder import DockingCorridorReward
from src.rewarders.quadratic_time_penalty import QuadraticTimePenalty
from src.rewarders.target_illumination_rewarder import IlluminationReward
from src.rewarders.dv_rewarder import DeltaVReward
from src.rewarders.sparse_event_rewarder import SparseEventReward
from src.rewarders.waypoint_phase_rewarder import WaypointGate, WaypointPhaseReward

from resources import (
    dv_reward_weight,
    approach_corridor_weight,
    time_penalty_weight,
    illumination_weight,
    sun_illumination_cone_angle_deg,
    illumination_cutoff_range,
    docking_port_boresight,
    SIM_TIME,
    docking_phase_range_threshold,
)

# ============================================================
# REWARD BUDGET  (expected cumulative contribution per episode)
# ============================================================
# Component              | Typical range       | Notes
# -----------------------|---------------------|---------------------------
# DeltaVReward           | -0.5  …  0          | -w * |Δv|^2 per burn
# WaypointPhaseReward    | -1.0  … +2.0        | log-MSE body-frame, both
#   dense                |                     |   phases + braking curve
#   sparse (waypoint)    | 0 or +5             | one-time on capture
# DockingCorridorReward  | -0.5  … +0.5        | alignment shaping ≤100 m
# QuadraticTimePenalty   |  0    … -80         | integral, zero early,
#                        |                     |   full weight at t=T
# IlluminationReward     | -0.1  … +0.1        | solar lighting guidance
# SparseEventReward      |  -2   or  +16…+20   | terminal: dock/coll/range
# ============================================================

# Set to True to require waypoint capture before the docking bonus is awarded.
WAYPOINT_GATE_ENABLED = True


def get_rewarders():
    """Builds and returns the tuple of rewarders for the RL environment."""
    gate = WaypointGate() if WAYPOINT_GATE_ENABLED else None
    return (
        DeltaVReward(
            reward_weight=dv_reward_weight,
            exponent=2.0,
        ),
        WaypointPhaseReward(gate=gate),
        DockingCorridorReward(
            weight=approach_corridor_weight,
            docking_port_boresight=docking_port_boresight,
            cutoff_range=docking_phase_range_threshold,
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
        SparseEventReward(waypoint_gate=gate),
    )
