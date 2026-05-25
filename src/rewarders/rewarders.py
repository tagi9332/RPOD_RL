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
# HIERARCHICAL REWARD STRUCTURE
# ============================================================
#
# The reward is composed of six components that together define a
# hierarchical MDP over two spatial phases and a terminal event layer.
#
# ── PHASE 0: Long-range approach  (Inspector > WAYPOINT_CAPTURE_RADIUS from waypoint)
# ──────────────────────────────────────────────────────────────────────────────────────
#  IlluminationReward      Active only when range > illumination_cutoff_range (150 m).
#                          Rewards approaching from the sunlit side of the RSO within
#                          a cone_angle_deg (60°) of the sun direction; penalises
#                          outside that cone proportionally to the angular deviation.
#
#  WaypointPhaseReward     Dense log-MSE of body-frame position error to the 30 m
#  (Phase 0, dense)        body-fixed standoff waypoint, normalised by MAX_REL_POS.
#                          Velocity braking term is proximity-scaled: zero when far
#                          from the waypoint, ramping to full weight at vel_onset_range.
#
#  DockingCorridorReward   Active only when range < docking_phase_range_threshold (120 m).
#                          Rewards corridor alignment proportionally to (cosθ − cos_limit)
#                          / (1 − cos_limit); penalises outside the corridor scaled by
#                          both angular deviation and proximity to the RSO.
#
#  DeltaVReward            −weight × |Δv|² applied on every step a burn is executed.
#                          Quadratic scaling discourages large impulsive manoeuvres.
#
#  QuadraticTimePenalty    Step-size-independent integral penalty. Rate ∝ (t/T)²,
#                          so the penalty is near-zero early and steep near episode
#                          end. Integrates to exactly −max_episode_penalty over a
#                          full episode regardless of drift step sizes.
#
# ── PHASE TRANSITION: Waypoint capture
# ──────────────────────────────────────────────────────────────────────────────────────
#  WaypointPhaseReward     One-time sparse bonus (waypoint_sparse_reward) awarded on
#  (sparse)                the single step the Inspector first enters the WAYPOINT_CAPTURE_RADIUS
#                          (5 m) sphere around the body-fixed standoff point. Phase is
#                          latched — the Inspector stays in Phase 1 for the remainder
#                          of the episode even if it exits the sphere.
#
# ── PHASE 1: Terminal ingress  (after waypoint capture)
# ──────────────────────────────────────────────────────────────────────────────────────
#  WaypointPhaseReward     Dense log-MSE of body-frame position error to the docking
#  (Phase 1, dense)        port (RSO body-frame origin), normalised by STANDOFF_DISTANCE.
#                          Velocity penalty is always-on and uses a stronger weight
#                          (vel_weight_phase1) to enforce controlled final ingress.
#
# ── TERMINAL EVENTS  (fire once, end episode)
# ──────────────────────────────────────────────────────────────────────────────────────
#  SparseEventReward       Three one-shot events detected on state transitions:
#    Docking success:      Conjunction within approach_corridor_angle_deg of the
#                          docking port boresight → +docking_reward × alignment_mult,
#                          where alignment_mult ∈ [misalignment_discount_factor, 1]
#                          scales with approach angle. Gated by WAYPOINT_GATE_ENABLED.
#    Collision:            Conjunction outside the corridor → +conjunction_penalty (<0).
#    Max-range violation:  Inspector exceeds max_range_radius → +max_range_penalty (<0).
#
# ── WAYPOINT GATE  (optional)
# ──────────────────────────────────────────────────────────────────────────────────────
#  When WAYPOINT_GATE_ENABLED = True, the docking bonus in SparseEventReward is
#  blocked unless WaypointPhaseReward has recorded a Phase 1 transition for that
#  satellite (i.e. the waypoint was physically captured during the episode).
#
# ============================================================

# Set to True to require waypoint capture before the docking bonus is awarded.
WAYPOINT_GATE_ENABLED = False


def get_rewarders(dv_weight=None):
    """Builds and returns the tuple of rewarders for the RL environment.

    Args:
        dv_weight: float, callable, or None.  If a callable is passed (e.g.
            randomizer.get_dv_weight), DeltaVReward will invoke it each episode
            reset so the active weight tracks whatever the randomizer sampled.
            Pass None to use the default dv_reward_weight from resources.
    """
    gate = WaypointGate() if WAYPOINT_GATE_ENABLED else None
    return (
        DeltaVReward(
            reward_weight=dv_weight if dv_weight is not None else dv_reward_weight,
            exponent=1.0,
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
