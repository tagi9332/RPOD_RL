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
# hierarchical MDP over three spatial phases and a terminal event layer:
#
#   Phase 1  Long-range rendezvous          initial location (~1000–1200 m) → 500 m
#   Phase 2  Close-range waypoint navigation 500 m → 30 m
#   Phase 3  Terminal ingress and docking    30 m → 5 m (CONJUNCTION_RADIUS)
#
# Phase 1 → 2 is not a latched state: range-gated terms (IlluminationReward,
# DockingCorridorReward, the waypoint braking ramp) switch on/off at 500 m from the
# instantaneous range. Phase 2 → 3 is latched inside WaypointPhaseReward the first
# time range < PHASE_TRANSITION_RANGE (30 m).
#
# Internal flag mapping: WaypointPhaseData.phase == 0 during Phases 1–2 and == 1
# during Phase 3. vel_weight_phase0 / vel_weight_phase1 follow the same convention.
#
# ── ALL PHASES
# ──────────────────────────────────────────────────────────────────────────────────────
#  DeltaVReward            −weight × |Δv|² applied on every step a burn is executed.
#                          Quadratic scaling discourages large impulsive manoeuvres.
#
#  QuadraticTimePenalty    Step-size-independent integral penalty. Rate ∝ (t/T)²,
#                          so the penalty is near-zero early and steep near episode
#                          end. Integrates to exactly −max_episode_penalty over a
#                          full episode regardless of drift step sizes.
#
# ── PHASE 1: Long-range rendezvous  (initial location → 500 m)
# ──────────────────────────────────────────────────────────────────────────────────────
#  IlluminationReward      Active only when range > illumination_cutoff_range (500 m).
#                          Rewards approaching from the sunlit side of the RSO within
#                          a cone_angle_deg (60°) of the sun direction; penalises
#                          outside that cone proportionally to the angular deviation.
#
#  WaypointPhaseReward     Dense log-MSE of body-frame position error to the 30 m
#  (dense, position only)  body-fixed standoff waypoint, normalised by MAX_REL_POS.
#                          No velocity term — the braking ramp is zero beyond
#                          vel_onset_range (500 m) from the waypoint.
#
# ── PHASE 2: Close-range waypoint navigation  (500 m → 30 m)
# ──────────────────────────────────────────────────────────────────────────────────────
#  WaypointPhaseReward     Same dense log-MSE to the 30 m standoff waypoint. Velocity
#  (dense, braking)        braking term ramps linearly from zero at vel_onset_range
#                          to full weight (vel_weight_phase0) at the waypoint.
#
#  DockingCorridorReward   Active only when range < docking_phase_range_threshold (500 m),
#                          through Phases 2 and 3. Rewards corridor alignment
#                          proportionally to (cosθ − cos_limit) / (1 − cos_limit);
#                          penalises outside the corridor scaled by both angular
#                          deviation and proximity to the RSO.
#
# ── PHASE 2 → 3 TRANSITION: Relative range
# ──────────────────────────────────────────────────────────────────────────────────────
#  WaypointPhaseReward     Phase 3 begins the first step the Inspector's relative range
#  (sparse)                to the docking port drops below PHASE_TRANSITION_RANGE (30 m),
#                          from any direction. One-time sparse bonus (waypoint_sparse_reward)
#                          on that step, scaled by transverse offset from the boresight
#                          axis (zero at ≥ WAYPOINT_CAPTURE_RADIUS). Phase is latched —
#                          the Inspector stays in Phase 3 for the remainder of the
#                          episode even if it moves back out.
#
# ── PHASE 3: Terminal ingress and docking  (30 m → 5 m)
# ──────────────────────────────────────────────────────────────────────────────────────
#  WaypointPhaseReward     Dense log-MSE of body-frame position error to the docking
#  (dense)                 port (RSO body-frame origin), normalised by PHASE_TRANSITION_RANGE.
#                          Velocity penalty is always-on and uses a stronger weight
#                          (vel_weight_phase1) to enforce controlled final ingress.
#
#  DockingCorridorReward   Continues from Phase 2 (see above).
#
# ── TERMINAL EVENTS  (fire once, end episode)
# ──────────────────────────────────────────────────────────────────────────────────────
#  SparseEventReward       Three one-shot events detected on state transitions.
#                          Docking / conjunction ends Phase 3 at CONJUNCTION_RADIUS (5 m):
#    Docking success:      Conjunction within approach_corridor_angle_deg of the
#                          docking port boresight → +docking_reward × alignment_mult
#                          + fuel bonus, where alignment_mult = 1 - angle/180.
#                          Gated by WAYPOINT_GATE_ENABLED.
#    Off-axis conjunction: Conjunction outside the corridor → +docking_reward × alignment_mult
#                          (no fuel bonus).
#    Max-range violation:  Inspector exceeds max_range_radius → −max_range_penalty.
#
# ── WAYPOINT GATE  (optional)
# ──────────────────────────────────────────────────────────────────────────────────────
#  When WAYPOINT_GATE_ENABLED = True, the docking bonus in SparseEventReward is
#  blocked unless WaypointPhaseReward has recorded a Phase 3 entry for that
#  satellite (i.e. the Inspector crossed PHASE_TRANSITION_RANGE during the episode).
#
# ============================================================

# Set to True to require Phase 3 entry before the docking bonus is awarded.
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
