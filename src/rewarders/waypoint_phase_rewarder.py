"""
Two-phase body-frame rewarder for docking with an RSO at arbitrary attitude.

Phase 0  —  approach to 30 m body-fixed standoff waypoint.
             Dense:  log-MSE of position error to waypoint.
             Vel:    proximity-scaled braking penalty (zero when far, grows near waypoint).
             Sparse: one-time bonus on first entering the capture sphere.

Phase 1  —  final ingress from standoff to docking port.
             Dense:  log-MSE of position error to docking port (body-frame origin).
             Vel:    always-on, stronger penalty to enforce controlled ingress.

Phase is latched: once the Inspector enters the capture sphere it stays in Phase 1
for the rest of the episode, preventing reward oscillation if it briefly exits.
"""

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from Basilisk.utilities.RigidBodyKinematics import MRP2C

from resources import (
    waypoint_pos_weight,
    waypoint_sparse_reward,
    vel_weight_phase0,
    vel_onset_range,
    vel_weight_phase1,
    STANDOFF_DISTANCE,
    WAYPOINT_CAPTURE_RADIUS,
    VEL_NORM,
    MAX_REL_POS,
    docking_port_boresight,
)

logger = logging.getLogger(__name__)

_BORESIGHT = np.asarray(docking_port_boresight, dtype=float)
_WAYPOINT_B = _BORESIGHT * STANDOFF_DISTANCE  # body-fixed standoff point


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class WaypointPhaseData(Data):
    def __init__(
        self,
        r_DC_C: np.ndarray = np.zeros(3),
        v_DC_C: np.ndarray = np.zeros(3),
        phase: int = 0,
        dist_to_waypoint: float = float("inf"),
        is_phase_transition: bool = False,
    ):
        self.r_DC_C = r_DC_C
        self.v_DC_C = v_DC_C
        self.phase = phase
        self.dist_to_waypoint = dist_to_waypoint
        self.is_phase_transition = is_phase_transition

    def __add__(self, other: Data) -> "WaypointPhaseData":
        if isinstance(other, WaypointPhaseData):
            return other  # always use the latest step's data
        return self

    def __repr__(self) -> str:
        return (
            f"WaypointPhaseData(phase={self.phase}, "
            f"d_wp={self.dist_to_waypoint:.1f}m, "
            f"|v|={np.linalg.norm(self.v_DC_C):.3f}m/s)"
        )


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

class WaypointPhaseDataStore(DataStore):
    data_type = WaypointPhaseData

    def get_log_state(self) -> Optional[Tuple]:
        """Returns (r_DC_C, v_DC_C, dist_to_waypoint, in_capture_sphere) or None for RSO."""
        if "RSO" in self.satellite.name:
            return None

        insp_dyn = self.satellite.dynamics
        rso_dyn = self.satellite.simulator.satellites[0].dynamics

        r_insp_N = np.array(insp_dyn.r_BN_N)
        r_rso_N = np.array(rso_dyn.r_BN_N)
        v_insp_N = np.array(insp_dyn.v_BN_N)
        v_rso_N = np.array(rso_dyn.v_BN_N)

        CN = MRP2C(np.array(rso_dyn.sigma_BN))

        r_DC_C = CN @ (r_insp_N - r_rso_N)
        v_DC_C = CN @ (v_insp_N - v_rso_N)  # approx: ignores omega×r Coriolis term

        dist_to_waypoint = float(np.linalg.norm(r_DC_C - _WAYPOINT_B))
        in_capture_sphere = dist_to_waypoint < WAYPOINT_CAPTURE_RADIUS

        return (r_DC_C, v_DC_C, dist_to_waypoint, in_capture_sphere)

    def compare_log_states(self, old_state: Any, new_state: Any) -> WaypointPhaseData:
        # Guard: RSO or first step after reset
        if old_state is None or new_state is None:
            return WaypointPhaseData()

        old_r, old_v, old_d, old_in = old_state
        new_r, new_v, new_d, new_in = new_state

        # Phase latches: once inside capture sphere, stays Phase 1
        old_phase = 1 if old_in else 0
        new_phase = 1 if (old_in or new_in) else 0
        is_transition = (old_phase == 0) and (new_phase == 1)

        return WaypointPhaseData(
            r_DC_C=new_r,
            v_DC_C=new_v,
            phase=new_phase,
            dist_to_waypoint=new_d,
            is_phase_transition=is_transition,
        )


# ---------------------------------------------------------------------------
# Rewarder
# ---------------------------------------------------------------------------

class WaypointPhaseReward(GlobalReward):
    """
    Replaces RelativeRangeLogReward with a two-phase body-frame reward.

    Reward function is log-MSE in both phases, matching the existing style of
    RelativeRangeLogReward.  The velocity penalty is proximity-scaled in Phase 0
    and always-on in Phase 1.
    """
    data_store_type = WaypointPhaseDataStore

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        rewards: dict[str, float] = {}

        for sat_id, data in new_data_dict.items():
            if "RSO" in sat_id or not isinstance(data, WaypointPhaseData):
                rewards[sat_id] = 0.0
                continue

            r = data.r_DC_C
            v = data.v_DC_C

            if data.phase == 0:
                # ----------------------------------------------------------
                # Phase 0: pull toward the 30 m body-fixed standoff waypoint
                # ----------------------------------------------------------
                pos_error = r - _WAYPOINT_B
                dist = data.dist_to_waypoint

                # Position: normalise each component by the max approach range
                pos_norm = pos_error / MAX_REL_POS  # shape (3,)

                # Velocity: braking curve — zero when far, ramps to full at waypoint
                prox = float(np.clip(1.0 - dist / vel_onset_range, 0.0, 1.0))
                vel_norm = (v / VEL_NORM) * vel_weight_phase0 * prox  # shape (3,)

            else:
                # ----------------------------------------------------------
                # Phase 1: pull toward docking port (RSO body-frame origin)
                # ----------------------------------------------------------
                # Normalise position by STANDOFF_DISTANCE so the agent sees a
                # [0, 1] gradient across the full Phase 1 range.
                pos_norm = r / STANDOFF_DISTANCE  # shape (3,)

                # Velocity: always-on, stronger weight
                vel_norm = (v / VEL_NORM) * vel_weight_phase1  # shape (3,)

            state = np.concatenate([pos_norm, vel_norm])
            mse = float(np.mean(state ** 2))
            dense_reward = waypoint_pos_weight * np.log(mse + 1e-8)

            # One-time sparse bonus on the step the agent first enters Phase 1
            sparse_bonus = waypoint_sparse_reward if data.is_phase_transition else 0.0

            rewards[sat_id] = float(dense_reward + sparse_bonus)

            if data.is_phase_transition:
                logger.info(
                    f"WAYPOINT CAPTURED by {sat_id}: "
                    f"d_wp={data.dist_to_waypoint:.1f}m, "
                    f"|v|={np.linalg.norm(v):.3f}m/s, "
                    f"sparse_bonus={sparse_bonus}"
                )

        return {k: v for k, v in rewards.items() if "RSO" not in k}
