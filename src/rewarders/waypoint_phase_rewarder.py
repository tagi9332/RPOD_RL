"""
Body-frame guidance rewarder for docking with an RSO at arbitrary attitude.

Covers all three mission phases:

Phase 1  —  long-range rendezvous (initial location → 500 m).
             Dense:  log-MSE of position error to the 30 m body-fixed standoff waypoint.
             Vel:    none — braking ramp is zero beyond vel_onset_range from the waypoint.

Phase 2  —  close-range waypoint navigation (500 m → 30 m).
             Dense:  same log-MSE of position error to the standoff waypoint.
             Vel:    proximity-scaled braking penalty (zero at vel_onset_range, full at waypoint).
             Sparse: one-time bonus on first crossing PHASE_TRANSITION_RANGE (entry to Phase 3).

Phase 3  —  terminal ingress and docking (30 m → 5 m).
             Dense:  log-MSE of position error to docking port (body-frame origin).
             Vel:    always-on, stronger penalty to enforce controlled ingress.

Phases 1 and 2 share the same reward branch (internal flag phase == 0) and differ only
in whether the braking ramp is active. The Phase 2 → 3 transition (internal flag
phase == 1) is based on relative range only: the Inspector enters Phase 3 the first
time ||r_DC_C|| < PHASE_TRANSITION_RANGE, regardless of approach direction.
Phase 3 is latched: once entered it holds for the rest of the episode, preventing
reward oscillation if the Inspector briefly moves back out.
"""

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from Basilisk.utilities.RigidBodyKinematics import MRP2C


class WaypointGate:
    """Shared state tracking the internal phase flag per satellite.

    WaypointPhaseReward writes the flag on every step (0 = Phases 1–2,
    1 = Phase 3); SparseEventReward reads it to gate the docking bonus on
    Phase 3 entry.
    """
    def __init__(self) -> None:
        self._phase: dict[str, int] = {}

    def update(self, sat_id: str, phase: int) -> None:
        self._phase[sat_id] = phase

    def waypoint_captured(self, sat_id: str) -> bool:
        return self._phase.get(sat_id, 0) >= 1

from resources import (
    waypoint_pos_weight,
    waypoint_sparse_reward,
    vel_weight_phase0,
    vel_onset_range,
    vel_weight_phase1,
    STANDOFF_DISTANCE,
    WAYPOINT_CAPTURE_RADIUS,
    PHASE_TRANSITION_RANGE,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ever_captured: bool = False

    def get_log_state(self) -> Optional[Tuple]:
        """Returns (r_DC_C, v_DC_C, dist_to_waypoint, in_phase3_range) or None for RSO."""
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
        in_phase3_range = float(np.linalg.norm(r_DC_C)) < PHASE_TRANSITION_RANGE

        return (r_DC_C, v_DC_C, dist_to_waypoint, in_phase3_range)

    def compare_log_states(self, old_state: Any, new_state: Any) -> WaypointPhaseData:
        # Guard: RSO or first step after reset
        if old_state is None or new_state is None:
            return WaypointPhaseData()

        _old_r, _old_v, _old_d, _old_in = old_state
        new_r, new_v, new_d, new_in = new_state

        # Fires exactly once per episode — when the agent first crosses PHASE_TRANSITION_RANGE
        # (Phase 2 → 3). Internal flag: 0 = Phases 1–2, 1 = Phase 3.
        is_transition = new_in and not self._ever_captured
        if new_in:
            self._ever_captured = True

        new_phase = 1 if self._ever_captured else 0

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
    Replaces RelativeRangeLogReward with a body-frame reward over all three phases.

    Reward function is log-MSE in every phase, matching the existing style of
    RelativeRangeLogReward.  The velocity penalty is off in Phase 1, proximity-scaled
    in Phase 2, and always-on in Phase 3.
    """
    data_store_type = WaypointPhaseDataStore

    def __init__(self, gate: Optional[WaypointGate] = None) -> None:
        super().__init__()
        self.gate = gate

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        rewards: dict[str, float] = {}

        for sat_id, data in new_data_dict.items():
            if "RSO" in sat_id or not isinstance(data, WaypointPhaseData):
                rewards[sat_id] = 0.0
                continue

            r = data.r_DC_C
            v = data.v_DC_C

            if self.gate is not None:
                self.gate.update(sat_id, data.phase)

            if data.phase == 0:
                # ----------------------------------------------------------
                # Phases 1–2: pull toward the 30 m body-fixed standoff waypoint
                # ----------------------------------------------------------
                pos_error = r - _WAYPOINT_B
                dist = data.dist_to_waypoint

                # Position: normalise each component by the max approach range
                pos_norm = pos_error / MAX_REL_POS  # shape (3,)

                # Velocity: braking curve — zero in Phase 1, ramps to full at waypoint in Phase 2
                prox = float(np.clip(1.0 - dist / vel_onset_range, 0.0, 1.0))
                vel_norm = (v / VEL_NORM) * vel_weight_phase0 * prox  # shape (3,)

            else:
                # ----------------------------------------------------------
                # Phase 3: pull toward docking port (RSO body-frame origin)
                # ----------------------------------------------------------
                # Normalise position by PHASE_TRANSITION_RANGE so the agent sees a
                # [0, 1] gradient across the full Phase 3 range.
                pos_norm = r / PHASE_TRANSITION_RANGE  # shape (3,)

                # Velocity: always-on, stronger weight
                vel_norm = (v / VEL_NORM) * vel_weight_phase1  # shape (3,)

            state = np.concatenate([pos_norm, vel_norm])
            mse = float(np.mean(state ** 2))
            dense_reward = -waypoint_pos_weight * np.log(mse + 1e-8)

            # One-time sparse bonus on the step the agent first enters Phase 3,
            # scaled by how close the entry point is to the docking port boresight axis.
            # Transverse offset 0 → full reward; offset ≥ WAYPOINT_CAPTURE_RADIUS → 0
            # reward (≈19.5° off-axis at a 30 m transition range).
            if data.is_phase_transition:
                error = data.r_DC_C - _WAYPOINT_B
                transverse = error - np.dot(error, _BORESIGHT) * _BORESIGHT
                transverse_dist = float(np.linalg.norm(transverse))
                alignment_score = max(0.0, 1.0 - transverse_dist / WAYPOINT_CAPTURE_RADIUS)
                sparse_bonus = waypoint_sparse_reward * alignment_score
            else:
                sparse_bonus = 0.0

            rewards[sat_id] = float(dense_reward + sparse_bonus)

            if data.is_phase_transition:
                logger.info(
                    f"PHASE 3 (TERMINAL INGRESS) ENTERED by {sat_id}: "
                    f"range={np.linalg.norm(r):.1f}m, "
                    f"d_wp={data.dist_to_waypoint:.1f}m, "
                    f"|v|={np.linalg.norm(v):.3f}m/s, "
                    f"transverse_offset={transverse_dist:.1f}m, "
                    f"alignment={alignment_score:.3f}, "
                    f"sparse_bonus={sparse_bonus:.2f}"
                )

        return {k: v for k, v in rewards.items() if "RSO" not in k}
