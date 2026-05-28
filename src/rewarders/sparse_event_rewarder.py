"""Terminal/event-based sparse rewards: docking success, collision, and max-range violation."""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from src.rewarders.waypoint_phase_rewarder import WaypointGate

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils.orbital import cd2hill
from Basilisk.utilities.RigidBodyKinematics import MRP2C

from resources import (
    docking_reward,
    max_range_penalty,
    approach_corridor_angle_deg,
    docking_port_boresight,
    fuel_bonus_max,
    fuel_bonus_baseline,
    DV_AVAILABLE_INIT,
)

logger = logging.getLogger(__name__)

_BORESIGHT = np.asarray(docking_port_boresight, dtype=float)


class SparseEventData(Data):
    """Carries the transition-detected events for one RL step."""
    def __init__(
        self,
        conjunction_angle_deg: Optional[float] = None,
        range_violated: bool = False,
        dv_remaining: float = 0.0,
    ):
        self.conjunction_angle_deg = conjunction_angle_deg
        self.range_violated = range_violated
        self.dv_remaining = dv_remaining

    def __add__(self, other: Data) -> "SparseEventData":
        if isinstance(other, SparseEventData):
            return SparseEventData(other.conjunction_angle_deg, other.range_violated, other.dv_remaining)
        return self

    def __repr__(self) -> str:
        return f"SparseEventData(angle={self.conjunction_angle_deg}, range_viol={self.range_violated}, dv_rem={self.dv_remaining:.1f})"


class SparseEventDataStore(DataStore):
    data_type = SparseEventData

    def get_log_state(self) -> Tuple[Optional[float], bool, float]:
        """Returns (conjunction_angle_deg | None, range_exceeded, dv_remaining)."""
        if "RSO" in self.satellite.name:
            return (None, False, 0.0)

        insp_dyn = self.satellite.dynamics
        rso_dyn = self.satellite.simulator.satellites[0].dynamics

        r_insp_N = np.array(insp_dyn.r_BN_N)
        r_rso_N = np.array(rso_dyn.r_BN_N)
        v_insp_N = np.array(insp_dyn.v_BN_N)
        v_rso_N = np.array(rso_dyn.v_BN_N)

        # Conjunction angle (None if no conjunction this step)
        conjunction_angle: Optional[float] = None
        if getattr(insp_dyn, "conjunctions", None):
            r_rel = r_insp_N - r_rso_N
            dist = np.linalg.norm(r_rel)
            if dist > 1e-6:
                CN = MRP2C(np.array(rso_dyn.sigma_BN))
                r_rel_B_hat = CN @ (r_rel / dist)
                cos_theta = float(np.clip(np.dot(r_rel_B_hat, _BORESIGHT), -1.0, 1.0))
                conjunction_angle = float(np.degrees(np.arccos(cos_theta)))

        # Max-range check
        rho, _ = cd2hill(r_rso_N, v_rso_N, r_insp_N, v_insp_N)
        max_range = self.satellite.sat_args.get("max_range_radius", 10_000)
        range_exceeded = bool(np.linalg.norm(rho) > max_range)

        dv_remaining = float(getattr(self.satellite.fsw, "dv_available", 0.0))

        return (conjunction_angle, range_exceeded, dv_remaining)

    def compare_log_states(self, old_state: Any, new_state: Any) -> SparseEventData:
        """Fire events only on the step they first occur (transition detection)."""
        old_angle, old_range, _old_dv = old_state
        new_angle, new_range, new_dv = new_state

        # Conjunction fires once: when it transitions from None → angle
        fire_conjunction = (new_angle is not None) and (old_angle is None)
        # Range penalty fires once: on first step into violation
        fire_range = new_range and not old_range

        return SparseEventData(
            conjunction_angle_deg=new_angle if fire_conjunction else None,
            range_violated=fire_range,
            dv_remaining=new_dv,
        )


class SparseEventReward(GlobalReward):
    """
    Computes terminal/event rewards inside the BSK-RL GlobalReward framework.

    Events handled:
      - Conjunction: +docking_reward * (1 - angle/180), max at 0° boresight alignment, 0 at 180°
      - Max-range violation: +max_range_penalty
    """
    data_store_type = SparseEventDataStore

    def __init__(
        self,
        corridor_angle_deg: float = approach_corridor_angle_deg,
        waypoint_gate: "Optional[WaypointGate]" = None,
    ):
        super().__init__()
        self.corridor_angle_deg = corridor_angle_deg
        self.waypoint_gate = waypoint_gate

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for sat_id, data in new_data_dict.items():
            if "RSO" in sat_id or not isinstance(data, SparseEventData):
                rewards[sat_id] = 0.0
                continue

            r = 0.0
            if data.conjunction_angle_deg is not None:
                angle = data.conjunction_angle_deg
                if self.waypoint_gate is not None and not self.waypoint_gate.waypoint_captured(sat_id):
                    logger.info(f"DOCKING BLOCKED (waypoint not captured) for {sat_id}: angle={angle:.1f}°")
                else:
                    alignment_mult = max(0.0, 1.0 - angle / 180.0)
                    scaled_reward = docking_reward * alignment_mult
                    r += scaled_reward

                    # Fuel efficiency bonus — only on true docking (within corridor)
                    fuel_bonus = 0.0
                    if angle <= self.corridor_angle_deg and DV_AVAILABLE_INIT > fuel_bonus_baseline:
                        fuel_frac = max(0.0, data.dv_remaining - fuel_bonus_baseline) / (DV_AVAILABLE_INIT - fuel_bonus_baseline)
                        fuel_bonus = fuel_bonus_max * fuel_frac
                        r += fuel_bonus

                    label = "DOCKING" if angle <= self.corridor_angle_deg else "CONJUNCTION"
                    logger.info(f"{label}: angle={angle:.1f}° reward={scaled_reward:.2f} fuel_bonus={fuel_bonus:.2f} dv_rem={data.dv_remaining:.1f}m/s")

            if data.range_violated:
                r += max_range_penalty
                logger.info(f"MAX RANGE VIOLATION: penalty={max_range_penalty}")

            rewards[sat_id] = r

        return {k: v for k, v in rewards.items() if "RSO" not in k}
