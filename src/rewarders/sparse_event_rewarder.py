"""Terminal/event-based sparse rewards: docking success, collision, and max-range violation."""

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils.orbital import cd2hill
from Basilisk.utilities.RigidBodyKinematics import MRP2C

from resources import (
    docking_reward,
    conjunction_penalty,
    max_range_penalty,
    misalignment_discount_factor,
    approach_corridor_angle_deg,
    docking_port_boresight,
)

logger = logging.getLogger(__name__)

_BORESIGHT = np.asarray(docking_port_boresight, dtype=float)


class SparseEventData(Data):
    """Carries the transition-detected events for one RL step."""
    def __init__(
        self,
        conjunction_angle_deg: Optional[float] = None,
        range_violated: bool = False,
    ):
        self.conjunction_angle_deg = conjunction_angle_deg
        self.range_violated = range_violated

    def __add__(self, other: Data) -> "SparseEventData":
        if isinstance(other, SparseEventData):
            return SparseEventData(other.conjunction_angle_deg, other.range_violated)
        return self

    def __repr__(self) -> str:
        return f"SparseEventData(angle={self.conjunction_angle_deg}, range_viol={self.range_violated})"


class SparseEventDataStore(DataStore):
    data_type = SparseEventData

    def get_log_state(self) -> Tuple[Optional[float], bool]:
        """Returns (conjunction_angle_deg | None, range_exceeded)."""
        if "RSO" in self.satellite.name:
            return (None, False)

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

        return (conjunction_angle, range_exceeded)

    def compare_log_states(self, old_state: Any, new_state: Any) -> SparseEventData:
        """Fire events only on the step they first occur (transition detection)."""
        old_angle, old_range = old_state
        new_angle, new_range = new_state

        # Conjunction fires once: when it transitions from None → angle
        fire_conjunction = (new_angle is not None) and (old_angle is None)
        # Range penalty fires once: on first step into violation
        fire_range = new_range and not old_range

        return SparseEventData(
            conjunction_angle_deg=new_angle if fire_conjunction else None,
            range_violated=fire_range,
        )


class SparseEventReward(GlobalReward):
    """
    Computes terminal/event rewards inside the BSK-RL GlobalReward framework.

    Events handled:
      - Docking (conjunction within approach corridor): +docking_reward x alignment_multiplier
      - Collision (conjunction outside corridor):        +conjunction_penalty
      - Max-range violation:                             +max_range_penalty
    """
    data_store_type = SparseEventDataStore

    def __init__(self, corridor_angle_deg: float = approach_corridor_angle_deg):
        super().__init__()
        self.corridor_angle_deg = corridor_angle_deg

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for sat_id, data in new_data_dict.items():
            if "RSO" in sat_id or not isinstance(data, SparseEventData):
                rewards[sat_id] = 0.0
                continue

            r = 0.0
            if data.conjunction_angle_deg is not None:
                angle = data.conjunction_angle_deg
                if angle <= self.corridor_angle_deg:
                    max_penalty_fraction = 1.0 - misalignment_discount_factor
                    alignment_mult = 1.0 - max_penalty_fraction * (angle / max(self.corridor_angle_deg, 1e-6))
                    r += docking_reward * alignment_mult
                    logger.info(f"DOCKING: angle={angle:.1f}° reward={docking_reward * alignment_mult:.2f}")
                else:
                    r += conjunction_penalty
                    logger.info(f"COLLISION: angle={angle:.1f}° penalty={conjunction_penalty}")

            if data.range_violated:
                r += max_range_penalty
                logger.info(f"MAX RANGE VIOLATION: penalty={max_range_penalty}")

            rewards[sat_id] = r

        return {k: v for k, v in rewards.items() if "RSO" not in k}
