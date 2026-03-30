import logging
from typing import Optional, Union, Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from Basilisk.utilities.RigidBodyKinematics import MRP2C

from resources import approach_corridor_angle_deg, docking_phase_range_threshold, approach_corridor_weight

logger = logging.getLogger(__name__)

# 1. The Container
class DockingCorridorData(Data):
    """Container for the relative position in the Chief's Body Frame (r_DC_C)."""
    def __init__(self, state_vector: Optional[np.ndarray] = None):
        self.state_vector = state_vector if state_vector is not None else np.zeros(3)

    def __add__(self, other: Data) -> "DockingCorridorData":
        if isinstance(other, DockingCorridorData):
            return DockingCorridorData(state_vector=other.state_vector)
        return self

    def __repr__(self) -> str:
        dist = np.linalg.norm(self.state_vector)
        return f"DockingCorridorData(norm_pos={dist:.2f})"


# 2. The DataStore (Handles Basilisk interactions and Pylance ignores)
class DockingCorridorDataStore(DataStore):
    data_type = DockingCorridorData

    def get_log_state(self) -> np.ndarray:
        """Calculate relative position rotated into the RSO (Chief) Body Frame."""
        # Get Inspector (Deputy) inertial position
        dep_dyn = self.satellite.simulator.satellites[1].dynamics 
        r_dep_N = np.array(dep_dyn.r_BN_N)  # type: ignore
        
        # Get RSO (Chief) inertial position and attitude
        chf_dyn = self.satellite.simulator.satellites[0].dynamics
        r_chf_N = np.array(chf_dyn.r_BN_N)  # type: ignore
        sigma_BN_chf = np.array(chf_dyn.sigma_BN)  # type: ignore
        
        # 1. Relative position in Inertial Frame
        r_DC_N = r_dep_N - r_chf_N
        
        # 2. Convert Chief MRP to Direction Cosine Matrix (Inertial -> Body)
        CN = MRP2C(sigma_BN_chf)
        
        # 3. Rotate relative position into Chief Body Frame
        r_DC_C = CN.dot(r_DC_N)
        
        return r_DC_C

    def compare_log_states(self, old_state: np.ndarray, new_state: np.ndarray) -> DockingCorridorData:
        return DockingCorridorData(state_vector=new_state)


# 3. The Rewarder (Pure math, including cutoff_range and dist_scale)
class DockingCorridorReward(GlobalReward):
    """
    Penalizes the Inspector for being outside a specified approach cone 
    relative to the Chief's boresight, active only within a cutoff range.
    """
    data_store_type = DockingCorridorDataStore

    def __init__(
        self, 
        weight: float = approach_corridor_weight, 
        docking_port_boresight: np.ndarray = np.array([0.0, 0.0, 1.0]), 
        corridor_angle_deg: float = approach_corridor_angle_deg,
        cutoff_range: float = docking_phase_range_threshold
    ):
        super().__init__()
        self.weight = weight
        self.docking_port_boresight = np.array(docking_port_boresight) / np.linalg.norm(docking_port_boresight)
        self.cos_limit = np.cos(np.radians(corridor_angle_deg))
        self.cutoff_range = cutoff_range

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        reward = {}
        for sat_id, data in new_data_dict.items():
            # Apply to any satellite that is NOT the RSO
            if "RSO" not in sat_id and isinstance(data, DockingCorridorData):
                r_DC_C = data.state_vector
                dist = np.linalg.norm(r_DC_C)
                
                # 3. RANGE SCHEDULING
                if 1e-6 < dist < self.cutoff_range:
                    r_DC_C_hat = r_DC_C / dist
                    cos_theta = np.dot(r_DC_C_hat, self.docking_port_boresight)
                    
                    if cos_theta < self.cos_limit:
                        # Outside the corridor: apply penalty scaled by proximity
                        dist_scale = (self.cutoff_range - dist) / self.cutoff_range
                        penalty = -abs(self.weight) * dist_scale * (self.cos_limit - cos_theta)
                        reward[sat_id] = float(penalty)
                    else:
                        # Inside the corridor: positive reward for alignment!
                        # Scale from 0.0 (at the edge of the cone) to 1.0 (perfectly centered)
                        alignment_score = (cos_theta - self.cos_limit) / (1.0 - self.cos_limit)
                        
                        # Apply the weight to the score
                        positive_reward = abs(self.weight) * alignment_score
                        reward[sat_id] = float(positive_reward)
                else:
                    # Outside cutoff range (or perfectly at origin)
                    reward[sat_id] = 0.0
            else:
                reward[sat_id] = 0.0

        # Debug logging
        # print(f"DockingCorridorReward: {reward}")
                
        # 6. RETURN FILTER
        # Filter out the RSO so the environment only steps the learning agents
        return {k: v for k, v in reward.items() if "RSO" not in k}