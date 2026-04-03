"""Data system for recording relative range and velocity rewards."""

import logging
from typing import Optional, Union, Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils.orbital import cd2hill

from resources import (
    final_docking_distance,
    approach_velocity_weight,
    rel_range_log_weight
)

logger = logging.getLogger(__name__)

class RelativeRangeData(Data):
    """Container for the current relative state of the satellite."""
    def __init__(self, state_vector: Optional[np.ndarray] = None):
        # store the most recent [x, y, z, vx, vy, vz]
        self.state_vector = state_vector if state_vector is not None else np.zeros(6)

    def __add__(self, other: Data) -> "RelativeRangeData":
            """In this case, addition just takes the most recent state."""
            if isinstance(other, RelativeRangeData):
                return RelativeRangeData(state_vector=other.state_vector)
            
            # if adding to a generic Data object, just return self
            return self

    def __repr__(self) -> str:
            # get norm for representation
            dist = np.linalg.norm(self.state_vector[:3])
            return f"RelativeRangeData(norm_pos={dist:.2f})"


class RelativeRangeDataStore(DataStore):
    data_type = RelativeRangeData

    def get_log_state(self) -> np.ndarray:
        """
        Pull relative position and velocity using the cd2hill conversion.
        """
        # get Inspector (Deputy) inertial state
        dep_dyn = self.satellite.simulator.satellites[1].dynamics 
        r_dep_N = np.array(dep_dyn.r_BN_N) # type: ignore
        v_dep_N = np.array(dep_dyn.v_BN_N) # type: ignore
        
        # get RSO (Chief) inertial state
        chf_dyn = self.satellite.simulator.satellites[0].dynamics
        r_chf_N = np.array(chf_dyn.r_BN_N) # type: ignore
        v_chf_N = np.array(chf_dyn.v_BN_N) # type: ignore
        
        # convert to Hill frame relative state
        rho_H, rho_dot_H = cd2hill(r_chf_N, v_chf_N, r_dep_N, v_dep_N)
        
        return np.hstack([rho_H, rho_dot_H])

    def compare_log_states(self, old_state: np.ndarray, new_state: np.ndarray) -> RelativeRangeData:
        """Pass the new state into the Data container."""
        return RelativeRangeData(state_vector=new_state)

class RelativeRangeLogReward(GlobalReward):
    data_store_type = RelativeRangeDataStore

    def __init__(
        self,
        alpha: float = rel_range_log_weight,
        delta_x_max: Union[list, np.ndarray, None] = None,
        final_docking_distance: float = final_docking_distance,
        velocity_weight: float = approach_velocity_weight,
    ):
        """
        Logarithmic reward for proximity operations.
        
        Args:
            alpha: Scaling factor for the log reward.
            delta_x_max: Normalization constants [x, y, z, vx, vy, vz].
            final_docking_distance: Distance threshold (meters) to start penalizing velocity more heavily.
            velocity_weight: Multiplier applied to velocity errors when inside the final docking distance.
        """
        super().__init__()
        self.alpha = alpha
        
        # New parameters for docking precision
        self.final_docking_distance = final_docking_distance
        self.velocity_weight = velocity_weight
        
        if delta_x_max is None:
            self.delta_x_max = np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])
        else:
            self.delta_x_max = np.array(delta_x_max)

    def calculate_reward(
        self, new_data_dict: Mapping[str, Data]
    ) -> dict[str, float]:
        """Calculate the log reward based on normalized MSE."""
        reward = {}
        for sat_id, data in new_data_dict.items():
            if isinstance(data, RelativeRangeData):
                state = data.state_vector.copy() # Use copy to prevent mutating the stored data
                
                if np.any(state):
                    # Normalize the state
                    normalized = state / self.delta_x_max

                    # Check relative distance and accentuate velocity if necessary
                    distance = np.linalg.norm(state[:3])
                    if distance < self.final_docking_distance:
                        # Multiply the velocity components (indices 3, 4, 5) by the weight
                        normalized[3:] *= self.velocity_weight

                    # Calculate MSE and log reward
                    mse = np.mean(normalized**2)
                    reward[sat_id] = float(self.alpha * np.log(mse + 1e-8))

                    # Debug prints
                    logger.debug(f"Reward Calculation for {sat_id}: distance={distance:.2f}, mse={mse}, reward={reward[sat_id]}")
                else:
                    reward[sat_id] = 0.0
            else:
                reward[sat_id] = 0.0
    
        return {k: v for k, v in reward.items() if "Inspector" in k}