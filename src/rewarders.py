"""Data system for recording relative range and velocity rewards."""

import logging
from typing import Optional, Union, Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils.orbital import cd2hill

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
        alpha: float = -0.1,
        delta_x_max: Union[list, np.ndarray, None] = None,
    ):
        """
        Logarithmic reward for proximity operations.
        
        Args:
            alpha: Scaling factor for the log reward.
            delta_x_max: Normalization constants [x, y, z, vx, vy, vz].
        """
        super().__init__()
        self.alpha = alpha
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
                state = data.state_vector
                
                if np.any(state):
                    # Normalize the state
                    normalized = state / self.delta_x_max

                    # Calculate MSE and log reward
                    mse = np.mean(normalized**2)
                    reward[sat_id] = float(self.alpha * np.log(mse + 1e-8))

                    # Debug prints
                    logger.debug(f"Reward Calculation for {sat_id}: state={state}, normalized={normalized}, mse={mse}, reward={reward[sat_id]}")
                else:
                    reward[sat_id] = 0.0
            else:
                reward[sat_id] = 0.0
    
        return {k: v for k, v in reward.items() if "Inspector" in k}

class RelativeCosineReward(GlobalReward):
    """Rewards the agent for moving directly toward the RSO."""
    data_store_type = RelativeRangeDataStore

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        reward = {}
        for sat_id, data in new_data_dict.items():
            if "Inspector" in sat_id and isinstance(data, RelativeRangeData):
                state = data.state_vector
                pos = state[:3]      # Relative Position (Inspector to RSO)
                vel = state[3:]      # Relative Velocity
                
                # Cosine Similarity = (A . B) / (||A|| * ||B||)
                pos_norm = np.linalg.norm(pos)
                vel_norm = np.linalg.norm(vel)
                
                if pos_norm > 1e-3 and vel_norm > 1e-4:
                    # -1 means moving directly toward, +1 means moving away
                    # We negate it so moving TOWARD gives +1.0
                    cos_theta = -np.dot(pos, vel) / (pos_norm * vel_norm)
                    
                    # Debug prints
                    logger.debug(f"Cosine Reward Calculation for {sat_id}: pos={pos}, vel={vel}, cos_theta={cos_theta}")
                    
                    # We only reward if moving toward (cos_theta > 0)
                    # This prevents rewarding "drifting away"
                    reward[sat_id] = float(self.weight * max(0, cos_theta))
                else:
                    reward[sat_id] = 0.0
        return reward
    