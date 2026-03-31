import logging
from typing import Optional, Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward

logger = logging.getLogger(__name__)

# 1. The Container
class IlluminationData(Data):
    """Container for the Inspector-to-RSO and Inspector-to-Sun relative vectors."""
    def __init__(self, state_vector: Optional[np.ndarray] = None):
        self.state_vector = state_vector if state_vector is not None else np.zeros(6)

    def __add__(self, other: Data) -> "IlluminationData":
        if isinstance(other, IlluminationData):
            return IlluminationData(state_vector=other.state_vector)
        return self

    def __repr__(self) -> str:
        return f"IlluminationData()"


# 2. The DataStore
class IlluminationDataStore(DataStore):
    data_type = IlluminationData

    def get_log_state(self) -> np.ndarray:
        """Extract inertial vectors to calculate phase angles."""
        dep_dyn = self.satellite.simulator.satellites[1].dynamics 
        chf_dyn = self.satellite.simulator.satellites[0].dynamics
        
        r_dep_N = np.array(dep_dyn.r_BN_N)  # type: ignore
        r_chf_N = np.array(chf_dyn.r_BN_N)  # type: ignore
        
        world = self.satellite.simulator.world
        sun_msg = world.gravFactory.spiceObject.planetStateOutMsgs[world.sun_index].read()
        r_sun_N = np.array(sun_msg.PositionVector)
        
        r_Insp_to_RSO_N = r_chf_N - r_dep_N
        r_Insp_to_Sun_N = r_sun_N - r_dep_N
        
        return np.concatenate([r_Insp_to_RSO_N, r_Insp_to_Sun_N])

    def compare_log_states(self, old_state: np.ndarray, new_state: np.ndarray) -> IlluminationData:
        return IlluminationData(state_vector=new_state)


# 3. The Rewarder
class IlluminationReward(GlobalReward):
    """
    Rewards the Inspector for approaching the RSO from the sunlit side within a specific cone.
    Active only when the Inspector is further than the cutoff_range.
    """
    data_store_type = IlluminationDataStore

    def __init__(
        self, 
        weight: float = 1.0, 
        cutoff_range: float = 100.0,
        cone_angle_deg: float = 60.0
    ):
        super().__init__()
        self.weight = abs(weight)
        self.cutoff_range = cutoff_range
        self.cos_limit = np.cos(np.radians(cone_angle_deg))

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        reward = {}
        for sat_id, data in new_data_dict.items():
            if "RSO" not in sat_id and isinstance(data, IlluminationData):
                state = data.state_vector
                r_Insp_to_RSO_N = state[0:3]
                r_Insp_to_Sun_N = state[3:6]
                
                dist_to_rso = np.linalg.norm(r_Insp_to_RSO_N)
                
                if dist_to_rso > self.cutoff_range:
                    dist_to_sun = np.linalg.norm(r_Insp_to_Sun_N)
                    
                    if dist_to_rso > 1e-6 and dist_to_sun > 1e-6:
                        u_cam = r_Insp_to_RSO_N / dist_to_rso
                        u_sun = r_Insp_to_Sun_N / dist_to_sun
                        
                        cos_phase_angle = np.clip(np.dot(u_cam, u_sun), -1.0, 1.0)
                        
                        if cos_phase_angle >= self.cos_limit:
                            # Inside the acceptable lighting cone: give flat positive reward
                            reward[sat_id] = float(self.weight)
                        else:
                            # Outside the lighting cone: penalize based on how far out they are
                            # This provides a gradient to guide the agent back into the light
                            penalty = -self.weight * (self.cos_limit - cos_phase_angle)
                            reward[sat_id] = float(penalty)
                    else:
                        reward[sat_id] = 0.0
                else:
                    # Inside the terminal cutoff range, lighting is no longer heavily enforced
                    reward[sat_id] = 0.0
            else:
                reward[sat_id] = 0.0

        return {k: v for k, v in reward.items() if "RSO" not in k}