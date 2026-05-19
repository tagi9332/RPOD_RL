import logging
from typing import Callable, Union, Any, Self

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.obs import ResourceRewardWeight

logger = logging.getLogger(__name__)


from resources import (
    dv_reward_weight,
    dv_constant_penalty
)

class ResourceData(Data):
    """Data for tracking a change in dV."""

    def __init__(self, resource_accumulated: float = 0.0) -> None:
        self.resource_accumulated = resource_accumulated

    def __add__(self, other: Data) -> Self: 
        if not isinstance(other, ResourceData):
            return NotImplemented
        return self.__class__(self.resource_accumulated + other.resource_accumulated)


class DeltaVDataStore(DataStore):
    """DataStore hardcoded to track satellite dv_available."""
    data_type = ResourceData

    def get_log_state(self) -> float:
        """
        Directly accesses dv_available. 
        Returns 0.0 if the attribute doesn't exist on the FSW model.
        """
        return getattr(self.satellite.fsw, "dv_available", 0.0)

    def compare_log_states(self, old_state: Any, new_state: Any) -> ResourceData:
        return ResourceData(new_state - old_state)


class DeltaVReward(GlobalReward):
    """
    Rewarder specifically for Delta V. 
    Penalizes large maneuvers using a quadratic scaling by default.
    """
    data_store_type = DeltaVDataStore

    def __init__(
        self,
        reward_weight: Union[float, Any] = dv_reward_weight, # Set your default weight here
        exponent: float = 1.0,
    ) -> None:
        super().__init__()
        self._reward_weight = reward_weight
        self.exponent = exponent
        self.data_store_kwargs = {}

    def reset_pre_sim_init(self) -> None:
        self.reward_weight = self._reward_weight() if callable(self._reward_weight) else self._reward_weight
        super().reset_pre_sim_init()

    def reset_post_sim_init(self) -> None:
        for satellite in self.scenario.satellites:
            for obs in satellite.observation_builder.observation_spec:
                if isinstance(obs, ResourceRewardWeight):
                    obs.weight_vector.append(-self.reward_weight)
        super().reset_post_sim_init()

    def calculate_reward(self, new_data_dict: dict[str, Data]) -> dict[str, float]:
        penalties = {}
        for sat_name, data in new_data_dict.items():
            if 'RSO' not in sat_name and isinstance(data, ResourceData):
                # Absolute value of the dV change
                dv_magnitude = abs(data.resource_accumulated)
                if dv_magnitude > 1e-6:
                    penalties[sat_name] = -self.reward_weight * (dv_magnitude ** self.exponent)
                    # penalties[sat_name] = dv_constant_penalty - (dv_magnitude * dv_reward_weight)
            else:
                penalties[sat_name] = 0.0
        return penalties