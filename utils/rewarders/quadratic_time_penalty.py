"""Data system for recording simulation time and applying time-based penalties."""

import logging
from typing import Optional, Union, Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward

logger = logging.getLogger(__name__)

# 1. The Container
class TimeData(Data):
    """Container for the current simulation time."""
    def __init__(self, current_time: Optional[float] = None):
        self.current_time = current_time if current_time is not None else 0.0

    def __add__(self, other: Data) -> "TimeData":
        """In this case, addition just takes the most recent time."""
        if isinstance(other, TimeData):
            return TimeData(current_time=other.current_time)
        return self

    def __repr__(self) -> str:
        return f"TimeData(sim_time={self.current_time:.1f}s)"


# 2. The DataStore
class TimeDataStore(DataStore):
    data_type = TimeData

    def get_log_state(self) -> float:
        """
        Pull the current simulation time from the Basilisk simulator.
        """
        # Access the simulator's internal clock
        current_time = self.satellite.simulator.sim_time 
        return float(current_time)

    def compare_log_states(self, old_state: float, new_state: float) -> TimeData:
        """Pass the new time into the Data container."""
        return TimeData(current_time=new_state)


# 3. The Rewarder
class QuadraticTimePenalty(GlobalReward):
    """
    Applies a time penalty that scales quadratically to prevent loitering.
    Penalty starts near zero and scales to `max_penalty` at `max_sim_time`.
    """
    data_store_type = TimeDataStore

    def __init__(
        self,
        max_penalty: float = -0.005,
        max_sim_time: float = 3000.0,
        power: float = 2.0
    ):
        """
        Args:
            max_penalty: The maximum negative reward applied at the end of the sim.
            max_sim_time: The maximum length of the episode in seconds.
            power: The exponent for scaling (2.0 = quadratic, 3.0 = cubic).
        """
        super().__init__()
        # Ensure max_penalty is negative so it always punishes
        self.max_penalty = -abs(max_penalty) 
        self.max_sim_time = float(max_sim_time)
        self.power = float(power)

    def calculate_reward(
        self, new_data_dict: Mapping[str, Data]
    ) -> dict[str, float]:
        """Calculate the polynomial time penalty."""
        reward = {}
        for sat_id, data in new_data_dict.items():
            if isinstance(data, TimeData):
                current_time = data.current_time
                
                # Normalize time fraction [0.0 to 1.0]
                time_fraction = min(current_time / self.max_sim_time, 1.0)
                
                # Calculate the polynomial penalty: P(t) = -W_max * (t / T_max)^k
                # Note: self.max_penalty is already negative
                step_penalty = self.max_penalty * (time_fraction ** self.power)
                
                reward[sat_id] = float(step_penalty)

                # Debug prints
                # logger.debug(f"Time Penalty for {sat_id}: time={current_time}s, fraction={time_fraction:.3f}, penalty={step_penalty:.5f}")
                # print(f"Time Penalty for {sat_id}: time={current_time}s, fraction={time_fraction:.3f}, penalty={step_penalty:.5f}")
                
            else:
                reward[sat_id] = 0.0
                
        # Filter to apply only to the Inspector
        return {k: v for k, v in reward.items() if "Inspector" in k}