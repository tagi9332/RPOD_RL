"""Step-size-independent time penalty via the integral of a polynomial rate."""

import logging
from typing import Mapping

import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward

from resources import SIM_TIME

logger = logging.getLogger(__name__)


class TimeData(Data):
    """Stores the sim-time window [old_time, new_time] for one RL step."""
    def __init__(self, old_time: float = 0.0, new_time: float = 0.0):
        self.old_time = old_time
        self.new_time = new_time

    def __add__(self, other: Data) -> "TimeData":
        if isinstance(other, TimeData):
            return TimeData(old_time=other.old_time, new_time=other.new_time)
        return self

    def __repr__(self) -> str:
        return f"TimeData([{self.old_time:.1f}s → {self.new_time:.1f}s])"


class TimeDataStore(DataStore):
    data_type = TimeData

    def get_log_state(self) -> float:
        return float(self.satellite.simulator.sim_time)

    def compare_log_states(self, old_state: float, new_state: float) -> TimeData:
        return TimeData(old_time=old_state, new_time=new_state)


class QuadraticTimePenalty(GlobalReward):
    """
    Step-size-independent time penalty.

    The penalty rate is proportional to (t/T)^power, so it is nearly zero
    early and steep near the episode end.  The per-step contribution is the
    exact integral of that rate over the drift window [t_prev, t_now]:

        step_penalty = -max_episode_penalty
                       * (t_now^(n+1) - t_prev^(n+1)) / T^(n+1)

    Summed over a complete episode this equals exactly -max_episode_penalty,
    regardless of drift step sizes.

    Args:
        max_episode_penalty: Total penalty accumulated over a full episode [+ve scalar].
        max_sim_time: Episode length in seconds.
        power: Rate exponent n (2 = quadratic rate, 3 = cubic rate).
    """
    data_store_type = TimeDataStore

    def __init__(
        self,
        max_episode_penalty: float = 65.0,
        max_sim_time: float = SIM_TIME,
        power: float = 2.0,
    ):
        super().__init__()
        self.max_episode_penalty = abs(max_episode_penalty)
        self.max_sim_time = float(max_sim_time)
        self.power = float(power)

    def calculate_reward(self, new_data_dict: Mapping[str, Data]) -> dict[str, float]:
        reward = {}
        T = self.max_sim_time
        n = self.power
        for sat_id, data in new_data_dict.items():
            if isinstance(data, TimeData):
                t0, t1 = data.old_time, data.new_time
                if t1 > t0 and T > 0:
                    # Integral of -P * (t/T)^n from t0 to t1
                    step_penalty = -self.max_episode_penalty * (t1**(n+1) - t0**(n+1)) / (T**(n+1))
                else:
                    step_penalty = 0.0
                reward[sat_id] = float(step_penalty)
            else:
                reward[sat_id] = 0.0
        return {k: v for k, v in reward.items() if "Inspector" in k}
