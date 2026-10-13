import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeDvLoggerCallback(BaseCallback):
    """
    Logs mean per-episode dv usage as rollout/ep_dv_mean.

    Requires the env to be wrapped with Monitor(info_keywords=("ep_dv_used",)) so the
    final-step dv lands in model.ep_info_buffer — the same 100-episode window SB3 uses
    for rollout/ep_rew_mean, so the two are directly comparable.
    """
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        dvs = [ep["ep_dv_used"] for ep in self.model.ep_info_buffer if "ep_dv_used" in ep]
        if dvs:
            self.logger.record("rollout/ep_dv_mean", float(np.mean(dvs)))
            self.logger.record("rollout/ep_dv_max", float(np.max(dvs)))
