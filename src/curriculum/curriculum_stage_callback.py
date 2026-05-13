"""
Callback that advances curriculum stage based on rolling episode success rate.

The stage is advanced by calling ``training_env.set_attr("curriculum_stage", n)``;
``CurriculumSb3BksEnv.reset()`` propagates the new value to the
``CurriculumSatArgRandomizer`` on the next episode initialisation.

Stage configs
-------------
0 → 1  advance when rolling success rate ≥ 70 % over 50 episodes
1 → 2  advance when rolling success rate ≥ 50 % over 100 episodes
2      terminal stage — never advanced further

Success is defined as any conjunction (``docked_state == True``) regardless of
approach corridor angle, so Stages 0/1 reward any clean terminal contact.
"""
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


STAGE_CONFIGS = [
    {
        "label":              "Stage 0 → 1  (terminal approach → correct-side capture)",
        "advance_threshold":  0.70,
        "min_episodes":       50,
    },
    {
        "label":              "Stage 1 → 2  (correct-side capture → full mission)",
        "advance_threshold":  0.50,
        "min_episodes":       100,
    },
    {
        "label":              "Stage 2  (full mission — terminal stage)",
        "advance_threshold":  None,
    },
]


class CurriculumStageCallback(BaseCallback):
    """
    Tracks rolling episode outcomes and advances ``curriculum_stage`` on all
    training envs when the configured success-rate threshold is reached.
    """

    def __init__(self, window: int = 100, initial_stage: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.window = window
        self.current_stage = initial_stage
        self._outcomes: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
        ):
            if done:
                success = bool(info.get("metrics", {}).get("docked_state", False))
                self._outcomes.append(success)

        cfg = STAGE_CONFIGS[self.current_stage]
        self.logger.record("curriculum/stage", self.current_stage)

        can_advance = (
            cfg["advance_threshold"] is not None
            and len(self._outcomes) >= cfg.get("min_episodes", 1)
            and self.current_stage < len(STAGE_CONFIGS) - 1
        )
        if can_advance:
            rate = float(np.mean(self._outcomes))
            self.logger.record("curriculum/rolling_success_rate", rate)

            if rate >= cfg["advance_threshold"]:
                self.current_stage += 1
                self._outcomes.clear()
                self.training_env.set_attr("curriculum_stage", self.current_stage)

                if self.verbose:
                    print(
                        f"\n[Curriculum] {cfg['label']} "
                        f"(success rate {rate:.1%} at step {self.num_timesteps:,})"
                    )

        return True
