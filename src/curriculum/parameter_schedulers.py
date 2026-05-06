"""
Curriculum learning callbacks for PPO training.

Each scheduler linearly interpolates its parameter from an initial (easy) value
to a final (hard) value over the course of training. All schedulers set an
attribute on Sb3BksEnv via SubprocVecEnv.set_attr(); the environment applies
the change to the relevant satellite/rewarder/randomizer on the next episode reset.

Usage in docking_sim_multi_process.py:

    from src.curriculum import ConjunctionRadiusScheduler, CorridorAngleScheduler, AttitudeErrorScheduler

    callbacks = [
        ConjunctionRadiusScheduler(initial_radius=200, final_radius=10),
        CorridorAngleScheduler(initial_angle_deg=360, final_angle_deg=30),
        AttitudeErrorScheduler(initial_error_deg=90, final_error_deg=5),
    ]
"""

from stable_baselines3.common.callbacks import BaseCallback


class LinearParameterScheduler(BaseCallback):
    """
    Base class: linearly schedules one scalar from `initial` to `final` over training.

    Subclasses must implement:
        _attr_name  — str: the Sb3BksEnv attribute set by set_attr
        _log_key    — str: TensorBoard key (e.g. "curriculum/conjunction_radius")
        _kwarg_name — str: keyword arg name passed to set_scheduled_parameters
                          (used only for documentation; actual propagation happens
                           via set_attr + reset)
    """

    _attr_name: str = ""
    _log_key: str = ""

    def __init__(self, initial: float, final: float, verbose: int = 0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final

    def _current_value(self) -> float:
        total_steps = self.locals.get("total_timesteps", 1)
        progress = min(1.0, max(0.0, self.num_timesteps / total_steps))
        return self.initial + progress * (self.final - self.initial)

    def _on_step(self) -> bool:
        value = self._current_value()
        self.training_env.set_attr(self._attr_name, value)
        self.logger.record(self._log_key, value)
        return True


class ConjunctionRadiusScheduler(LinearParameterScheduler):
    """
    Linearly shrinks the Inspector's conjunction (docking success) radius.

    Example: start at 200 m (easy, large capture sphere) → 10 m (tight docking).
    """
    _attr_name = "scheduled_conjunction_radius"
    _log_key = "curriculum/conjunction_radius"

    def __init__(self, initial_radius: float = 200.0, final_radius: float = 10.0, verbose: int = 0):
        super().__init__(initial=initial_radius, final=final_radius, verbose=verbose)


class CorridorAngleScheduler(LinearParameterScheduler):
    """
    Linearly tightens the docking approach corridor cone angle.

    Example: start at 360° (omnidirectional, no penalty) → 30° (tight corridor).
    Both SparseEventReward and DockingCorridorReward are updated simultaneously
    via Sb3BksEnv.set_scheduled_parameters.
    """
    _attr_name = "scheduled_corridor_angle_deg"
    _log_key = "curriculum/corridor_angle_deg"

    def __init__(self, initial_angle_deg: float = 360.0, final_angle_deg: float = 30.0, verbose: int = 0):
        super().__init__(initial=initial_angle_deg, final=final_angle_deg, verbose=verbose)


class AttitudeErrorScheduler(LinearParameterScheduler):
    """
    Linearly tightens the maximum RSO attitude offset used during episode init.

    Only active when rso_att_type="near_velocity". Controls how far the RSO's
    attitude may deviate from velocity-pointing at the start of each episode.

    Example: start at 90° (random hemisphere) → 5° (nearly aligned).
    """
    _attr_name = "scheduled_max_error_deg"
    _log_key = "curriculum/rso_max_error_deg"

    def __init__(self, initial_error_deg: float = 90.0, final_error_deg: float = 5.0, verbose: int = 0):
        super().__init__(initial=initial_error_deg, final=final_error_deg, verbose=verbose)
