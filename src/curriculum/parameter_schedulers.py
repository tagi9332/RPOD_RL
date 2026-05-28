"""
Curriculum learning callbacks for PPO training.

Each scheduler linearly interpolates its parameter from an initial (easy) value
to a final (hard) value over the course of training. Each scheduler calls
Sb3BksEnv.set_scheduled_parameters() via SubprocVecEnv.env_method(), which
propagates the change to the relevant satellite/rewarder/randomizer immediately.

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
        _log_key    — str: TensorBoard key (e.g. "curriculum/conjunction_radius")
        _kwarg_name — str: keyword arg name passed to Sb3BksEnv.set_scheduled_parameters

    Uses env_method (not set_attr) so the call propagates through gymnasium wrapper
    layers via __getattr__ delegation, reaching Sb3BksEnv regardless of wrapper order.
    """

    _log_key: str = ""
    _kwarg_name: str = ""

    def __init__(self, initial: float, final: float, eval_env=None, verbose: int = 0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final
        self.eval_env = eval_env

    def _current_value(self) -> float:
        total_steps = self.locals.get("total_timesteps", 1)
        progress = min(1.0, max(0.0, self.num_timesteps / total_steps))
        return self.initial + progress * (self.final - self.initial)

    def _on_step(self) -> bool:
        value = self._current_value()
        self.training_env.env_method("set_scheduled_parameters", **{self._kwarg_name: value})
        if self.eval_env is not None:
            self.eval_env.env_method("set_scheduled_parameters", **{self._kwarg_name: value})
        self.logger.record(self._log_key, value)
        return True


class ConjunctionRadiusScheduler(LinearParameterScheduler):
    """
    Linearly shrinks the Inspector's conjunction (docking success) radius.

    Example: start at 200 m (easy, large capture sphere) → 10 m (tight docking).
    """
    _log_key = "curriculum/conjunction_radius"
    _kwarg_name = "conjunction_radius"

    def __init__(self, initial_radius: float = 200.0, final_radius: float = 10.0, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_radius, final=final_radius, eval_env=eval_env, verbose=verbose)


class CorridorAngleScheduler(LinearParameterScheduler):
    """
    Linearly tightens the docking approach corridor cone angle.

    Example: start at 360° (omnidirectional, no penalty) → 30° (tight corridor).
    Both SparseEventReward and DockingCorridorReward are updated simultaneously
    via Sb3BksEnv.set_scheduled_parameters.
    """
    _log_key = "curriculum/corridor_angle_deg"
    _kwarg_name = "corridor_angle_deg"

    def __init__(self, initial_angle_deg: float = 360.0, final_angle_deg: float = 30.0, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_angle_deg, final=final_angle_deg, eval_env=eval_env, verbose=verbose)


class AttitudeErrorScheduler(LinearParameterScheduler):
    """
    Linearly tightens the maximum RSO attitude offset used during episode init.

    Only active when rso_att_type="near_velocity". Controls how far the RSO's
    attitude may deviate from velocity-pointing at the start of each episode.

    Example: start at 90° (random hemisphere) → 5° (nearly aligned).
    """
    _log_key = "curriculum/rso_max_error_deg"
    _kwarg_name = "max_error_deg"

    def __init__(self, initial_error_deg: float = 90.0, final_error_deg: float = 5.0, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_error_deg, final=final_error_deg, eval_env=eval_env, verbose=verbose)


class DeltaVPenaltyScheduler(LinearParameterScheduler):
    """
    Linearly increases the delta-v penalty weight over the course of training.

    Starts with a small penalty (or zero) so the agent first learns to dock, then
    gradually raises the cost of fuel use to encourage efficient manoeuvres.

    Pass positive values — DeltaVReward negates internally so larger = stronger penalty.
    Example: start at 0.0 (no fuel penalty) → 0.45 (strong quadratic penalty).
    """
    _log_key = "curriculum/dv_penalty_weight"
    _kwarg_name = "dv_penalty_weight"

    def __init__(self, initial_weight: float = 0.0, final_weight: float = 0.45, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_weight, final=final_weight, eval_env=eval_env, verbose=verbose)


class MaxDriftDurationScheduler(LinearParameterScheduler):
    """
    Linearly increases the maximum drift duration after each impulse.

    Longer drifts require the agent to plan further ahead (harder); starting
    short lets it learn coarse maneuvering before committing to sparse actions.

    Example: start at 30 s (easy, frequent corrections) → 120 s (hard, long coasts).
    """
    _log_key = "curriculum/max_drift_duration"
    _kwarg_name = "max_drift_duration"

    def __init__(self, initial_duration: float = 30.0, final_duration: float = 120.0, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_duration, final=final_duration, eval_env=eval_env, verbose=verbose)


class MaxDVScheduler(LinearParameterScheduler):
    """
    Linearly decreases the maximum delta-V per impulse.

    Reducing max_dv forces the agent toward smaller, more efficient maneuvers
    that match tighter real-world propulsion constraints.

    Example: start at 2.0 m/s (easy, ample thrust) → 0.5 m/s (hard, tight budget).
    """
    _log_key = "curriculum/max_dv"
    _kwarg_name = "max_dv"

    def __init__(self, initial_dv: float = 2.0, final_dv: float = 0.5, eval_env=None, verbose: int = 0):
        super().__init__(initial=initial_dv, final=final_dv, eval_env=eval_env, verbose=verbose)
