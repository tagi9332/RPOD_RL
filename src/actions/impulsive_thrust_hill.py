import numpy as np
from gymnasium import spaces
from bsk_rl.act.continuous_actions import ContinuousAction
from typing import Optional

class ImpulsiveThrust(ContinuousAction):
    def __init__(
        self,
        name: str = "thrust_act",
        max_dv: float = float("inf"),
        max_drift_duration: float = float("inf"),
        fsw_action: Optional[str] = None,
    ) -> None:
        """Perform an impulsive thrust and drift for some duration.

        Args:
            name: Name of the action.
            max_dv: Maximum delta-V that can be applied. [m/s]
            max_drift_duration: Maximum duration to drift after applying the delta-V. [s]
            fsw_action: Name of the FSW action to activate during the drift period.
        """
        super().__init__(name)
        self.max_dv = max_dv
        self.max_drift_duration = max_drift_duration
        self.fsw_action = fsw_action

    @property
    def space(self) -> spaces.Box:
        """Return the action space.

        All four dimensions are fixed to [-1, 1] regardless of max_dv or
        max_drift_duration.  This prevents SB3's DiagGaussianDistribution
        (mean≈0, std=1 at init) from clipping 92%+ of samples to the boundary
        when max_dv is small (e.g. 0.1 m/s), and decouples the action-space
        shape from these parameters so models remain valid across curriculum
        changes.  set_action decodes both dimensions back to physical units.
        """
        return spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4,  1.0, dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

    @property
    def action_description(self) -> list[str]:
        """Description of the continuous action space."""
        return ["dV_N_x", "dV_N_y", "dV_N_z", "duration"]

    def set_action(self, action: np.ndarray) -> None:
        """Thrust the satellite with a given inertial delta-V and drift for some duration.

        Args:
            action: vector of [dV_N_x, dV_N_y, dV_N_z, duration], all normalized to
                [-1, 1].  dV components are scaled by max_dv; duration is mapped
                linearly from [-1, 1] to [2*sim_rate, max_drift_duration].
        """
        assert len(action) == 4, "Action must have 4 elements."

        # 1. Decode dV from normalized [-1, 1] to physical units.
        # The network output is in [-1, 1] per component; scaling by max_dv gives
        # [-max_dv, max_dv] per component, with sphere magnitude up to sqrt(3)*max_dv
        # for corner actions.
        dv_N = action[0:3] * self.max_dv
        requested_mag = np.linalg.norm(dv_N)

        # 2. Track how far outside the physical sphere the decoded action landed.
        overshoot = max(0.0, float(requested_mag - self.max_dv))
        self.satellite.latest_action_overshoot = overshoot

        # 3. Clamp to sphere — only fires for off-axis corner actions where
        # individual components are near ±1 simultaneously.
        if requested_mag > self.max_dv:
            self.satellite.logger.info(
                f"Thrust clamped from {requested_mag:.4f} m/s to {self.max_dv} m/s."
            )
            dv_N = (dv_N / requested_mag) * self.max_dv

        # Decode normalized duration from [-1, 1] to [dt_min, dt_max] seconds.
        # A network output of 0 (natural init) maps to the midpoint of the range.
        dt_min = 2.0 * self.satellite.simulator.sim_rate
        dt_max = self.max_drift_duration
        dt_normalized = float(np.clip(action[3], -1.0, 1.0))
        dt = dt_min + (dt_normalized + 1.0) / 2.0 * (dt_max - dt_min)

        self.satellite.logger.info(
            f"Thrusting with inertial dV {dv_N} with {dt} second drift."
        )
        self.satellite.latest_burn_dv_mag = float(np.linalg.norm(dv_N))
        self.satellite.latest_burn_drift_duration = float(dt)
        self.satellite.fsw.action_impulsive_thrust(dv_N)
        self.satellite.update_timed_terminal_event(
            self.satellite.simulator.sim_time + dt
        )

        # Activate the FSW action for the drift period
        # TODO: should wait until action_impulsive_thrust is done if not immediate
        if self.fsw_action is not None:
            getattr(self.satellite.fsw, self.fsw_action)()
            self.satellite.logger.info(f"FSW action {self.fsw_action} activated.")


class ImpulsiveThrustHill(ImpulsiveThrust):
    def __init__(self, chief_name, *args, **kwargs):
        """Impulsive thrusts in the Hill frame.

        Args:
            chief_name: Chief to use for Hill frame.
            *args: Passed to ``ImpulsiveThrust``.
            **kwargs: Passed to ``ImpulsiveThrust``.
        """
        self.chief_name = chief_name
        super().__init__(*args, **kwargs)

    def reset_post_sim_init(self) -> None:
        """Connect to the chief satellite.

        :meta private:
        """
        self.chief = self.satellite.simulator.get_satellite(self.chief_name)

    def set_action(self, action: np.ndarray) -> None:
        """Activate the action by setting the continuous value."""
        dv_H = action[0:3]
        dt = action[3]

        NH = self.chief.dynamics.HN.T
        dv_N = NH @ dv_H

        super().set_action(np.concatenate((dv_N, [dt])))