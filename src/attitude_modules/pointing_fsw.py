from typing import TYPE_CHECKING, Iterable, Any
from bsk_rl.sim.fsw import RSOInspectorFSWModel, Task
from bsk_rl.utils.functional import default_args
from Basilisk.fswAlgorithms import locationPointing, attTrackingError, mrpFeedback

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite

class AlwaysPointFSWModel(RSOInspectorFSWModel):
    # This whitelists the keys so the Satellite doesn't throw a KeyError
    @default_args(
        inst_pHat_B=[0, 0, 1],
        K_pointing=3.5,
        P_pointing=30.0
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locPoint = None

    def _make_task_list(self) -> list:
        # Only return our custom task; this ignores all the heavy 
        # RSOInspectorFSWModel instrument/data tasks.
        return [self.LocPointTask(self, priority=100)]

    def set_target_rso(self, rso: "Satellite") -> None:
        """Manually link the target's position to our pointing module."""
        if self.locPoint is not None:
            self.locPoint.scTargetInMsg.subscribeTo(
                rso.dynamics.simpleNavObject.transOutMsg
            )

    class LocPointTask(Task):
        @property
        def name(self) -> str:
            # The framework requires an explicit name for the Basilisk task
            return "AlwaysPointTask"

        def _create_module_data(self) -> None:
            # 1. Create the Basilisk C-wrapped modules
            self.locPoint = locationPointing.locationPointing()
            self.trackingErrorData = attTrackingError.attTrackingError()
            self.trackingErrorConfig = attTrackingError.attTrackingErrorConfig()
            self.mrpControl = mrpFeedback.mrpFeedbackConfig()
            
            # Link to parent so set_target_rso can find it
            self.fsw.locPoint = self.locPoint

        def _setup_fsw_objects(self, **kwargs) -> None:
            # 2. Configure parameters and connect messages (The "Wiring")
            
            # Pull values from sat_args (whitelisted in __init__ above)
            sat_args = self.fsw.satellite.sat_args
            
            # Configure Controller
            self.mrpControl.K = sat_args.get("K_pointing")
            self.mrpControl.P = sat_args.get("P_pointing")
            
            # Configure Guidance
            self.locPoint.pHat_B = sat_args.get("inst_pHat_B")
            self.locPoint.scAttInMsg.subscribeTo(self.fsw.dynamics.simpleNavObject.attOutMsg)
            self.locPoint.scTransInMsg.subscribeTo(self.fsw.dynamics.simpleNavObject.transOutMsg)
            self.locPoint.useBoresightRateDamping = 1

            # Configure Error Sensing
            self.trackingErrorConfig.attNavInMsg.subscribeTo(self.fsw.dynamics.simpleNavObject.attOutMsg)
            self.trackingErrorConfig.attRefInMsg.subscribeTo(self.locPoint.attRefOutMsg)

            # Finalize Connections
            self.mrpControl.guidInMsg.subscribeTo(self.trackingErrorConfig.attGuidOutMsg)
            self.fsw.dynamics.extForceTorqueObject.cmdTorqueInMsg.subscribeTo(
                self.mrpControl.cmdTorqueOutMsg
            )

            # Add everything to the task execution list
            self._add_model_to_task(self.locPoint)
            self._add_model_to_task(self.trackingErrorData, self.trackingErrorConfig)
            self._add_model_to_task(self.mrpControl)