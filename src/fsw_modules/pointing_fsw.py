from typing import TYPE_CHECKING, Iterable

from Basilisk.architecture import messaging
from bsk_rl.sim.fsw import RSOInspectorFSWModel as LibraryRSOInspectorFSWModel
from bsk_rl.sim.fsw import ContinuousImagingFSWModel, action
from bsk_rl.utils.functional import default_args

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite

class RSOInspectorFSWModel(LibraryRSOInspectorFSWModel):
    
    def _make_task_list(self):
        """Ensure the pointing task is added to the FSW task list."""
        return super()._make_task_list() + [self.LocPointTask(self)]

    class LocPointTask(LibraryRSOInspectorFSWModel.LocPointTask):
        """Task to point at the RSO and trigger the instrument."""
        name = "locPointTask"

        def __init__(self, fsw, priority=96) -> None:
            super().__init__(fsw, priority)

        @default_args(inst_pHat_B=[0, 0, 1], pointing_target_sc="RSO")
        def setup_location_pointing(
            self, inst_pHat_B: Iterable[float], pointing_target_sc: str, **kwargs
        ) -> None:
            """Set up the pointing geometry and connect attitude guidance."""
            self.locPoint.pHat_B = inst_pHat_B
            self.locPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.locPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.locPoint.useBoresightRateDamping = 1

            # --- THE FIX: Connect the target message immediately during setup ---
            target_sat = next((sat for sat in self.fsw.simulator.satellites if sat.name == pointing_target_sc), None)
            if target_sat is not None:
                self.locPoint.scTargetInMsg.subscribeTo(
                    target_sat.dynamics.simpleNavObject.transOutMsg
                )
            else:
                raise ValueError(f"Could not find target satellite '{pointing_target_sc}' in the simulator!")
            # ------------------------------------------------------------------

            # RSO Inspector guidance connection
            messaging.AttGuidMsg_C_addAuthor(
                self.locPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )
            
            # AlwaysPoint direct reference connection (Forces the spacecraft to rotate)
            self.fsw.dynamics.scObject.attRefInMsg.subscribeTo(
                self.locPoint.attRefOutMsg
            )

            self._add_model_to_task(self.locPoint, priority=1198)

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Set the instrument controller parameters for scanning."""
            self.insControl.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControl.useRateTolerance = 1
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            self.insControl.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            
            # Only use this module to check for pointing requirements
            self.access_msg = messaging.AccessMsg()
            payload = messaging.AccessMsgPayload()
            payload.hasAccess = 1
            self.access_msg.write(payload)
            self.insControl.accessInMsg.subscribeTo(self.access_msg)

            self._add_model_to_task(self.insControl, priority=987)

        def reset_for_action(self) -> None:
            """Crucial: Don't disable the pointing module each RL step."""
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.name)

    @action
    def action_inspect_rso(self) -> None:
        """Action to inspect the RSO (Turns on the instrument payload)."""
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.instrument.nodeDataName = "inspect_rso"
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)