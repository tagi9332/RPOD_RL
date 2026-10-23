import os

import numpy as np
from Basilisk.architecture import sysModel
from Basilisk.simulation import vizInterface
from Basilisk.utilities import vizSupport
from Basilisk.utilities import macros

from resources import (
    approach_corridor_angle_deg,
    inspector_boresight,
    docking_port_boresight,
)

# Visual-only sizes (do not affect the sim)
CORRIDOR_CONE_HEIGHT = 200.0     # [m] docking corridor length drawn from the RSO port
PORT_BORESIGHT_HEIGHT = 200.0    # [m] docking port boresight line length
OPTICS_FOV_HALF_ANGLE_DEG = 20.0 # [deg] inspector sensor FOV half-angle
OPTICS_FOV_HEIGHT = 30.0         # [m] inspector sensor FOV cone length
OPTICS_BORESIGHT_HEIGHT = 50.0   # [m] inspector optics boresight line length
BORESIGHT_HALF_ANGLE_DEG = 0.5   # [deg] near-zero cone angle used to draw a boresight "line"
CAMERA_BORESIGHT_OFFSET = 2.0    # [m] camera offset along the boresight, past the solar array model

# The recorder task is appended to RSODynProcess, the lowest-priority process in the sim
# (World=300, Dyn=200, FSW=100, RSODyn=1), so it runs only after every spacecraft has been
# propagated to the current time.
VIZ_HOST_PROCESS = "RSODynProcess"


def _vec(v):
    return [float(x) for x in v]


class _RangeRecorder(sysModel.SysModel):
    """
    Feeds the inspector-RSO range to a Vizard generic storage HUD panel (Vizard has no
    free-text overlay). The bar maximum is the largest range seen so far.
    """

    def __init__(self, rso_sc, insp_sc):
        super().__init__()
        self.ModelTag = "vizRangeRecorder"
        self.rso_sc = rso_sc
        self.insp_sc = insp_sc

        # VizSpacecraftData stores raw GenericStorage pointers, so the panel must be owned by
        # something that outlives vizard_output(); this module is held by the sim's task list.
        # The recorder serializes the panel's currentValue/maxValue each frame, so they are set
        # directly (no input msg is linked).
        self.panel = vizInterface.GenericStorage()
        self.panel.label = "Range to RSO"
        self.panel.units = "m"

    def UpdateState(self, CurrentSimNanos):
        r_rso = np.array(self.rso_sc.scStateOutMsg.read().r_BN_N)
        r_insp = np.array(self.insp_sc.scStateOutMsg.read().r_BN_N)
        rel_range = float(np.linalg.norm(r_insp - r_rso))

        self.panel.currentValue = rel_range
        self.panel.maxValue = max(self.panel.maxValue, rel_range)


def vizard_output(env, output_folder, run_idx):
    """
    Attach a Vizard recorder to the current episode. Must be called after env.reset().

    Centers the camera on the inspector and shows relative motion in its Hill frame, the RSO docking
    corridor and port boresight, the inspector optics boresight/FOV, and an inspector HUD panel
    with the live inspector-RSO range.
    """
    sim = env.simulator
    rso_sc = env.satellites[0].dynamics.scObject
    insp_sc = env.satellites[1].dynamics.scObject
    rso_name = rso_sc.ModelTag
    insp_name = insp_sc.ModelTag

    viz_save_dir = os.path.abspath(os.path.join(output_folder, "vizard_data"))
    os.makedirs(viz_save_dir, exist_ok=True)
    viz_filepath = os.path.join(viz_save_dir, f"run_{run_idx + 1}_vizard.bin")

    # Each satellite has its own dynamics process. Recording from inside one of them
    # (e.g. DynamicsTaskRSO) logs the other spacecraft one step stale, which at orbital
    # speed shows up as a ~7.5 km along-track offset between the chief and deputy path.
    # Creating a new process here instead would need TotalSim.assignRemainingProcs() since the
    # sim is already initialized, and that deadlocks the simulator teardown on the next reset.
    viz_proc = next(p for p in sim.procList if p.Name == VIZ_HOST_PROCESS)
    viz_task_name = "VizardTask"
    viz_proc.addTask(sim.CreateNewTask(viz_task_name, macros.sec2nano(sim.sim_rate)))

    # vizSupport keeps standard cameras and instrument GUI settings in module-level lists that
    # enableUnityVisualization does not clear, so they would pile up across runs in the same worker.
    del vizSupport.stdCameraList[:]
    del vizSupport.instrumentGuiSettingList[:]

    # Relative range readout. Higher priority than the Vizard recorder (default -1) so the
    # range is updated before the recorder serializes it in the same step.
    range_recorder = _RangeRecorder(rso_sc, insp_sc)
    sim.AddModelToTask(viz_task_name, range_recorder, 100)

    viz = vizSupport.enableUnityVisualization(
        sim,
        viz_task_name,
        [rso_sc, insp_sc],
        saveFile=viz_filepath,
        genericStorageList=[None, [range_recorder.panel]],
    )

    # Correct date for Earth orientation / lighting
    viz.epochInMsg.subscribeTo(sim.world.gravFactory.epochMsg)

    # A. Trajectory & relative motion (Hill frame)
    # The relative-orbit chief defaults to the camera target (Inspector), so relative lines show
    # the RSO path about the Inspector; switch the chief to the RSO in Vizard's GUI if needed.
    # Don't set liveSettings.relativeOrbitChief: Vizard's chief dropdown throws a
    # NullReferenceException on it (at load it leaves only a blank background).
    viz.settings.orbitLinesOn = 2              # 2 = Relative to chief spacecraft
    viz.settings.trueTrajectoryLinesOn = 2     # 2 = True path relative to chief
    viz.settings.relativeOrbitFrame = 1        # 1 = Hill frame
    viz.settings.showHillFrame = 1             # Hill frame of the camera target (Inspector)
    viz.settings.mainCameraTarget = insp_name  # Center the view on the Inspector

    # B. Spacecraft frames & labels
    viz.settings.showSpacecraftLabels = 1
    viz.settings.spacecraftCSon = 1            # Body axes (shows attitude of both vehicles)
    viz.settings.showCSLabels = 1

    # C. HUD settings
    vizSupport.setInstrumentGuiSetting(viz, spacecraftName=insp_name, showGenericStoragePanel=1)
    viz.settings.viewCameraBoresightHUD = 1
    viz.settings.viewCameraFrustumHUD = 1
    viz.settings.showDataRateDisplay = -1
    viz.settings.ambient = 0.5

    # --- RSO (chief) ---

    # 1. Docking keep-in corridor; turns on/off as the inspector enters/leaves it
    vizSupport.createConeInOut(viz,
        fromBodyName=rso_name,
        toBodyName=insp_name,
        normalVector_B=_vec(docking_port_boresight),
        incidenceAngle=approach_corridor_angle_deg * macros.D2R,
        coneHeight=CORRIDOR_CONE_HEIGHT,
        coneColor=vizSupport.toRGBA255("green", alpha=0.25),
        isKeepIn=True,
        coneName="dockingCorridor"
    )

    # 2. Docking port boresight
    vizSupport.createConeInOut(viz,
        fromBodyName=rso_name,
        toBodyName=insp_name,
        normalVector_B=_vec(docking_port_boresight),
        incidenceAngle=BORESIGHT_HALF_ANGLE_DEG * macros.D2R,
        coneHeight=PORT_BORESIGHT_HEIGHT,
        coneColor="lime",
        isKeepIn=True,
        coneName="portBoresight"
    )

    # --- Inspector (deputy) ---

    # 3. Optics boresight
    vizSupport.createConeInOut(viz,
        fromBodyName=insp_name,
        toBodyName=rso_name,
        normalVector_B=_vec(inspector_boresight),
        incidenceAngle=BORESIGHT_HALF_ANGLE_DEG * macros.D2R,
        coneHeight=OPTICS_BORESIGHT_HEIGHT,
        coneColor="magenta",
        isKeepIn=True,
        coneName="opticsBoresight"
    )

    # 4. Optics FOV; turns on/off as the RSO enters/leaves the FOV
    vizSupport.createConeInOut(viz,
        fromBodyName=insp_name,
        toBodyName=rso_name,
        normalVector_B=_vec(inspector_boresight),
        incidenceAngle=OPTICS_FOV_HALF_ANGLE_DEG * macros.D2R,
        coneHeight=OPTICS_FOV_HEIGHT,
        coneColor=vizSupport.toRGBA255("cyan", alpha=0.25),
        isKeepIn=True,
        coneName="opticsFOV"
    )

    # 5. Inspector -> Sun and Inspector -> RSO lines (the angle between them is the sun angle)
    vizSupport.createPointLine(viz, fromBodyName=insp_name, toBodyName="sun", lineColor="yellow")
    vizSupport.createPointLine(viz, fromBodyName=insp_name, toBodyName=rso_name, lineColor="white")

    # 6. Camera view through the inspector optics
    # A zero position_B makes Vizard auto-place the camera off-axis; an explicit offset along the
    # boresight keeps the frustum centered on it while clearing the spacecraft and solar arrays.
    boresight_hat = np.asarray(inspector_boresight, dtype=float)
    boresight_hat /= np.linalg.norm(boresight_hat)
    vizSupport.createStandardCamera(viz,
        spacecraftName=insp_name,
        setMode=1,
        pointingVector_B=_vec(boresight_hat),
        position_B=_vec(CAMERA_BORESIGHT_OFFSET * boresight_hat),
        fieldOfView=2 * OPTICS_FOV_HALF_ANGLE_DEG * macros.D2R,
        displayName="Inspector Optics"
    )

    # Initialize memory without wiping RL states
    viz.Reset(0)

    return viz
