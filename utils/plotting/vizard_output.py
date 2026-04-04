import os

from Basilisk.utilities import vizSupport
from Basilisk.utilities import macros

from resources import (
    approach_corridor_angle_deg,
    inspector_boresight,
)

def vizard_output(env, output_folder, run_idx):

    # 1. Grab the raw Basilisk simulator directly from the base env
    sim = env.simulator

    # 2. Extract the C++ spacecraft objects so Vizard knows what to draw
    rso_sc = env.satellites[0].dynamics.scObject
    insp_sc = env.satellites[1].dynamics.scObject
    sc_objects = [rso_sc, insp_sc]

    # 3. Create a unique save path for this run's Vizard data
    viz_save_dir = os.path.abspath(os.path.join(output_folder, "vizard_data"))
    os.makedirs(viz_save_dir, exist_ok=True)
    viz_filepath = os.path.join(viz_save_dir, f"run_{run_idx + 1}_vizard.bin") 

    # 4. Find a valid task name dynamically from the C++ core
    task_names = [task.TaskPtr.TaskName for proc in sim.TotalSim.processList for task in proc.processTasks]
    viz_task_name = next((name for name in task_names if 'dyn' in name.lower()), task_names[0])

    # 5. Enable the recorder and capture the viz module object
    viz = vizSupport.enableUnityVisualization(
        sim, 
        viz_task_name, 
        sc_objects, 
        saveFile=viz_filepath
    )

    # --- THE OVERHAUL: RPO & DOCKING VISUALS ---

    # A. Trajectory & Relative Motion
    viz.settings.orbitLinesOn = 2              # 2 = Relative to chief spacecraft
    viz.settings.trueTrajectoryLinesOn = 2     # 2 = True path relative to chief
    viz.settings.relativeOrbitFrame = 1        # 1 = Use Hill Frame (Standard for RPO)
    viz.settings.showHillFrame = 1             # Draw the Hill frame axes (Radial, Along-track, Cross-track)
    viz.settings.relativeOrbitChief = rso_sc.ModelTag # Set RSO as the center of the universe

    # B. Spacecraft Frames & Labels
    viz.settings.showSpacecraftLabels = 1
    viz.settings.spacecraftCSon = 1            # Show local X, Y, Z axes of the spacecraft
    viz.settings.showCSLabels = 1              # Label those axes (helps align docking ports)

    # C. Camera & HUD Settings
    viz.settings.mainCameraTarget = insp_sc.ModelTag # Focus the camera on your RL agent
    viz.settings.viewCameraBoresightHUD = 1    # Draws a line out the front of your camera
    viz.settings.viewCameraFrustumHUD = 1      # Draws the sensor cone
    viz.settings.showDataRateDisplay = -1      # Hide clutter
    viz.settings.ambient = 0.5                 # Brighten the dark side of the spacecraft a bit

    # 1. The Docking Keep-In Corridor
    # createConeInOut draws a physical cone and checks the angle between the 
    # normal vector and the vector to the target body.
    vizSupport.createConeInOut(viz,
        fromBodyName=rso_sc.ModelTag,      # Cone originates from the RSO
        toBodyName=insp_sc.ModelTag,       # Evaluates if the Inspector is inside it
        normalVector_B=[0, 0, 1],    # Direction of the RSO's docking port (+X axis)
        incidenceAngle=approach_corridor_angle_deg * macros.D2R,  # 15-degree half-angle safe approach corridor
        coneHeight=200.0,                  # Draw the cone out to 200 meters
        coneColor="green", 
        isKeepIn=True,                     # True = Keep-In cone
        coneName="dockingCorridor"
    )

    # 2. Inspector Sensor Boresight
    vizSupport.createConeInOut(viz,
        fromBodyName=insp_sc.ModelTag,     # Originates from Inspector
        toBodyName=rso_sc.ModelTag,        # Evaluates if RSO is within the FOV
        normalVector_B=inspector_boresight,    # Assuming Inspector sensor points along its +X
        incidenceAngle=20.0 * macros.D2R,  # 20-degree half-angle FOV
        coneHeight=30,                  # Max range of the sensor
        coneColor="cyan",
        isKeepIn=True,                     
        coneName="sensorBoresight"
    )

    # 3. Dynamic Sun Vector
    # Draws a line constantly pointing from the Inspector to the Sun.
    vizSupport.createPointLine(viz,
        fromBodyName=insp_sc.ModelTag,
        toBodyName="sun",                  # Vizard recognizes "sun" natively
        lineColor="yellow" 
    )

    # 4. Target Vector 
    # Draws a line from the Inspector directly to the RSO.
    vizSupport.createPointLine(viz,
        fromBodyName=insp_sc.ModelTag,
        toBodyName=rso_sc.ModelTag,
        lineColor="white"     
    )

    # 6. Initialize memory without wiping RL states
    viz.Reset(0)
