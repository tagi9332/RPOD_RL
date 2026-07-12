import numpy as np

# Simulation Time Limit
SIM_TIME = 10800  # seconds (3 hours)

# Simulation Time Step
SIM_DT = 1.0

# Delta-V Action Limits
MAX_DV = 0.1  # m/s  — matches 10 N thruster on 200 kg s/c
DV_AVAILABLE_INIT = 150  # m/s — initial fuel budget (must match inspector_sat_args)
MAX_DRIFT_DURATION = 180 # s  (keep <~42s for rollouts to fit inside one episode)

# Relative State Initialization Bounds
MAX_REL_POS = 1005  # meters
MIN_REL_POS = 1000    # meters
MAX_REL_VEL = 1.00    # m/s
MIN_REL_VEL = 0.0      # m/s
INITIAL_APPROACH_VEL = 0  # m/s — initial velocity directed toward RSO

# Conjunction Radius for Docking success
CONJUNCTION_RADIUS = 335  # meters

# --- REWARDER PARAMETERS ---
approach_corridor_angle_deg=180 # 180° corresponds to no corridor constraint; all conjunctions count as dockings

# Phase Transition Parameters
docking_phase_range_threshold=500
final_docking_distance=30

# Waypoint Phase Parameters
STANDOFF_DISTANCE = 30.0      # m — body-fixed waypoint distance from docking port
WAYPOINT_CAPTURE_RADIUS = 10.0  # m — sphere radius that triggers Phase 0→1 transition
VEL_NORM = 1.0                # m/s — body-frame velocity normalization in reward

# Satellite Boresights
inspector_boresight = np.array([0.0, 0.0, 1.0])
docking_port_boresight = np.array([0.0, 0.0, 1.0])

# Illumination Reward Parameters
sun_illumination_cone_angle_deg = 60
illumination_cutoff_range = 500

# --- CONFIGURATION DICTIONARIES ---
rso_sat_args = dict(
    conjunction_radius=2.0,
    K=7.0 / 20,
    P=35.0 / 20,
    Ki=1e-6,
    dragCoeff=0.0,
    batteryStorageCapacity=1e9, 
    storedCharge_Init=1e9,
    wheelSpeeds=[0.0, 0.0, 0.0],
    u_max=2.0,
)

inspector_sat_args = dict(
    imageAttErrorRequirement=3.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e12,
    storedCharge_Init=1e12,
    conjunction_radius=CONJUNCTION_RADIUS,
    dv_available_init=150,
    max_range_radius=5000,
    chief_name="RSO",
    u_max=2.0
)

