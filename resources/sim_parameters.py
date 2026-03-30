# Simulation Time Limit
SIM_TIME = 10800

# Simulation Time Step
SIM_DT = 1.0

# Delta-V Action Limits
MAX_DV = 0.2  # m/s
MAX_DRIFT_DURATION = 2000 # s

# Relative State Initialization Bounds
MAX_REL_POS = 505.0  # meters
MIN_REL_POS = 500     # meters
MAX_REL_VEL = 0.01    # m/s
MIN_REL_VEL = 0.0      # m/s

# --- REWARDER PARAMETERS ---
# Approach Corridor Angle (degrees)
approach_corridor_angle_deg=90

# Phase Transition Parameters
docking_phase_range_threshold=500

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
    imageAttErrorRequirement=1.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e12,
    storedCharge_Init=1e12,
    conjunction_radius=30,
    dv_available_init=150,
    max_range_radius=5000,
    chief_name="RSO",
    u_max=2.0
)

