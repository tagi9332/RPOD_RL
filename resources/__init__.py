from .constants import R_EARTH
from .weights import (
    dv_reward_weight,
    rel_range_log_weight,
    docking_reward,
    misalignment_discount_factor,
    max_range_penalty,
    approach_corridor_weight,
    conjunction_penalty,
    time_penalty_weight,
    illumination_weight,
    approach_velocity_weight

)
from .hyperparameters import (
    learning_rate,
    entropy_coeff,
    max_grad_norm,
    clip_range
)

from .sim_parameters import (
    SIM_TIME,
    SIM_DT,
    MAX_REL_POS,
    MAX_REL_VEL,
    MIN_REL_POS,
    MIN_REL_VEL,
    MAX_DV,
    MAX_DRIFT_DURATION,
    rso_sat_args,
    inspector_sat_args,
    approach_corridor_angle_deg,
    docking_phase_range_threshold,
    inspector_boresight,
    docking_port_boresight,
    sun_illumination_cone_angle_deg,
    illumination_cutoff_range,
    final_docking_distance,
    final_corridor_angle_deg
)