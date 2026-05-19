# ResourceReward
dv_reward_weight=0.5
dv_constant_penalty=0

# RelativeRangeLogReward
rel_range_log_weight=-0.0003
approach_velocity_weight = 5.0

# Docking Reward
docking_reward=20

# Misalignment Discount Factor (Deprecated)
misalignment_discount_factor=0.5

# Max Range Penalty
max_range_penalty=-20

# Boresight Alignment Reward
approach_corridor_weight=0.05

# Time Penalty — TOTAL penalty over a full episode (step-size independent integral form).
# At t=T: cumulative = -time_penalty_weight.  At t=T/3: cumulative ≈ -time_penalty_weight*3.7%.
# Set > expected dense-reward for a full loitering run (~55) to make coasting unprofitable.
time_penalty_weight=40.0

# # Conjunction Penalty (Deprecated)
conjunction_penalty=-20 

# Illumination Reward
illumination_weight=0.0025

# WaypointPhaseReward
# Same log-MSE formulation as RelativeRangeLogReward; replaces it entirely.
waypoint_pos_weight = -0.0003       # log-MSE alpha (same magnitude as rel_range_log_weight)
waypoint_sparse_reward = 10.0        # one-time bonus when agent first reaches the 30m standoff
# Phase 0 (approach to waypoint): velocity penalty scales with proximity to waypoint.
# vel_weight_phase0 multiplies the normalised velocity inside the MSE.
# Braking curve is zero vel_onset_range+ m from waypoint, ramps to full at the waypoint.
# With MAX_DV=0.5 m/s and ~30s steps, stopping from ~3 m/s needs ~6 burns over ~400m.
vel_weight_phase0 = 1.0             # loose — agent can sprint early, must slow near waypoint
vel_onset_range = 500.0             # m — distance at which Phase 0 braking starts
# Phase 1 (final approach): always-on, stronger weight enforces controlled ingress.
vel_weight_phase1 = 5.0             # firm — same as legacy approach_velocity_weight