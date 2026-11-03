# ResourceReward
dv_reward_weight=0.75

# RelativeRangeLogReward
rel_range_log_weight=0.0003  # applied as -weight * log(MSE)
approach_velocity_weight = 5.0

# Docking Reward
docking_reward=20

# Max Range Penalty
max_range_penalty=20  # magnitude; subtracted from reward on violation

# Boresight Alignment Reward
approach_corridor_weight=0.05

# Time Penalty — TOTAL penalty over a full episode (step-size independent integral form).
time_penalty_weight=10

# Illumination Reward
illumination_weight=0.0025

# WaypointPhaseReward
# Same log-MSE formulation as RelativeRangeLogReward; replaces it entirely.
waypoint_pos_weight = 0.0003        # log-MSE alpha, applied as -weight * log(MSE) (same as rel_range_log_weight)
waypoint_sparse_reward = 10.0        # one-time bonus on Phase 3 entry (first crossing of the 30 m range)
# Phases 1–2 (rendezvous + waypoint navigation): velocity penalty scales with proximity to waypoint.
# vel_weight_phase0 multiplies the normalised velocity inside the MSE ("phase0" = internal flag for Phases 1–2).
# Braking curve is zero vel_onset_range+ m from waypoint (Phase 1), ramps to full at the waypoint (Phase 2).
# With MAX_DV=0.5 m/s and ~30s steps, stopping from ~3 m/s needs ~6 burns over ~400m.
vel_weight_phase0 = 1.0             # loose — agent can sprint in Phase 1, must slow in Phase 2
vel_onset_range = 500.0             # m — distance from waypoint at which Phase 2 braking starts
# Phase 3 (terminal ingress and docking): always-on, stronger weight enforces controlled ingress.
# ("phase1" = internal flag for Phase 3.)
vel_weight_phase1 = 5.0             # firm — same as legacy approach_velocity_weight