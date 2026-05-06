# ResourceReward
dv_reward_weight=-0.05
dv_constant_penalty=0

# RelativeRangeLogReward
rel_range_log_weight=-0.0003
approach_velocity_weight = 5.0

# Docking Reward
docking_reward=20

# Misalignment Discount Factor (This determines how much to penalize dockings at the extreme edge of the approach corridor)
misalignment_discount_factor=0.8

# Max Range Penalty
max_range_penalty=-20

# Boresight Alignment Reward
approach_corridor_weight=0.01

# Time Penalty — TOTAL penalty over a full episode (step-size independent integral form).
# At t=T: cumulative = -time_penalty_weight.  At t=T/3: cumulative ≈ -time_penalty_weight*3.7%.
# Set > expected dense-reward for a full loitering run (~55) to make coasting unprofitable.
time_penalty_weight=80.0

# Conjunction Penalty
conjunction_penalty=-2

# Illumination Reward
illumination_weight=0.0025