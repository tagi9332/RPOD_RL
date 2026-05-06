# ResourceReward
dv_reward_weight=-0.05
dv_constant_penalty=0

# RelativeRangeLogReward
rel_range_log_weight=-0.0003
approach_velocity_weight = 5.0

# Docking Reward
docking_reward=10

# Misalignment Discount Factor (This determines how much to penalize dockings at the extreme edge of the approach corridor)
misalignment_discount_factor=0.8

# Max Range Penalty
max_range_penalty=-10

# Boresight Alignment Reward
approach_corridor_weight=0.01

# Time Penalty (quadratic-rate integral form: -w*(t^3-t_prev^3)/(3*T^2))
# w=0.02 gives: full-coast penalty=-72 (net=55-72=-17), dock@1000s penalty=-0.06 (net=9.94)
time_penalty_weight=0.015

# Conjunction Penalty
conjunction_penalty=-2

# Illumination Reward
illumination_weight=0.0025