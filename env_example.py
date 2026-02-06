import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Reward Function Definition
# -----------------------------
def reward(delta_x, delta_x_max, alpha=1.0):
    """
    Computes R1,t = α * log( ( || δx ⊘ δx_max || / ||1_6|| )^2 )
    Inputs:
        delta_x      : (6,) ndarray - relative state
        delta_x_max  : (6,) ndarray - normalization vector
        alpha        : scalar reward gain
    """
    one_norm = np.linalg.norm(np.ones(6))  # ||1_6||
    norm_ratio = np.linalg.norm(delta_x / delta_x_max) / one_norm
    
    # Avoid log(0)
    norm_ratio = np.maximum(norm_ratio, 1e-12)
    
    return alpha * np.log(norm_ratio**2)

# -----------------------------
# Simulation Parameters
# -----------------------------
alpha = 1.0
delta_x_max = np.array([500, 500, 500, 5, 5, 5])  # normalization values
num_points = 200

# Sweep δx along a scaling factor between 0 and 1
scales = np.linspace(0, 1, num_points)

# For plotting, apply δx = s * δx_max * direction
direction = np.array([1, 1, 1, 1, 1, 1])  # any direction works; normalized automatically
direction = direction / np.linalg.norm(direction)

rewards = []
for s in scales:
    delta_x = s * delta_x_max * direction  
    rewards.append(reward(delta_x, delta_x_max, alpha))

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(scales, rewards, linewidth=2)
plt.title("Reward Response vs Relative State Magnitude", fontsize=14)
plt.xlabel("Relative Error Magnitude Scale (0 → 1)", fontsize=12)
plt.ylabel("Reward R₁,t", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
