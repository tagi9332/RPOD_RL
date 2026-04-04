import numpy as np

# Assuming these are available in your environment
from bsk_rl.utils.orbital import random_unit_vector
from resources import (
    MIN_REL_POS,
    MAX_REL_POS,
    MIN_REL_VEL,
    MAX_REL_VEL
)

def generate_random_inspector_state(min_pos=MIN_REL_POS, max_pos=MAX_REL_POS, 
                                    min_vel=MIN_REL_VEL, max_vel=MAX_REL_VEL):
    """
    Generates a random 6-DOF relative state [rx, ry, rz, vx, vy, vz] 
    for an inspector satellite.
    
    Args:
        min_pos (float): Minimum relative position distance.
        max_pos (float): Maximum relative position distance.
        min_vel (float): Minimum relative velocity magnitude.
        max_vel (float): Maximum relative velocity magnitude.
        
    Returns:
        np.ndarray: A 6-element array containing the relative position and velocity.
    """
    pos = random_unit_vector() * np.random.uniform(min_pos, max_pos)
    vel = random_unit_vector() * np.random.uniform(min_vel, max_vel)
    
    return np.concatenate((pos, vel))

if __name__ == "__main__":
    # 1. Generate a single random starting state to be used across all resets
    fixed_state = generate_random_inspector_state()
    
    print("--- Fixed Inspector State Generated ---")
    print(f"Position (x,y,z): {fixed_state[:3]}")
    print(f"Velocity (u,v,w): {fixed_state[3:]}\n")
        
    # Example usage in the simulation framework:
    # env_args = randomizer_func(simulation_satellites)