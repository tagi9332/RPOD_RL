import numpy as np

# Basilisk core
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP

# BSK-RL framework
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief

# Import constants and sim parameters
from resources import (
    R_EARTH,
    MIN_REL_POS,
    MAX_REL_POS,
    MIN_REL_VEL,
    MAX_REL_VEL
)

def sat_arg_randomizer_rso_inertial(satellites):
    a = (R_EARTH*1000) + np.random.uniform(35776.0*1000, 35796.0*1000) # Near GEO orbit
    e = np.random.uniform(0.0, 0.0005)
    chief_orbit = random_orbit(a=a, e=e)
    inspectors = [sat for sat in satellites if "Inspector" in sat.name]
    rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]
    args = {}
    
    for inspector in inspectors:
        relative_randomizer = relative_to_chief(
            chief_name="RSO", chief_orbit=chief_orbit,
            deputy_relative_state={
                inspector.name: lambda: np.concatenate((random_unit_vector() * np.random.uniform(MIN_REL_POS, MAX_REL_POS), random_unit_vector() * np.random.uniform(MIN_REL_VEL, MAX_REL_VEL))),
            },
        )
        args.update(relative_randomizer([rso, inspector]))
    
    # --- RSO Random Inertial Attitude Logic ---
    
    # 1. Generate a uniformly distributed random quaternion
    u = np.random.uniform(0, 1, 3)
    q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
    q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
    q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
    q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
    
    # 2. Convert to MRPs
    sigma_init = np.array([q1, q2, q3]) / (1 + q0)
    
    # 3. Enforce the MRP short rotation (shadow set: magnitude <= 1)
    if np.linalg.norm(sigma_init) > 1:
        sigma_init = -sigma_init / (np.linalg.norm(sigma_init)**2)
        
    args[rso]["sigma_init"] = sigma_init
    
    # 4. Hold inertially fixed (zero angular velocity)
    args[rso]["omega_init"] = np.array([0.0, 0.0, 0.0])
    
    return args