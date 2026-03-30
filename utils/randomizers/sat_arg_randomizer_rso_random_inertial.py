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

def make_sat_arg_randomizer(mode="train"):
    """
    Factory function to generate a sat_arg_randomizer based on the execution mode.
    - "train": Fully randomizes RSO and Inspector on every reset.
    - "test": Randomizes RSO once, then keeps its absolute orbit and attitude 
              consistent across all subsequent resets. Inspector remains randomized.
    """
    
    # This dictionary acts as our memory cache across environment resets
    persistent_rso_state = {}

    def sat_arg_randomizer(satellites):
        nonlocal persistent_rso_state # Allows us to modify the outer dictionary

        # Determine if we need to generate fresh RSO parameters
        generate_new_rso = (mode == "train") or (not persistent_rso_state)

        if generate_new_rso:
            # 1. Generate new Chief Orbit
            a = (R_EARTH*1000) + np.random.uniform(35776.0*1000, 35796.0*1000)
            e = np.random.uniform(0.0, 0.0005)
            chief_orbit = random_orbit(a=a, e=e)

            # 2. Generate new RSO Attitude (MRP)
            u = np.random.uniform(0, 1, 3)
            q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
            q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
            q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
            q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
            
            sigma_init = np.array([q1, q2, q3]) / (1 + q0)
            if np.linalg.norm(sigma_init) > 1:
                sigma_init = -sigma_init / (np.linalg.norm(sigma_init)**2)

            omega_init = np.array([0.0, 0.0, 0.0])

            # Cache these so we can reuse them if mode == "test"
            persistent_rso_state = {
                "chief_orbit": chief_orbit,
                "sigma_init": sigma_init,
                "omega_init": omega_init
            }

        # Retrieve the parameters (either fresh ones or cached ones)
        chief_orbit = persistent_rso_state["chief_orbit"]
        sigma_init = persistent_rso_state["sigma_init"]
        omega_init = persistent_rso_state["omega_init"]

        # --- Apply the states to the satellites ---
        inspectors = [sat for sat in satellites if "Inspector" in sat.name]
        rso = [satellite for satellite in satellites if satellite.name == "RSO"][0]
        args = {}
        
        # The Inspector's relative state is ALWAYS randomized, regardless of mode
        for inspector in inspectors:
            relative_randomizer = relative_to_chief(
                chief_name="RSO", chief_orbit=chief_orbit,
                deputy_relative_state={
                    inspector.name: lambda: np.concatenate((
                        random_unit_vector() * np.random.uniform(MIN_REL_POS, MAX_REL_POS), 
                        random_unit_vector() * np.random.uniform(MIN_REL_VEL, MAX_REL_VEL)
                    )),
                },
            )
            args.update(relative_randomizer([rso, inspector]))
        
        # Apply the RSO attitude logic
        args[rso]["sigma_init"] = sigma_init
        args[rso]["omega_init"] = omega_init
        
        return args

    return sat_arg_randomizer