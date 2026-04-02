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

# Standard Earth gravitational parameter in m^3/s^2
MU_EARTH = 3.986004418e14 

def make_sat_arg_randomizer(mode="train", rso_att_type="near_velocity",max_error_deg=5):
    """
    Args:
        mode (str): "train" (randomize every reset) or "test" (persist RSO).
        rso_att_type (str): "random" (inertial), "velocity" (prograde), or "near_velocity" (perturbed prograde).
        max_error_deg (float): Maximum pointing error in degrees. Only used if rso_att_type is "near_velocity".
    """
    persistent_rso_state = {}

    def sat_arg_randomizer(satellites):
        nonlocal persistent_rso_state
        generate_new_rso = (mode == "train") or (not persistent_rso_state)

        if generate_new_rso:
            # 1. Generate Chief Orbit (in meters)
            a_meters = (R_EARTH*1000) + np.random.uniform(35776.0*1000, 35796.0*1000)
            e = np.random.uniform(0.0, 0.0005)
            chief_orbit = random_orbit(a=a_meters, e=e)

            # 2. Determine RSO Attitude
            if rso_att_type in ["velocity", "near_velocity"]:
                r_N, v_N = elem2rv(MU_EARTH, chief_orbit)
                # Create the Velocity-Aligned Frame
                # Unit 1: Prograde (Velocity direction)
                i_v = v_N / np.linalg.norm(v_N)
                
                # Unit 2: Orbit Normal (Cross-track)
                h_vec = np.cross(r_N, v_N)
                i_n = h_vec / np.linalg.norm(h_vec)
                
                # Unit 3: Completes the right-handed set (Radial-ish)
                i_b = np.cross(i_v, i_n)
                
                # DCM [VN]: Rows are the unit vectors of the V-frame in N-frame
                dcm_VN = np.array([i_v, i_n, i_b])
                
                # Apply Perturbation if "near_velocity"
                if rso_att_type == "near_velocity":
                    # 5% pointing error (approx 0.05 radians or ~2.86 degrees)
                    max_error_rad = np.radians(max_error_deg)
                    angle = np.random.uniform(0, max_error_rad)
                    
                    # Generate a random axis of rotation
                    axis = random_unit_vector()
                    
                    # Build Skew-Symmetric matrix K for the axis
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    
                    # Rodrigues' rotation formula to create the error DCM
                    dcm_err = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                    
                    # Apply the error rotation to the base V-bar DCM
                    dcm_VN = np.dot(dcm_err, dcm_VN)
                
                # Convert final DCM to MRP
                sigma_init = C2MRP(dcm_VN)
            
            else: # "random"
                u = np.random.uniform(0, 1, 3)
                q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
                q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
                q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
                q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
                
                sigma_init = np.array([q1, q2, q3]) / (1 + q0)
                if np.linalg.norm(sigma_init) > 1:
                    sigma_init = -sigma_init / (np.linalg.norm(sigma_init)**2)

            persistent_rso_state = {
                "chief_orbit": chief_orbit,
                "sigma_init": sigma_init,
                "omega_init": np.array([0.0, 0.0, 0.0])
            }

        # --- Retrieval and Assignment ---
        chief_orbit = persistent_rso_state["chief_orbit"]
        sigma_init = persistent_rso_state["sigma_init"]
        omega_init = persistent_rso_state["omega_init"]

        inspectors = [sat for sat in satellites if "Inspector" in sat.name]
        rso = [s for s in satellites if s.name == "RSO"][0]
        args = {}
        
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
        
        args[rso]["sigma_init"] = sigma_init
        args[rso]["omega_init"] = omega_init
        
        return args

    return sat_arg_randomizer