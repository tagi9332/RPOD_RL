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

def sat_arg_randomizer(satellites):
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
    
    mu = rso.sat_args_generator["mu"]
    r_N, v_N = elem2rv(mu, args[rso]["oe"])
    r_hat = r_N / np.linalg.norm(r_N)
    v_hat = v_N / np.linalg.norm(v_N)
    x = r_hat
    z = np.cross(r_hat, v_hat); z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    HN = np.array([x, y, z]); BH = np.eye(3)
    a = chief_orbit.a; T = np.sqrt(a**3 / mu) * 2 * np.pi # type: ignore
    omega_BN_N = z * 2 * np.pi / T
    args[rso]["sigma_init"] = C2MRP(BH @ HN)
    args[rso]["omega_init"] = BH @ HN @ omega_BN_N
    return args