"""
Stage-aware satellite argument randomizer for 3-stage reverse curriculum.

Stage 0  (terminal):    Inspector 20-24 m from docking port along the boresight
                        axis. All inits land inside the 10 m capture sphere so
                        Phase 1 activates by the second step.

Stage 1  (capture):     Inspector 65-615 m from docking port, strictly on the
                        +boresight side. Trains Phase 0 deceleration and waypoint
                        capture leading into the Stage 0 behaviour.

Stage 2  (full):        Full random 1800-2000 m, any direction. Matches the
                        production training distribution.

Body → Hill frame (velocity-aligned RSO, exact for Stages 0/1):
    DCM_HB = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    r_Hill = [-body_x, body_z, body_y]

    body_x = -Hill_x  (negated radial)
    body_y =  Hill_z  (orbit normal)
    body_z =  Hill_y  (along-track / velocity = docking boresight)
"""
import numpy as np
from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief

from resources import R_EARTH, MIN_REL_POS, MAX_REL_POS, MIN_REL_VEL, MAX_REL_VEL, INITIAL_APPROACH_VEL

MU_EARTH = 3.986004418e14  # m^3/s^2

# z-axis (boresight) range and lateral scatter per stage.
# Stage 0: z=20-24 m keeps all inits inside the 10 m capture sphere
#          (waypoint at z=15, so min dist = 5 m, max dist ≈ 9.95 m).
# Stage 1: z=65-615 m — 50-600 m past the waypoint on the correct side.
_STAGE_PARAMS = {
    0: {"z_range": (20.0, 24.0),  "lat_range": 3.0,  "vel_range": 0.02},
    1: {"z_range": (65.0, 615.0), "lat_range": 10.0, "vel_range": 0.05},
}


class CurriculumSatArgRandomizer:
    """
    Set ``.stage`` before each episode reset; CurriculumSb3BksEnv.reset()
    does this automatically from the curriculum callback.

    Stages 0/1 force exact velocity alignment (rso_att_type="velocity") so
    the body→Hill transform is exact. Stage 2 uses the configured rso_att_type.
    """

    def __init__(
        self,
        stage: int = 0,
        mode: str = "train",
        rso_att_type: str = "near_velocity",
        max_error_deg: float = 5.0,
        use_original_vel_dist: bool = True,
    ):
        self.stage = stage
        self.mode = mode
        self.rso_att_type = rso_att_type
        self.max_error_deg = max_error_deg
        self.use_original_vel_dist = use_original_vel_dist
        self._persistent_rso_state: dict = {}

    def _sample_rso_state(self) -> dict:
        a = (R_EARTH * 1000) + np.random.uniform(35_776e3, 35_796e3)
        e = np.random.uniform(0.0, 0.0005)
        chief_orbit = random_orbit(a=a, e=e)

        r_N, v_N = elem2rv(MU_EARTH, chief_orbit)

        # Stages 0/1: exact velocity alignment for clean body→Hill conversion.
        att_type = "velocity" if self.stage < 2 else self.rso_att_type

        if att_type in ("velocity", "near_velocity", "anti_velocity"):
            i_v = (-v_N if att_type == "anti_velocity" else v_N)
            i_v = i_v / np.linalg.norm(i_v)
            h = np.cross(r_N, v_N)
            i_n = h / np.linalg.norm(h)
            body_z = i_v
            body_y = i_n
            body_x = np.cross(body_y, body_z)
            dcm_VN = np.array([body_x, body_y, body_z])

            if att_type == "near_velocity":
                max_rad = np.radians(self.max_error_deg)
                angle = np.random.uniform(0, max_rad)
                axis = random_unit_vector()
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ])
                dcm_err = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                dcm_VN = dcm_err @ dcm_VN

            sigma_init = C2MRP(dcm_VN)

        else:  # random attitude
            u = np.random.uniform(0, 1, 3)
            q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
            q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
            q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
            q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
            sigma_init = np.array([q1, q2, q3]) / (1 + q0)
            if np.linalg.norm(sigma_init) > 1:
                sigma_init = -sigma_init / np.linalg.norm(sigma_init) ** 2

        return {
            "chief_orbit": chief_orbit,
            "sigma_init": sigma_init,
            "omega_init": np.zeros(3),
        }

    def __call__(self, satellites):
        if (self.mode == "train") or (not self._persistent_rso_state):
            self._persistent_rso_state = self._sample_rso_state()

        chief_orbit = self._persistent_rso_state["chief_orbit"]
        sigma_init  = self._persistent_rso_state["sigma_init"]
        omega_init  = self._persistent_rso_state["omega_init"]

        rso = next(s for s in satellites if s.name == "RSO")
        inspectors = [s for s in satellites if "Inspector" in s.name]
        args = {}

        for inspector in inspectors:
            stage = self.stage

            if stage < 2:
                p = _STAGE_PARAMS[stage]

                def _hill_state(params=p):
                    bz  = np.random.uniform(*params["z_range"])
                    lat = np.random.uniform(-params["lat_range"], params["lat_range"], 2)
                    # body [lat[0], lat[1], bz] → Hill [-lat[0], bz, lat[1]]
                    r_H = np.array([-lat[0], bz, lat[1]])
                    v_H = np.random.uniform(-params["vel_range"], params["vel_range"], 3)
                    return np.concatenate([r_H, v_H])

                deputy_fn = _hill_state

            else:
                use_orig = self.use_original_vel_dist

                def deputy_fn():
                    r = random_unit_vector() * np.random.uniform(MIN_REL_POS, MAX_REL_POS)
                    if use_orig:
                        v = random_unit_vector() * np.random.uniform(MIN_REL_VEL, MAX_REL_VEL)
                    else:
                        v = -r / np.linalg.norm(r) * INITIAL_APPROACH_VEL
                    return np.concatenate([r, v])

            rel_rand = relative_to_chief(
                chief_name="RSO",
                chief_orbit=chief_orbit,
                deputy_relative_state={inspector.name: deputy_fn},
            )
            args.update(rel_rand([rso, inspector]))

        args[rso]["sigma_init"] = sigma_init
        args[rso]["omega_init"] = omega_init
        return args
