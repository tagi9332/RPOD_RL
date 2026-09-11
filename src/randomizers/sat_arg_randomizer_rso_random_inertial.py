import numpy as np

from Basilisk.utilities.orbitalMotion import elem2rv
from Basilisk.utilities.RigidBodyKinematics import C2MRP
from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief

from resources import R_EARTH, MIN_REL_POS, MAX_REL_POS, MIN_REL_VEL, MAX_REL_VEL, INITIAL_APPROACH_VEL
from resources import dv_reward_weight as _default_dv_weight

MU_EARTH = 3.986004418e14  # m^3/s^2


class SatArgRandomizer:
    """
    Randomises RSO orbit and attitude and inspector relative state each episode.

    Attributes:
        mode:                 "train" (re-randomise every reset) or "test" (persist RSO).
        rso_att_type:         "random" | "velocity" | "anti_velocity" | "near_velocity" |
                              "rand_v_bar" | "near_rand_v_bar" |
                              "radial" | "anti_radial" | "normal" | "anti_normal".
        max_error_deg:        Maximum pointing error for "near_velocity" mode.
                              Mutable — updated by AttitudeErrorScheduler during training.
        fixed_inspector_state: Optional 6-element [rx,ry,rz,vx,vy,vz] in Hill frame.
    """

    # rso_att_type options:
    #   "random"          – uniformly random orientation
    #   "velocity"        – body-z along +v-bar
    #   "anti_velocity"   – body-z along -v-bar
    #   "near_velocity"   – body-z near +v-bar within max_error_deg
    #   "rand_v_bar"      – body-z randomly along ±v-bar each episode
    #   "near_rand_v_bar" – body-z randomly along ±v-bar with max_error_deg error
    #   "radial"          – body-z along +r-bar (radial)
    #   "anti_radial"     – body-z along -r-bar
    #   "normal"          – body-z along +h-bar (orbit normal)
    #   "anti_normal"     – body-z along -h-bar

    def __init__(
        self,
        mode: str = "train",
        rso_att_type: str = "near_velocity",
        max_error_deg: float = 5.0,
        fixed_inspector_state=None,
        dv_weight_mean: float | None = None,
        dv_weight_std: float = 0.05,
        dv_weight_min: float = 0.0,
        use_original_vel_dist: bool = True,
    ):
        self.mode = mode
        self.rso_att_type = rso_att_type
        self.max_error_deg = max_error_deg
        self.fixed_inspector_state = fixed_inspector_state
        self._persistent_rso_state: dict = {}
        self.use_original_vel_dist = use_original_vel_dist

        self.dv_weight_mean = dv_weight_mean
        self.dv_weight_std = dv_weight_std
        self.dv_weight_min = dv_weight_min
        # Holds the weight sampled for the current episode.
        self._dv_reward_weight: float = dv_weight_mean if dv_weight_mean is not None else _default_dv_weight

    def get_dv_weight(self) -> float:
        """Returns the dv penalty weight for the current episode."""
        return self._dv_reward_weight

    def __call__(self, satellites):
        if self.dv_weight_mean is not None and self.mode == "train":
            sampled = np.random.normal(self.dv_weight_mean, self.dv_weight_std)
            self._dv_reward_weight = max(self.dv_weight_min, sampled)

        generate_new_rso = (self.mode == "train") or (not self._persistent_rso_state)

        if generate_new_rso:
            a_meters = (R_EARTH * 1000) + np.random.uniform(35776.0 * 1000, 35796.0 * 1000)
            e = np.random.uniform(0.0, 0.0005)
            # random_orbit() expects `a` in km and internally does oe.a = a * 1e3;
            # a_meters is already in meters, so convert down or the semi-major axis
            # ends up 1000x too large (near-zero mean motion, degenerate Hill frame,
            # and observed runaway RW-acceleration warnings from the attitude FSW).
            chief_orbit = random_orbit(a=a_meters / 1000.0, e=e)

            _INERTIAL_ATT_MODES = {
                "velocity", "anti_velocity", "near_velocity",
                "rand_v_bar", "near_rand_v_bar",
                "radial", "anti_radial",
                "normal", "anti_normal",
            }

            if self.rso_att_type in _INERTIAL_ATT_MODES:
                r_N, v_N = elem2rv(MU_EARTH, chief_orbit)
                i_r = r_N / np.linalg.norm(r_N)
                i_v = v_N / np.linalg.norm(v_N)
                h_vec = np.cross(r_N, v_N)
                i_n = h_vec / np.linalg.norm(h_vec)

                if self.rso_att_type == "velocity":
                    body_z = i_v
                elif self.rso_att_type == "anti_velocity":
                    body_z = -i_v
                elif self.rso_att_type in ("near_velocity", "rand_v_bar", "near_rand_v_bar"):
                    sign = np.random.choice([-1, 1]) if self.rso_att_type in ("rand_v_bar", "near_rand_v_bar") else 1
                    body_z = sign * i_v
                elif self.rso_att_type == "radial":
                    body_z = i_r
                elif self.rso_att_type == "anti_radial":
                    body_z = -i_r
                elif self.rso_att_type == "normal":
                    body_z = i_n
                else:  # anti_normal
                    body_z = -i_n

                # body_y: use orbit normal for non-normal modes, radial for normal modes
                body_y = i_r if self.rso_att_type in ("normal", "anti_normal") else i_n
                body_x = np.cross(body_y, body_z)
                body_x = body_x / np.linalg.norm(body_x)
                dcm_VN = np.array([body_x, body_y, body_z])

                if self.rso_att_type in ("near_velocity", "near_rand_v_bar"):
                    max_error_rad = np.radians(self.max_error_deg)
                    angle = np.random.uniform(0, max_error_rad)
                    axis = random_unit_vector()
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ])
                    dcm_err = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    dcm_VN = dcm_err @ dcm_VN

                sigma_init = C2MRP(dcm_VN)

            else:  # "random" — uniformly distributed orientation
                u = np.random.uniform(0, 1, 3)
                q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
                q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
                q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
                q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
                sigma_init = np.array([q1, q2, q3]) / (1 + q0)
                if np.linalg.norm(sigma_init) > 1:
                    sigma_init = -sigma_init / (np.linalg.norm(sigma_init) ** 2)

            self._persistent_rso_state = {
                "chief_orbit": chief_orbit,
                "sigma_init": sigma_init,
                "omega_init": np.zeros(3),
            }

        chief_orbit = self._persistent_rso_state["chief_orbit"]
        sigma_init = self._persistent_rso_state["sigma_init"]
        omega_init = self._persistent_rso_state["omega_init"]

        rso = next(s for s in satellites if s.name == "RSO")
        inspectors = [s for s in satellites if "Inspector" in s.name]
        args = {}

        for inspector in inspectors:
            if self.fixed_inspector_state is not None:
                deputy_state_func = lambda: np.array(self.fixed_inspector_state)
            else:
                use_orig = self.use_original_vel_dist

                def deputy_state_func():
                    pos = random_unit_vector() * np.random.uniform(MIN_REL_POS, MAX_REL_POS)
                    if use_orig:
                        vel = random_unit_vector() * np.random.uniform(MIN_REL_VEL, MAX_REL_VEL)
                    else:
                        vel = -(pos / np.linalg.norm(pos)) * INITIAL_APPROACH_VEL
                    return np.concatenate((pos, vel))

            relative_randomizer = relative_to_chief(
                chief_name="RSO",
                chief_orbit=chief_orbit,
                deputy_relative_state={inspector.name: deputy_state_func},
            )
            args.update(relative_randomizer([rso, inspector]))

        args[rso]["sigma_init"] = sigma_init
        args[rso]["omega_init"] = omega_init
        return args


def make_sat_arg_randomizer(
    mode: str = "train",
    rso_att_type: str = "near_velocity",
    max_error_deg: float = 5.0,
    fixed_inspector_state=None,
    dv_weight_mean: float | None = None,
    dv_weight_std: float = 0.05,
    dv_weight_min: float = 0.0,
    use_original_vel_dist: bool = False,
) -> SatArgRandomizer:
    """Factory — returns a SatArgRandomizer instance (callable, mutable)."""
    return SatArgRandomizer(
        mode=mode,
        rso_att_type=rso_att_type,
        max_error_deg=max_error_deg,
        fixed_inspector_state=fixed_inspector_state,
        dv_weight_mean=dv_weight_mean,
        dv_weight_std=dv_weight_std,
        dv_weight_min=dv_weight_min,
        use_original_vel_dist=use_original_vel_dist,
    )
