import numpy as np
from Basilisk.utilities.RigidBodyKinematics import MRP2C, C2MRP

def custom_r_DC_C(deputy, chief):
    """
    Relative position of the Deputy to the Chief, expressed in the Chief's body frame.
    """
    r_DC_N = np.array(deputy.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)
    CN = MRP2C(chief.dynamics.sigma_BN)
    return CN @ r_DC_N


def custom_v_DC_C(deputy, chief):
    """
    Relative velocity of the Deputy w.r.t. the Chief, expressed in the Chief's body frame.

    Uses the inertial relative velocity rotated into the Chief body frame.  The
    full expression also includes a Coriolis term (omega_BN x r_DC_N) from the
    Chief's rotation, which is omitted here.  For slowly-tumbling GEO RSOs this
    approximation error is small (<0.03 m/s for |omega| < 0.001 rad/s, |r| < 30 m).
    """
    v_DC_N = np.array(deputy.dynamics.v_BN_N) - np.array(chief.dynamics.v_BN_N)
    CN = MRP2C(chief.dynamics.sigma_BN)
    return CN @ v_DC_N


def make_dist_to_waypoint_fn(standoff_distance: float, boresight: np.ndarray):
    """
    Factory: returns an observation function that gives the scalar distance
    from the deputy to the body-fixed standoff waypoint.

    Returns a 1-element array so BSK-RL's RelativeProperties can handle it.
    At Phase 0 start (~1500 m away) this is ~1500 m; at the waypoint it is 0.
    Normalise by MAX_REL_POS in the observation spec for a [0, ~1] range.
    """
    waypoint_B = np.asarray(boresight, dtype=float) * standoff_distance

    def dist_to_waypoint(deputy, chief):
        r_DC_N = np.array(deputy.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)
        CN = MRP2C(chief.dynamics.sigma_BN)
        r_DC_C = CN @ r_DC_N
        return np.array([np.linalg.norm(r_DC_C - waypoint_B)])

    return dist_to_waypoint
