import numpy as np
from Basilisk.utilities.RigidBodyKinematics import MRP2C, C2MRP


def boresight_in_hill(deputy, chief):
    """RSO docking boresight (body-z) expressed in the Hill frame.

    Gives the agent explicit knowledge of which direction to approach from
    in the frame it acts in, enabling generalization across arbitrary RSO attitudes.
    Returns a unit vector — normalize by 1.0.
    """
    CN = MRP2C(np.array(chief.dynamics.sigma_BN))
    boresight_N = CN.T @ np.array([0.0, 0.0, 1.0])
    HN = np.array(chief.dynamics.HN)
    return HN @ boresight_N


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
    Factory: returns an observation function giving the signed distance along
    the docking-port boresight axis relative to the standoff waypoint.

    Value = (r_DC_C · boresight) - standoff_distance:
      > 0  : inspector is ahead of the waypoint on the correct approach side
      = 0  : inspector is at the waypoint along the boresight axis
      < 0  : inspector is behind the waypoint (wrong-side / overshoot)

    Returns a 1-element array. Normalise by MAX_REL_POS for a [-1, +1] range.
    """
    boresight_hat = np.asarray(boresight, dtype=float)
    boresight_hat = boresight_hat / np.linalg.norm(boresight_hat)

    def boresight_signed_dist(deputy, chief):
        r_DC_N = np.array(deputy.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)
        CN = MRP2C(chief.dynamics.sigma_BN)
        r_DC_C = CN @ r_DC_N
        return np.array([float(np.dot(r_DC_C, boresight_hat)) - standoff_distance])

    return boresight_signed_dist
