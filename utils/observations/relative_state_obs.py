import numpy as np
from Basilisk.utilities.RigidBodyKinematics import MRP2C, C2MRP

def custom_r_DC_C(deputy, chief):
    """
    Relative position of the Deputy to the Chief, expressed in the Chief's body frame.
    """
    # 1. Inertial relative position: r_DC_N
    r_DC_N = np.array(deputy.dynamics.r_BN_N) - np.array(chief.dynamics.r_BN_N)
    
    # 2. Chief Inertial to Body DCM: [CN]
    CN = MRP2C(chief.dynamics.sigma_BN)
    
    # 3. Rotate inertial position into Chief body frame: r_DC_C = [CN] * r_DC_N
    return CN @ r_DC_N