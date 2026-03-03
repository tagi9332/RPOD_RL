import numpy as np
from Basilisk.utilities.RigidBodyKinematics import MRP2C, C2MRP

def custom_sigma_DC(deputy, chief):
    """
    Relative attitude of the Deputy satellite to the Chief satellite.
    Returns the MRPs representing the rotation from Chief body to Deputy body.
    """
    # Get DCMs from Inertial (N) to Body (B)
    CN = MRP2C(chief.dynamics.sigma_BN)
    DN = MRP2C(deputy.dynamics.sigma_BN)
    
    # Calculate relative DCM: [DC] = [DN][CN]^T
    DC = DN @ CN.T
    
    # Convert relative DCM back to an MRP array
    return C2MRP(DC)
