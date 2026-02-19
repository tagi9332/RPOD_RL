import numpy as np

def MRP2C(sigma):
    """Converts Modified Rodrigues Parameters to a Direction Cosine Matrix (Rotation Matrix)."""
    sigma = np.array(sigma)
    s_sq = np.dot(sigma, sigma)
    if s_sq > 1.0: # Check for shadow set switching if needed, though BSK handles this usually
        pass 
        
    skew_s = np.array([
        [0, -sigma[2], sigma[1]],
        [sigma[2], 0, -sigma[0]],
        [-sigma[1], sigma[0], 0]
    ])
    
    # MRP Rotation Matrix Formula
    return np.eye(3) + (8 * np.dot(skew_s, skew_s) - 4 * (1 - s_sq) * skew_s) / ((1 + s_sq)**2)