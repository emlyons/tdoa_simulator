import numpy as np
from .base import UKFBase

# Coordinated Turn with Velocity and Rate (CTVR) UKF
# constant linear and angular velocity
class CTVR_UKF(UKFBase):
    def __init__(self, state, P=None, Q=None, R=None, alpha=1e-3, beta=2, kappa=0):
        if P is None:
            P = np.eye(5) * 1e-2  # State covariance matrix
        if Q is None:
            Q = np.eye(5) * 1e-2  # Process noise covariance
        if R is None:
            R = np.eye(2) * 0.5  # Measurement noise covariance
        super().__init__(state, P, Q, R, alpha=alpha, beta=beta, kappa=kappa)


    def process_model(self, state_aug, dt):
        x, y, v, theta, omega = state_aug[:5]
    
        # linear motion case (omega ~ 0)
        if abs(omega) < 1e-6:
            x += v*np.sin(theta)*dt
            y += v*np.cos(theta)*dt
        
        # non linear motion case (|omega| > 0)
        else:
            x += (v / omega)*(np.cos(theta) - np.cos(theta+omega*dt))
            y += -(v / omega)*(np.sin(theta) - np.sin(theta+omega*dt))

        theta += omega*dt
        state = np.array([x + state_aug[5], y + state_aug[6], v + state_aug[7],
                  theta + state_aug[8], omega + state_aug[9],
                  state_aug[5], state_aug[6], state_aug[7], state_aug[8], state_aug[9]])
        return state

    def measurement_model(self, state):
        x, y, v, theta, omega, e_x, e_y = state
        return np.array([x + e_x, y + e_y])
    