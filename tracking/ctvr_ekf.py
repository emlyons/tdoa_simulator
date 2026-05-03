import numpy as np
from .base import EKFBase

# Coordinated Turn with Velocity and Rate (CTVR) EKF
# constant linear and angular velocity
class CTVR_EKF(EKFBase):
    def __init__(self, state):
        P = np.eye(5) * 1e-2  # State covariance matrix
        Q = np.eye(5) * 1e-2  # Process noise covariance
        R = np.eye(2) * 0.5  # Measurement noise covariance
        I = np.eye(5)
        super().__init__(state, P, Q, R, I)

    def process_model_jacobian(self, dt):
        x, y, v, theta, omega = self.state
        F = np.eye(5)

        # linear motion case
        if abs(omega) < 1e-6:
            F[0,2] = np.sin(theta)*dt
            F[0,3] = v*np.cos(theta)*dt
            F[1,2] = np.cos(theta)*dt
            F[1,3] = -v*np.sin(theta)*dt
            self.F = F

        # non linear motion case
        else:
            theta_new = theta + omega*dt

            # ∂X_new/∂V = (1/W)[cos(T) - cos(T+W*dt)]
            F[0,2] = (1/omega) * (np.cos(theta) - np.cos(theta_new))

            # ∂X_new/∂T = (V/W)[-sin(T) + sin(T+W*dt)]
            F[0,3] = (v/omega) * (-np.sin(theta) + np.sin(theta_new))

            # ∂X_new/∂W = -(V/W²)[cos(T) - cos(T+W*dt)] + (V/W)*sin(T+W*dt)*dt
            F[0,4] = -(v/omega**2) * (np.cos(theta) - np.cos(theta_new)) + \
                    (v/omega) * np.sin(theta_new) * dt

            # ∂Y_new/∂V = (1/W)[-sin(T) + sin(T+W*dt)]
            F[1,2] = (1/omega) * (-np.sin(theta) + np.sin(theta_new))

            # ∂Y_new/∂T = (V/W)[-cos(T) + cos(T+W*dt)]
            F[1,3] = (v/omega) * (-np.cos(theta) + np.cos(theta_new))

            # ∂Y_new/∂W = -(V/W²)[-sin(T) + sin(T+W*dt)] + (V/W)*cos(T+W*dt)*dt
            F[1,4] = -(v/omega**2) * (-np.sin(theta) + np.sin(theta_new)) + \
                    (v/omega) * np.cos(theta_new) * dt

            # ∂T_new/∂W = dt
            F[3,4] = dt

        return F

    def process_model(self, dt):
        x, y, v, theta, omega = self.state
    
        # linear motion case (omega ~ 0)
        if abs(omega) < 1e-6:
            x += v*np.sin(theta)*dt
            y += v*np.cos(theta)*dt
        
        # non linear motion case (|omega| > 0)
        else:
            x += (v / omega)*(np.cos(theta) - np.cos(theta+omega*dt))
            y += -(v / omega)*(np.sin(theta) - np.sin(theta+omega*dt))

        theta += omega*dt
        state = np.array([x, y, v, theta, omega])
        return state

    def measurement_model_jacobian(self):
        H = np.array([[ 1,  0,  0,  0,  0 ],
                      [ 0,  1,  0,  0,  0 ]])
        return H
    
    def measurement_model(self):
        x, y, v, theta, omega = self.state
        return np.array([x, y])
