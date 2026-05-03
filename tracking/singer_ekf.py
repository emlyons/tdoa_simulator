import numpy as np
from .base import EKFBase

# Coordinated Turn with Velocity and Rate (CTVR) EKF
# constant linear and angular velocity
class SINGER_EKF(EKFBase):
    def __init__(self, state, R, tau):
        P = np.eye(6) * 1e-2  # State covariance matrix
        Q = np.eye(6) * 1e-2  # Process noise covariance
        # R = np.eye(3) * 0.5  # Measurement noise covariance
        I = np.eye(6)
        self.tau = tau
        super().__init__(state, P, Q, R, I)

    def process_model_jacobian(self, dt):
        a = 1/self.tau
        F = [[ 1,   dt, (a*dt - 1 + np.exp(-a*dt))/(a**2), 0,   0,                    0                 ],
             [ 0,    1,        (1 - np.exp(-a*dt))/a     , 0,   0,                    0                 ],
             [ 0,    0,            np.exp(-a*dt)         , 0,   0,                    0                 ],
             [ 0,    0,            0                     , 1,   dt,   (a*dt - 1 + np.exp(-a*dt))/(a**2) ],
             [ 0,    0,            0                     , 0,   1,           (1 - np.exp(-a*dt))/a      ],
             [ 0,    0,            0                     , 0,   0,               np.exp(-a*dt)          ]]

        return np.array(F)

    def process_model(self, dt):
        x, _, _, y, _, _ = self.state
        F = self.process_model_jacobian(dt)
        state = F @ self.state
        return state

    def measurement_model_jacobian(self):
        x = float(self.state[0, 0])  # scalar
        y = float(self.state[3, 0])  # scalar (state = [x, vx, ax, y, vy, ay]^T)

        r_1 = np.linalg.norm(np.array([x, y]) - self.s_1)
        r_2 = np.linalg.norm(np.array([x, y]) - self.s_2)
        r_3 = np.linalg.norm(np.array([x, y]) - self.s_3)
        r_ref = np.linalg.norm(np.array([x, y]) - self.s_ref)

        H = np.array([
            [(x - self.s_1[0]) / r_1 - (x - self.s_ref[0]) / r_ref, 0.0, 0.0,
             (y - self.s_1[1]) / r_1 - (y - self.s_ref[1]) / r_ref, 0.0, 0.0],
            [(x - self.s_2[0]) / r_2 - (x - self.s_ref[0]) / r_ref, 0.0, 0.0,
             (y - self.s_2[1]) / r_2 - (y - self.s_ref[1]) / r_ref, 0.0, 0.0],
            [(x - self.s_3[0]) / r_3 - (x - self.s_ref[0]) / r_ref, 0.0, 0.0,
             (y - self.s_3[1]) / r_3 - (y - self.s_ref[1]) / r_ref, 0.0, 0.0],
        ], dtype=float)
        H = H / self.c
        return H
    
    def measurement_model(self):
        s = np.asarray(self.state, dtype=float).reshape(-1)
        x, y = s[0], s[3]
        pos = np.array([x, y], dtype=float)

        d_ref = np.linalg.norm(pos - np.asarray(self.s_ref, dtype=float))
        dt_s1 = (1.0 / self.c) * (np.linalg.norm(pos - np.asarray(self.s_1, dtype=float)) - d_ref)
        dt_s2 = (1.0 / self.c) * (np.linalg.norm(pos - np.asarray(self.s_2, dtype=float)) - d_ref)
        dt_s3 = (1.0 / self.c) * (np.linalg.norm(pos - np.asarray(self.s_3, dtype=float)) - d_ref)

        return np.array([[dt_s1], [dt_s2], [dt_s3]], dtype=float)
    