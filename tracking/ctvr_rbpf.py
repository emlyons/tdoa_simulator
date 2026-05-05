import numpy as np
from .base import MargenalizedParticleFilterBase
from scipy.stats import multivariate_normal

# Coordinated Turn with Velocity and Rate (CTVR) Particle Filter
# constant linear and angular velocity
class CTVR_RBPF(MargenalizedParticleFilterBase):
    def __init__(self, particles):
        super().__init__(particles)
        self.Q_l = np.eye(2) * 0.2 # process noise covariance
        self.Q_nl = np.eye(3) * 0.2 
        self.R = np.eye(2) * 0.2 # measurement noise covariance
        self.F = np.eye(2) # measurement matrix for linear update
        self.H = np.eye(2) # measurement matrix for non linear update

    def process_model_linear(self, dt):
        x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11 = self.particles.T
        N = len(x)

        # Batch state prediction: (N, 2) @ (2, 2).T
        predicted = np.stack([u_v, u_dw], axis=1) @ self.F.T  # (N, 2)
        v, dw = predicted[:, 0], predicted[:, 1]

        # Batch covariance update: P_i = F @ P_i @ F.T + Q
        # Build (N, 2, 2) batch of diagonal covariances
        P_batch = np.zeros((N, 2, 2))
        P_batch[:, 0, 0] = P00
        P_batch[:, 0, 1] = P01
        P_batch[:, 1, 0] = P10
        P_batch[:, 1, 1] = P11

        # einsum does F @ P_i @ F.T for all N particles simultaneously
        P_batch = np.einsum('ij,njk,lk->nil', self.F, P_batch, self.F) + self.Q_l

        P00 = P_batch[:, 0, 0]
        P01 = P_batch[:, 0, 1]
        P10 = P_batch[:, 1, 0]
        P11 = P_batch[:, 1, 1]
        self.particles = np.stack((x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11), axis=1)   

    def process_model_non_linear(self, dt):
        # for each particle, apply the process model to get the new particle state
        x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11 = self.particles.T

        i_linear = np.where(dw < 1e-6)
        i_nonlinear = np.where(dw >= 1e-6)

        x[i_nonlinear] += (v[i_nonlinear] / dw[i_nonlinear])*(np.cos(w[i_nonlinear]) - np.cos(w[i_nonlinear]+dw[i_nonlinear]*dt))
        y[i_nonlinear] += -(v[i_nonlinear] / dw[i_nonlinear])*(np.sin(w[i_nonlinear]) - np.sin(w[i_nonlinear]+dw[i_nonlinear]*dt))

        x[i_linear] += v[i_linear]*np.sin(w[i_linear])*dt
        y[i_linear] += v[i_linear]*np.cos(w[i_linear])*dt
        
        w += dw*dt

        x += np.random.normal(0, np.sqrt(self.Q_nl[0,0]), size=len(x))
        y += np.random.normal(0, np.sqrt(self.Q_nl[1,1]), size=len(y))
        w += np.random.normal(0, np.sqrt(self.Q_nl[2,2]), size=len(w)) # add process noise to non linear part only

        self.particles = np.stack((x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11), axis=1) # add process noise to non linear part only
        return self.particles
    
    def linear_update(self, measurement):
        x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11 = self.particles.T
        N = len(x)

        # Per-particle H: (N, 2, 2) — maps [v, ω] → [vx, vy] prediction
        H = np.zeros((N, 2, 2))
        H[:, 0, 0] = np.sin(w)
        H[:, 1, 0] = np.cos(w)

        # Build batch P: (N, 2, 2)
        P_batch = np.empty((N, 2, 2))
        P_batch[:, 0, 0] = P00
        P_batch[:, 0, 1] = P01
        P_batch[:, 1, 0] = P10
        P_batch[:, 1, 1] = P11

        # S = H_i @ P_i @ H_i.T + R -> (N, 2, 2)
        S = np.einsum('nij,njk,nlk->nil', H, P_batch, H) + self.R

        # K = P_i @ H_i.T @ S^-1 -> (N, 2, 2)
        PHt = np.einsum('nij,nkj->nik', P_batch, H)  # (N, 2, 2)
        S_inv = np.linalg.inv(S)                       # (N, 2, 2)
        K = np.einsum('nij,njk->nik', PHt, S_inv)      # (N, 2, 2)

        # Innovation: z - H_i @ [u_v, u_dw] per particle -> (N, 2)
        pred = np.stack([u_v, u_dw], axis=1)                    # (N, 2)
        predicted_meas = np.einsum('nij,nj->ni', H, pred)       # (N, 2)
        innovation = measurement[np.newaxis, :] - predicted_meas # (N, 2)

        # State update: [u_v, u_dw] += K @ innovation
        u_state = np.stack([u_v, u_dw], axis=1)
        u_state += np.einsum('nij,nj->ni', K, innovation)
        u_v = u_state[:, 0]
        u_dw = u_state[:, 1]

        # Covariance update: P = (I - K @ H_i) @ P
        KH = np.einsum('nij,njk->nik', K, H)
        I_KH = np.eye(2)[np.newaxis, :, :] - KH
        P_batch = np.einsum('nij,njk->nik', I_KH, P_batch)

        P00 = P_batch[:, 0, 0]
        P01 = P_batch[:, 0, 1]
        P10 = P_batch[:, 1, 0]
        P11 = P_batch[:, 1, 1]

        self.particles = np.stack((x, y, w, v, dw, u_v, u_dw, P00, P01, P10, P11), axis=1)

    def measurement_likelyhood_non_linear(self, measurement):
        # for each particle... get particle positions x, y
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        p_measurement = np.zeros_like(x)
        for i, (_x, _y) in enumerate(zip(x, y)):
            p_measurement[i] = multivariate_normal.pdf([measurement], mean=[_x, _y], cov=self.R)
        return p_measurement