import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

## Make Two Motions Modles
    # For Each Motion Model,
        # Use TOA position measurements, track with EKF

        # Use TDOA position measurements, track with EKF, UKF, PMF or Particle Filter for the two motion models
class KalmanBase:
    def __init__(self, state, P, Q, R):
        self.state = state
        self.P = P
        self.Q = Q
        self.R = R

    def predict(self, dt):
        raise NotImplementedError("predict must be implemented by subclass")

    def update(self, dt):
        raise NotImplementedError("update must be implemented by subclass")

    def get_state(self):
        return self.state

class EKFBase(KalmanBase):
    def __init__(self, state, P, Q, R, I):
        self.I = I
        super().__init__(state, P, Q, R)

    def predict(self, dt):
        F = self.process_model_jacobian(dt)
        self.state = self.process_model(dt)
        self.P = F @ self.P @ F.T + F @ self.Q @ F.T
        return

    def update(self, measurement):
        H = self.measurement_model_jacobian()
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        innovation = measurement - self.measurement_model()
        self.state = self.state + K @ innovation
        self.P = (self.I - K @ H) @ self.P @ (self.I - K @ H).T + K @ self.R @ K.T
        return

    def process_model_jacobian(self, dt):
        raise NotImplementedError("process_model_jacobian must be implemented by subclass")

    def process_model(self, dt):
        raise NotImplementedError("process_model must be implemented by subclass")

    def measurement_model_jacobian(self):
        raise NotImplementedError("measurement_model_jacobian must be implemented by subclass")

    def measurement_model(self):
        raise NotImplementedError("measurement_model must be implemented by subclass")


class EKF2Base(KalmanBase):
    def __init__(self, state, P, Q, R):
        super().__init__(state, P, Q, R)

    def predict(self, dt):
        Fp = self.process_model_jacobian(dt)
        Fpp = self.process_model_hessian(dt)
        self.state = self.process_model(dt) + 0.5 * np.einsum('kij,ij->k', Fpp, self.P).reshape(self.state.shape) # tr(Fpp[k] @ P)
        M = np.einsum('kij,jl->kil', Fpp, self.P)  # M[k] = Fpp[k] @ P
        dP = 0.5 * np.einsum('kij,lji->kl', M, M)  # dP[k,l] = 0.5 * tr(M[k] @ M[l])
        self.P = Fp @ self.P @ Fp.T + Fp @ self.Q @ Fp.T + dP
        return

    def update(self, measurement):
        Hp = self.measurement_model_jacobian()
        Hpp = self.measurement_model_hessian()

        M = np.einsum('kij,jl->kil', Hpp, self.P)  # M[k] = Hpp[k] @ P
        dH = 0.5 * np.einsum('kij,lji->kl', M, M)  # dH[k,l] = 0.5 * tr(M[k] @ M[l])
        S = Hp @ self.P @ Hp.T + self.R + dH
        K = self.P @ Hp.T @ np.linalg.inv(S)

        pred_meas = self.measurement_model()
        innovation = measurement - pred_meas - 0.5 * np.einsum('kij,ij->k', Hpp, self.P).reshape(pred_meas.shape) # tr(Hpp[k] @ P)
        self.state = self.state + K @ innovation
        
        self.P = self.P - self.P @ Hp.T @ np.linalg.inv(S) @ Hp @ self.P
        return

    def process_model_jacobian(self, dt):
        raise NotImplementedError("process_model_jacobian must be implemented by subclass")

    def process_model_hessian(self, dt):
        raise NotImplementedError("process_model_hessian must be implemented by subclass")

    def process_model(self, dt):
        raise NotImplementedError("process_model must be implemented by subclass")

    def measurement_model_jacobian(self):
        raise NotImplementedError("measurement_model_jacobian must be implemented by subclass")
    
    def measurement_model_hessian(self):
        raise NotImplementedError("measurement_model_hessian must be implemented by subclass")

    def measurement_model(self):
        raise NotImplementedError("measurement_model must be implemented by subclass")


class UKFBase(KalmanBase):
    def __init__(self, state, P, Q, R, alpha=1e3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        super().__init__(state, P, Q, R)

    def predict(self, dt):

        # augmented state space allows both P and Q to be captured by the sigma points.
        # without augmenting the covariance used for sigma pointswould only be P...
        state_aug = np.concatenate((self.state, np.zeros_like(self.state))) # augment state vector
        cov_aug = np.block([[self.P, np.zeros_like(self.P)], [np.zeros_like(self.Q), self.Q]]) # augment covariance matrix
        
        S, Wm, Wc = self.sigma_points(state_aug, cov_aug)# generate sigma points and weights from state and P

        # apply process model to each sigma
        # apply weights to get predicted state and P and combine with Q
        state_aug = np.zeros_like(state_aug)
        cov_aug = np.zeros_like(cov_aug)

        S_est = [self.process_model(s, dt) for s in S]

        for s, wm in zip(S_est, Wm):
            state_aug += wm * s

        for s, wc in zip(S_est, Wc):
            cov_aug += wc * np.outer(s - state_aug, s - state_aug)

        self.state = state_aug[:self.state.shape[0]]
        self.P = cov_aug[:self.P.shape[0], :self.P.shape[1]]
        return

    def update(self, measurement):
        n = self.state.shape[0]
        m = measurement.shape[0]

        # augmented state space allows both P and R to be captured by the sigma points.
        # without augmenting the covariance used for sigma points would only be R...
        state_aug = np.concatenate((self.state, np.zeros(m))) # augment state vector
        cov_aug = np.block([[self.P, np.zeros((n, m))], [np.zeros((m, n)), self.R]]) # augment covariance matrix

        # generate sigma points
        S, Wm, Wc = self.sigma_points(state_aug, cov_aug)
        Y = [self.measurement_model(s) for s in S]

        S = S[:, :n] # only the state part of the sigma points is relevant for the update

        Y_est = np.zeros_like(measurement)
        for y, wm in zip(Y, Wm):
            Y_est += wm * y

        P_xy = np.zeros((n, m))
        for (s, y, wc) in zip(S, Y, Wc):
            P_xy += wc * np.outer(s - self.state, y - Y_est)

        P_yy = np.zeros((m, m))
        for (y, wc) in zip(Y, Wc):
            P_yy += wc * np.outer(y - Y_est, y - Y_est)

        self.state += (P_xy @ np.linalg.inv(P_yy) @ (measurement - Y_est))
        self.P -= (P_xy @ np.linalg.inv(P_yy) @ P_xy.T)
        return

    def _lambda(self, n):
        return self.alpha**2 * (n + self.kappa) - n

    def sigma_points(self, state_aug, cov_aug):
        n = state_aug.shape[0]
        X = np.zeros((2 * n + 1, n))
        Wm = np.zeros((2 * n + 1))
        Wc = np.zeros((2 * n + 1))
        l = self._lambda(n)
        L = np.linalg.cholesky(cov_aug)

        X[0] = state_aug
        Wm[0] = l / (n + l)
        Wc[0] = l / (n + l) + (1 - self.alpha**2 + self.beta)

        for i in range(1,n+1): # 1 to n
            # each sigma point lies in an orthogonal direction on the error ellipse
            X[i] = state_aug + np.sqrt(n + l) * L[:, i-1]
            Wm[i] = 1 / (2 * (n + l))
            Wc[i] = 1 / (2 * (n + l))

        for i in range(n+1,2*n+1): # n+1 to 2*n
            # sample in the opposite direction to capture negative deviations, the ellipse is two sided
            X[i] = state_aug - np.sqrt(n + l) * L[:, i-n-1]
            Wm[i] = 1 / (2 * (n + l))
            Wc[i] = 1 / (2 * (n + l))

        return X, Wm, Wc
    
    def process_model(self, state_aug, dt) -> np.array:
        raise NotImplementedError("process_model must be implemented on augmented space by subclass")
    
    def measurement_model(self, state):
        raise NotImplementedError("measurement_model must be implemented by subclass")
    
    

class ParticleFilterBase:
    def __init__(self, particles):
        self.particles = particles
        self.weights = np.ones(len(self.particles)) / len(self.particles)
        self.neff_threshold = len(self.particles) / 3
        
    def update(self, measurement, dt):
        # 1. measurement update
        self.weights = self.measurement_likelyhood(measurement) * self.weights # W_i_k+1 = P(measurement | particle_i) * W_i_k <- measurement model
        
        # 2. normalize weights
        self.weights /= np.sum(self.weights) # normalize updated weights

        # 3. estimate
        x_est = self.weights @ self.particles # estimate x_est = sum(W_i_k+1 * x_i_k) <- state estimation

        # 4. resample
        # only resample if Neff is below threshold (particle depletion)
        if not self.n_eff_status():
            indices = np.random.choice(len(self.particles), len(self.particles), p=self.weights)
            self.particles = self.particles[indices] # x_i_k resample particles based on weights
            self.weights = np.ones(len(self.particles)) / len(self.particles) # reset weights after resampling
        
        # 5. process model
        self.particles = self.process_model(dt) # x_i_k+1 = f(x_i_k) <- process model
        return x_est

    def n_eff_status(self):
        return (1 / np.sum(self.weights**2)) > self.neff_threshold

    def process_model(self, dt):
        raise NotImplementedError("process_model must be implemented by subclass")

    def measurement_likelyhood(self, measurement):
        raise NotImplementedError("measurement_likelyhood must be implemented by subclass")
    