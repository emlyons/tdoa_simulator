import numpy as np
from .base import ParticleFilterBase
from scipy.stats import multivariate_normal

# Coordinated Turn with Velocity and Rate (CTVR) Particle Filter
# constant linear and angular velocity
class CTVR_PF(ParticleFilterBase):
    def __init__(self, particles, Q, R):
        super().__init__(particles)
        self.Q = Q # process noise covariance
        self.R = R # measurement noise covariance

    def process_model(self, dt):
        # for each particle, apply the process model to get the new particle state
        x, y, v, theta, omega = self.particles.T

        i_linear = np.where(omega < 1e6)
        i_nonlinear = np.where(omega >= 1e-6)

        x[i_nonlinear] += (v[i_nonlinear] / omega[i_nonlinear])*(np.cos(theta[i_nonlinear]) - np.cos(theta[i_nonlinear]+omega[i_nonlinear]*dt))
        y[i_nonlinear] += -(v[i_nonlinear] / omega[i_nonlinear])*(np.sin(theta[i_nonlinear]) - np.sin(theta[i_nonlinear]+omega[i_nonlinear]*dt))

        x[i_linear] += v[i_linear]*np.sin(theta[i_linear])*dt
        y[i_linear] += v[i_linear]*np.cos(theta[i_linear])*dt
        
        theta += omega*dt
        self.particles = np.stack((x, y, v, theta, omega), axis=1) + np.random.multivariate_normal(mean=np.zeros(5), cov=self.Q, size=len(self.particles)) # add process noise
        return self.particles

    def measurement_likelyhood(self, measurement):
        # for each particle... get particle positions x, y
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        p_measurement = np.zeros_like(x)
        for i, (_x, _y) in enumerate(zip(x, y)):
            p_measurement[i] = multivariate_normal.pdf([measurement], mean=[_x, _y], cov=self.R)
        return p_measurement
    