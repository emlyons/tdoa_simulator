import numpy as np
from simulator.utils import SPEED_OF_SOUND
from simulator.types import Point3D

def toa_std(toa):
    # std = sum (toa[i] - toa[i-1] - mean)^2 / (N-1)
    pulse_period = np.diff(toa)
    _std = np.std(pulse_period)
    return _std

def target_localization(sensor_array):
    # iterate through time's of arrival across all sensors
    N = len(sensor_array[0].result.toa)

    # iterate through TOA index in all sensors
    positions = []
    timestamps = []
    source_loc = np.array([0.0, 0.0])

    sigma = np.array([toa_std(sensor.result.toa) for sensor in sensor_array])

    for i in range(N):
        time_of_arrival = np.array([sensor_array[0].result.toa[i],
                                    sensor_array[1].result.toa[i],
                                    sensor_array[2].result.toa[i],
                                    sensor_array[3].result.toa[i]])
        
        # ignore z-dimension for localization.
        sensor_loc = np.array([sensor.state.loc.to_array()[:2] for sensor in sensor_array])
        
        # TODO: this should also return the emission time
        source_loc, t_emission = tdoa_model_solution_nls_gauss_newton(time_of_arrival,
                                                                      sensor_loc,
                                                                      SPEED_OF_SOUND,
                                                                      sigma)
        # ^---position measurement, to be combined with state prediction in tracking filter to determine 
        # a refined position estimate

        positions.append(Point3D(source_loc[0], source_loc[1], 0.0))
        timestamps.append(t_emission)

    return np.array(positions), np.array(timestamps)

# TDOA model
# tdoa_i = (1/c) * (sqrt((x_s-x_i)^2 + (y_s-y_i)^2) - sqrt((x_s-x_ref)^2 + (y_s-y_ref)^2)) + noise_i 
# sensor index: i = 1, 2, 3, 4,
# time of signal emission: t_emission
# position of signal source: (x_s, y_s)
# speed of signal propagation: c

# NLS w/ Gauss-Newton Solution
def tdoa_model_solution_nls_gauss_newton(t_i, loc_rx, c, sigma, num_iter=4):
    t_true = t_i[1:] - t_i[0]

    loc_src_est = np.array([2,-2])

    for _ in range(num_iter):
        t_residual = t_true - tdoa_model(loc_rx, loc_src_est, c)
        H, _ = tdoa_model_jacobian(loc_rx, loc_src_est, c)
        d_loc_src = np.linalg.pinv(H.T @ H) @ H.T @ t_residual
        loc_src_est = loc_src_est + d_loc_src

    # for each receiver, take distance  t_emissions = t_i - (1/c) * (rx to loc source)
    # take inverse var weighted average of t_emissions
    t_e = t_i - (1/c) * np.linalg.norm(loc_src_est - loc_rx, axis=1)
    t_emission = np.dot(t_e, 1/sigma**2) / np.sum(1/sigma**2)
    return loc_src_est, t_emission

def tdoa_model(loc_rx, loc_src, c):
    ref_t = (1/c) * np.linalg.norm(loc_src - loc_rx[0], axis=0)
    tdoa = (1/c) * np.linalg.norm(loc_src - loc_rx[1:], axis=1) - ref_t
    return tdoa

def tdoa_model_jacobian(r_loc, s_loc, c, sigma=None):
    ref = (1/c) * (s_loc - r_loc[0]) / np.linalg.norm(s_loc - r_loc[0])
    H = (1/c) * np.divide((s_loc - r_loc[1:]), np.linalg.norm(s_loc - r_loc[1:], axis=1)[:, None]) - ref
    
    # assumption: var(i) = sqrt(var(toa_i)**2 + var(toa_ref)**2)
    # this this assumption you can work out the matrix from Cov(e_i, e_j) = e_i * e_j, for e_i = var(i)
    # diags == var(i) ** 2, non diags == var(toa_ref) ** 2

    if sigma is not None:
        sigma_diag = (sigma[1:]**2 + sigma[0]**2)
        sigma_non_diag = sigma[0]**2
        diag = sigma_diag * np.eye(len(sigma)-1)
        non_diag = sigma_non_diag * (np.ones(len(sigma)-1) - np.eye(len(sigma)-1))
        R = diag + non_diag
        R_inv = np.linalg.pinv(R)
    else:
        R_inv = None
    return H, R_inv
