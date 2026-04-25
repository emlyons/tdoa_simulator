import math
import numpy as np
from .types import State, Point3D

CHIRP_FREQ_MIN = 10
CHIRP_FREQ_MAX = 100
SPEED_OF_SOUND = 343.0  # m/s

def time_correction(sensor, source_state, time, time_receive, c=SPEED_OF_SOUND):

    dvec, dist = sensor.state.distance_to(source_state)

    f = time + dist / c - time_receive
    df = 1 + source_state.vel.dot(dvec) / (c * dist)
    time_correction = -f/df
    return time + time_correction

def get_doppler_factor(state_source, state_receiver, speed_in_medium=SPEED_OF_SOUND):
    s_vel = state_source.vel.to_array()
    s_loc = state_source.loc.to_array()
    r_vel = state_receiver.vel.to_array()
    r_loc = state_receiver.loc.to_array()

    # get path from source to receiver
    path = r_loc - s_loc
    path = np.divide(path, np.linalg.norm(path))

    # get magnitude of projection of source velocity onto path
    s_v_mag = np.dot(s_vel, path)

    # get magnitude of projection of receiver velocity onto reversed path
    r_v_mag = np.dot(r_vel, -path)

    doppler = (speed_in_medium + r_v_mag) / (speed_in_medium - s_v_mag)
    return doppler

def get_envelope(T, t, doppler, sigma=0.4):
    W0 = np.exp(-((0.5*T)/(sigma * 0.5 * T))**2)
    Wn = np.exp(-((t*doppler-0.5*T)/(sigma * 0.5 * T))**2) - W0 # zero amplitude at tails
    return Wn

def make_chirp(start_time, duration, sample_time, doppler=1.0):
    f_min = CHIRP_FREQ_MIN*doppler
    f_max = CHIRP_FREQ_MAX*doppler
    t = (sample_time - start_time) # apply doppler shift
    Amplitude = 20.0 * get_envelope(duration, t, doppler)
    Thalf = duration/2
    CR = (f_max - f_min)/Thalf

    if t < 0 or t > duration: # outside chirp
        return 0.0 # add noise
    
    elif t <= Thalf: # rising
        phase = 2*np.pi*(f_min*t + 0.5*CR*t**2)
        
    else: # falling
        tau = t - duration/2
        phase_offset = f_min*Thalf + 0.5*CR*Thalf**2
        phase = 2*np.pi*(phase_offset + f_max*tau - 0.5*CR*tau**2)
    signal = Amplitude * np.sin(phase)

    return signal

def signal(time, epoch, interval, duration, doppler_factor):
    # find preceding chirp start
    pulse_number = int(get_pulse_number(time, epoch, interval))
    last_emit = epoch + pulse_number * interval
    sample = make_chirp(last_emit, duration, time, doppler_factor)
    return sample

def get_pulse_number(time, epoch, interval):
    return np.floor(((time - epoch) / interval))
