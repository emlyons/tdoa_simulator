from scipy.signal import fftconvolve, find_peaks, hilbert, windows
from simulator.utils import make_chirp
import numpy as np

def find_peaks(sensor):

    reference_signal, _ = get_reference_signal(sensor.sampling_rate)
    metric = matched_filter(sensor.data, reference_signal)

    # i want to iterate through entries until finding a local maximum
    # a local maximum is defined as the middle point of a group of seven points that lie on a concave down parabola with the middle point as the vertex
    peak_candidates = []
    for i in range(3, len(metric) - 3):
        if metric[i] > sensor.threshold:
            if metric[i] > metric[i-1] and metric[i] > metric[i+1] and metric[i] > metric[i-2] and metric[i] > metric[i+2] and metric[i] > metric[i-3] and metric[i] > metric[i+3]:
                peak_candidates.append(i)

    # peak suppresion
    # iterate through peak candidates starting from second, if the previous candidate is within 10ms of the current candidate, remove the one with the smaller metric value
    filtered_peaks = []
    for i in range(len(peak_candidates)):
        if i == 0:
            filtered_peaks.append(peak_candidates[i])
        else:
            if sensor.timestamp[peak_candidates[i]] - sensor.timestamp[filtered_peaks[-1]] < 0.25:
                if metric[peak_candidates[i]] > metric[filtered_peaks[-1]]:
                    filtered_peaks[-1] = peak_candidates[i]
            else:
                filtered_peaks.append(peak_candidates[i])

    final_peaks = fit_peaks(metric, sensor.timestamp, filtered_peaks)

    return final_peaks, metric, sensor.timestamp


def get_reference_signal(sampling_rate):
    signal_duration = 0.1 # 100ms
    time = np.linspace(0, signal_duration, int(sampling_rate * signal_duration)) # 100ms duration
    reference_signal = np.zeros_like(time)
    for i, t in enumerate(time):
        reference_signal[i] = make_chirp(0.0, signal_duration, t)
    return reference_signal, time


def matched_filter(signal, reference_signal, use_analytic=True, ncc=True):
    x = np.asarray(signal, dtype=float)
    r = np.asarray(reference_signal, dtype=float)

    # 2) taper reference to suppress sidelobes
    w = windows.hamming(len(r))
    r = r * w

    # 3) matched filter
    if use_analytic:
        x_f = hilbert(x)
        r_f = hilbert(r)
        k = np.conjugate(r_f[::-1])
    else:
        x_f = x
        k = r[::-1]

    # normalize kernel energy
    if ncc:
        k = k / (np.linalg.norm(k) + 1e-10)
    
    corr = fftconvolve(x_f, k, mode="same")
    metric = np.abs(corr)
    M = len(r)

    if len(metric) == 0:
        return metric, np.array([])


    return metric


def fit_peaks(metric, time, peaks):
    # fit a gaussian to the metric values around each peak to get a more accurate estimate of the peak location
    # the fit window will be 7 points centered at the peaks index
    # the peak of the gaussian fitting will be the best estimate of the peak location in time
    # x input will be the time values of the 7 points, y input will be the metric values of the 7 points
    final_peaks = []
    for peak in peaks:
        x = time[peak-3:peak+4]
        y = metric[peak-3:peak+4]

        # fit a gaussian to x and y
        A = np.max(y)
        mu = x[np.argmax(y)]
        sigma = 0.01

        # define the gaussian function
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        # fit the gaussian to the data using least squares
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(gaussian, x, y, p0=[A, mu, sigma])

        final_peaks.append(popt[1])
    return final_peaks


import pickle
from matplotlib import pyplot as plt

if __name__ == "__main__":
    sensor_array = pickle.load(open('./data/sensor_array.pkl', 'rb'))

    reference_signal, _ = get_reference_signal(10000)

    peaks, metric, aligned_time = find_peaks(sensor_array[0], threshold=50)
    peaks = peaks[:4]
    # print(f'{peaks}')

    N_toa = 3
    N_s = 1500

    plt.figure(figsize=(12, 6))

    ts = np.multiply(sensor_array[0].data, 1000)

    plt.plot(sensor_array[0].timestamp[:N_s+100], ts[:N_s+100])
    # plt.plot(time[:N_s], corr[:N_s])
    # plt.plot(toa_mic_1[:N_toa], [0] * N_toa, 'or')
    plt.plot(sensor_array[0].timestamp[:N_s+100], metric[:N_s+100])
    plt.plot(peaks, [300] * len(peaks), 'or')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('./IMG.png')
