import math
import numpy as np
import matplotlib.pyplot as plt

from .types import State, Point3D

class Channel:
    def __init__(self, gain, noise):
        self.gain = gain
        self.noise = noise

    def apply(self, sample, distance):
        noise = self.noise * np.random.normal(0.0, 1.0) 
        attenuation = math.pow(distance, -2)
        return self.gain * (attenuation * sample + noise)

class Sensor:
    def __init__(self, x, y, z, sampling_rate, threshold, channel):
        self.state = State(Point3D(x, y, z), Point3D(0, 0, 0), Point3D(0, 0, 0), 0)
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.channel = channel
        self.data = []
        self.timestamp = []

    def add_sample(self, time, sample):
        self.data.append(sample)
        self.timestamp.append(time)

    def display(self):
        plt.plot(self.timestamp, self.data)
        plt.xlabel(f'timestamp (s)')
        plt.ylabel(f'amplitude')