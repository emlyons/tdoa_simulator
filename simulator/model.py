import numpy as np

from .utils import get_pulse_number, signal, time_correction, get_doppler_factor
from .target import Target

class ModelConfig:
    def __init__(self, positions, sampling_rate):
        self.positions = positions
        self.sampling_rate = sampling_rate
        self.epoch = 0
        self.emission_duration = 0.1 # 100ms
        self.emission_rate = 0.5 # 2hz
        self.speed_of_sound = 343 # Speed of sound in air in m/s
        self.speed_of_sound_error = 0.5 # chirp-level sigma in m/s
        self.clock_error = 1e-4 # 100us

class Model:
    def __init__(self, config, sensor_array):
        self.config = config
        self.target = Target(config.positions)
        self.sensor_array = sensor_array
        self.clock = self.target.get_time_bounds()[0]
        self.end_time = self.target.get_time_bounds()[1]

    def tick(self):
        self.clock += self.config.sampling_rate**-1
        if self.clock >= self.end_time:
            return False
        return True

    def sample(self):
        current_state = self.target.get_state(self.clock)
        pulse_number = int(get_pulse_number(self.clock, self.config.epoch, self.config.emission_rate))

        # Treat environmental sound speed variation as a chirp-level effect
        c_eff = self.config.speed_of_sound # self._get_speed_of_sound_with_error(pulse_number + 173)

        for s_id, sensor in enumerate(self.sensor_array):# at sample

            source = current_state
            t_clock = self.clock
            time = t_clock

            # 2 iterations gets us to 10e-8 error
            num_iters = 2
            for _ in range(num_iters):
                time = time_correction(sensor, source, time, t_clock, c_eff)
                source = self.target.get_state(time)

            _, dist = sensor.state.distance_to(source)

            doppler_factor = get_doppler_factor(sensor.state, source, c_eff)

            # calculate pulse number based on emmission rate and epoch
            # use pulse number to seed a random number generator for a timeshift
            # this time shift will change per pulse
            t_err = self._get_time_with_clock_error(time, pulse_number + 7 * s_id)

            sample = signal(t_err, self.config.epoch, self.config.emission_rate,
                            self.config.emission_duration, doppler_factor)
            sample = sensor.channel.apply(sample, dist)
            sensor.add_sample(self.clock, sample)

    def _get_speed_of_sound_with_error(self, seed):
        rng = np.random.default_rng(seed)
        return self.config.speed_of_sound + rng.normal(0.0, self.config.speed_of_sound_error)

    def _get_time_with_clock_error(self, time, seed):
        rng = np.random.default_rng(seed)
        time_error = rng.normal(0.0, self.config.clock_error)
        return time + time_error