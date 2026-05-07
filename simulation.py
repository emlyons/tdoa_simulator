import numpy as np
from scipy.io import loadmat
from scipy.special import erf
import h5py
from pathlib import Path
from matplotlib import pyplot as plt
from simulator.model import ModelConfig, Model
from simulator.sensor import Channel, Sensor, Result
from analysis import get_reference_signal, find_peaks
from plotting import plot_cdf, plot_toa_distribution
from localization import target_localization
from tracking.ctvr_ekf import CTVR_EKF
from tracking.ctvr_ukf import CTVR_UKF
from tracking.singer_ekf import SINGER_EKF

import pickle
import argparse
import json
from tqdm import tqdm


CALIBRATION_CACHE_FILE = "output/model_calibration.pkl"
CALIBRATION_REPORT_FILE = "output/calibration_report.json"
SIMULATION_CACHE_FILE = "output/model_simulation.pkl"
SIMULATION_REPORT_FILE = "output/simulation_report.json"


def _load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = config_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(
            f"Configuration file is empty: {config_path}. "
            "Populate it with valid JSON or point to a different config file."
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc


def _build_sensor_array(cal_config: dict):
    sensors = cal_config.get("sensors")
    if isinstance(sensors, list) and sensors:
        microphone_array = []
        for sensor in sensors:
            x = float(sensor["x"])
            y = float(sensor["y"])
            z = float(sensor.get("z", 0.0))
            microphone_array.append(
                Sensor(x, y, z,
                    cal_config.get("sampling_rate", 1000),
                    float(sensor.get("threshold", 50)),
                    Channel(
                        gain=float(sensor.get("gain", 1)),
                        noise=float(sensor.get("noise", 1.0))
                    )))
        return microphone_array

    sensor_positions = cal_config.get("sensor_positions")
    if isinstance(sensor_positions, list) and sensor_positions:
        microphone_array = []
        for pos in sensor_positions:
            if len(pos) == 2:
                x, y = pos
                z = 0.0
            elif len(pos) >= 3:
                x, y, z = pos[:3]
            else:
                raise ValueError("Each entry in 'sensor_positions' must have at least [x, y].")
            microphone_array.append(Sensor(float(x), float(y), float(z)))
        return microphone_array

    raise ValueError(
        "Calibration config must include either a non-empty 'sensors' list "
        "or a non-empty 'sensor_positions' list."
    )

def get_calibration_data(cache: bool):
    if cache and Path(CALIBRATION_CACHE_FILE).exists():
        print("loading calibration data from cache...")
        model_cal = pickle.load(open(CALIBRATION_CACHE_FILE, "rb"))
    else:
        print("generating from calibration model...")

        config_path = Path("./config/cal_config.json")
        try:
            cal_config = _load_json_config(config_path)
        except Exception:
            raise Exception(f"Failed to load calibration config from {config_path}. Please ensure the file exists and contains valid JSON.")

        microphone_array = _build_sensor_array(cal_config)

        target_x = float(cal_config.get("target_x", 0.0))
        target_y = float(cal_config.get("target_y", 0.0))
        duration_s = float(cal_config.get("duration", 10.0))
        sampling_rate = float(cal_config.get("sampling_rate", 1000.0))

        # Static target trajectory for calibration: [time, y, x]
        t = np.arange(0.0, duration_s, 1.0 / sampling_rate)
        positions = np.zeros((len(t), 3))
        positions[:, 0] = t
        positions[:, 1] = target_y
        positions[:, 2] = target_x

        # Create model configuration and model
        config_cal = ModelConfig(positions=positions, sampling_rate=sampling_rate)
        model_cal = Model(config_cal, microphone_array)

        # Run the simulation for calibration
        n_samples = 0
        total_steps = int((model_cal.end_time - model_cal.clock) * sampling_rate)
        for _ in tqdm(range(total_steps), desc="Calibrating", unit="sample"):
            model_cal.tick()
            model_cal.sample()
            n_samples += 1

        print(f"Calibration completed with {n_samples} samples collected.")

        # Cache the calibration results
        print(f"Caching calibration results to {CALIBRATION_CACHE_FILE}...")
        pickle.dump(model_cal, open(CALIBRATION_CACHE_FILE, "wb"))

    return model_cal


def analyze_calibration_data(model: Model):
    results = {}
    toa_data = {}
    for i, sensor in enumerate(model.sensor_array):
        toa, corr, _ = find_peaks(sensor)

        if len(toa) < 2:
            continue
        
        pulse_period = np.diff(toa)
        pp_mean = np.mean(pulse_period)
        pp_std = np.std(pulse_period)
        sensor.result = Result(toa=toa, loc=sensor.state.loc)

        plot_toa_distribution(toa, f"output/sensor_{i}_toa_distribution.png")
        plot_cdf(pulse_period, i, pp_std, file_name=f"output/sensor_{i}_pulse_period_cdf.png")

    target_loc, target_timestamps = target_localization(model.sensor_array)

    # convert dict to json serializable format
    results_serializable = {f"Sensor_{i}": {"pp_mean": np.diff(sensor.result.toa).mean(),
                                            "pp_std": np.diff(sensor.result.toa).std(),
                                            "num_samples": len(sensor.result.toa),
                                            "toa": sensor.result.toa,
                                            "x": sensor.result.loc.x,
                                            "y": sensor.result.loc.y,
                                            "z": sensor.result.loc.z} for i, sensor in enumerate(model.sensor_array)}
    results_serializable["target_location"] = [{"x": loc.x, "y": loc.y, "z": loc.z, "timestamp": ts} for loc, ts in zip(target_loc, target_timestamps)]

    with open(CALIBRATION_REPORT_FILE, "w") as f:
        json.dump(results_serializable, f, indent=4)

    return results

def get_simulation_data(cache: bool):
    if cache and Path(SIMULATION_CACHE_FILE).exists():
        print("loading simulation data from cache...")
        model = pickle.load(open(SIMULATION_CACHE_FILE, "rb"))
    else:
        print("running simulation model...")

        config_path = Path("./config/simulation_config.json")
        try:
            simulation_config = _load_json_config(config_path)
        except Exception:
            raise Exception(f"Failed to load simulation config from {config_path}. Please ensure the file exists and contains valid JSON.")

        microphone_array = _build_sensor_array(simulation_config)

        sampling_rate = float(simulation_config.get("sampling_rate", 1000.0))

        positions = pickle.load(open('./data/position.pkl', 'rb'))

        # Create model configuration and model
        model_config = ModelConfig(positions=positions, sampling_rate=sampling_rate)
        model = Model(model_config, microphone_array)

        # Run the simulation for calibration
        n_samples = 0
        total_steps = int((model.end_time - model.clock) * sampling_rate)
        for _ in tqdm(range(total_steps), desc="Simulating", unit="sample"):
            model.tick()
            model.sample()
            n_samples += 1

        print(f"Simulation completed with {n_samples} samples collected.")

        # Cache the simulation results
        print(f"Caching simulation results to {SIMULATION_CACHE_FILE}...")
        pickle.dump(model, open(SIMULATION_CACHE_FILE, "wb"))
    return model

def analyze_simulation_data(model: Model):
    results = {}
    toa_data = {}
    for i, sensor in enumerate(model.sensor_array):
        toa, corr, _ = find_peaks(sensor)

        if len(toa) < 2:
            continue
        
        pulse_period = np.diff(toa)
        pp_mean = np.mean(pulse_period)
        pp_std = np.std(pulse_period)
        sensor.result = Result(toa=toa, loc=sensor.state.loc)

        plot_toa_distribution(toa, f"output/sensor_{i}_toa_distribution.png")
        plot_cdf(pulse_period, i, pp_std, file_name=f"output/sensor_{i}_pulse_period_cdf.png")

    target_loc, target_timestamps = target_localization(model.sensor_array)

    # CTVR EKF tracking of target location
    loc_ekf = []
    #                      x               y             v   Hdg  dHdg
    state = np.array([target_loc[0].x, target_loc[0].y, 0.0, 0.0, 0.0])
    ekf = CTVR_EKF(state, P=np.eye(5)*0.5, Q=np.eye(5)*0.03, R=np.eye(2) * 0.2)
    for i in range(len(target_loc)):
        measurement = np.array([target_loc[i].x, target_loc[i].y])
        dt = target_timestamps[i] - target_timestamps[i-1]
        ekf.predict(dt)
        ekf.update(measurement)
        state = ekf.get_state()
        loc_ekf.append((state[0], state[1]))

    # CTVR UKF tracking of target location
    loc_ukf = []
    #                      x               y             v   Hdg  dHdg
    state = np.array([target_loc[0].x, target_loc[0].y, 0.0, 0.0, 0.0])
    ukf = CTVR_UKF(state, P=np.eye(5)*0.5, Q=np.eye(5)*0.03, R=np.eye(2) * 0.2,
                   alpha=1e-3, beta=2, kappa=0)
    for i in range(len(target_loc)):
        measurement = np.array([target_loc[i].x, target_loc[i].y])
        dt = target_timestamps[i] - target_timestamps[i-1]
        ukf.predict(dt)
        ukf.update(measurement)
        state = ukf.get_state()
        loc_ukf.append((state[0], state[1]))

    # Singer EKF tracking of target location
    # TODO: Measurement is TDOAs
    # loc_singer_ekf = []
    # #                      x           vx   ax        y           vy   ay
    # state = np.array([target_loc[0].x, 0.0, 0.0, target_loc[0].y, 0.0, 0.0])
    # singer_ekf = SINGER_EKF(state, P=np.eye(6)*0.5, Q=np.eye(6)*0.03, R=np.eye(3) * 0.2, tau=0.5)
    # for i in range(len(target_loc)):
    #     measurement = np.array([[target_loc[i].x, target_loc[i].y]]).T
    #     dt = target_timestamps[i] - target_timestamps[i-1]
    #     singer_ekf.predict(dt)
    #     singer_ekf.update(measurement)
    #     state = singer_ekf.get_state()
    #     loc_singer_ekf.append((state[0], state[3]))


    true_locations = pickle.load(open('./data/position.pkl', 'rb'))

    # convert dict to json serializable format
    results_serializable = {f"Sensor_{i}": {"pp_mean": np.diff(sensor.result.toa).mean(),
                                            "pp_std": np.diff(sensor.result.toa).std(),
                                            "num_samples": len(sensor.result.toa),
                                            "toa": sensor.result.toa,
                                            "x": sensor.result.loc.x,
                                            "y": sensor.result.loc.y,
                                            "z": sensor.result.loc.z} for i, sensor in enumerate(model.sensor_array)}
    results_serializable["target_location"] = [{"x": loc[2], "y": loc[1], "timestamp": loc[0]} for loc in true_locations]
    results_serializable["target_location_measured"] = [{"x": loc.x, "y": loc.y, "z": loc.z, "timestamp": ts} for loc, ts in zip(target_loc, target_timestamps)]
    results_serializable["target_location_ekf"] = [{"x": loc[0], "y": loc[1], "timestamp": ts} for loc, ts in zip(loc_ekf, target_timestamps)]
    results_serializable["target_location_ukf"] = [{"x": loc[0], "y": loc[1], "timestamp": ts} for loc, ts in zip(loc_ukf, target_timestamps)]
    results_serializable["target_location_singer_ekf"] = [{"x": loc[0], "y": loc[1], "timestamp": ts} for loc, ts in zip(loc_singer_ekf, target_timestamps)]

    with open(SIMULATION_REPORT_FILE, "w") as f:
        json.dump(results_serializable, f, indent=4)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Fusion Simulation")
    parser.add_argument(
        "--mode",
        choices=["calibration", "simulation"],
        required=True,
        help="Mode to run the simulation: 'calibration' or 'simulation'"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching"
    )
    args = parser.parse_args()

    # make output directory if doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if args.mode == "calibration":
        cal_data = get_calibration_data(args.cache)
        cal_results = analyze_calibration_data(cal_data)
        ## generate calibration results report

    elif args.mode == "simulation":
        simulation_data = get_simulation_data(args.cache)
        simulation_results = analyze_simulation_data(simulation_data)
        print(f'length: {len(simulation_data.sensor_array[0].data)}')
        plt.plot(simulation_data.sensor_array[0].timestamp, simulation_data.sensor_array[0].data)
        plt.plot(simulation_data.sensor_array[1].timestamp, simulation_data.sensor_array[1].data)
        plt.plot(simulation_data.sensor_array[2].timestamp, simulation_data.sensor_array[2].data)
        plt.plot(simulation_data.sensor_array[3].timestamp, simulation_data.sensor_array[3].data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.savefig("output/simulation_signal.png")

