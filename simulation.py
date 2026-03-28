import numpy as np
from scipy.io import loadmat
import h5py
from pathlib import Path
from matplotlib import pyplot as plt
from simulator.model import ModelConfig, Model
from simulator.sensor import Channel, Sensor
from analysis import get_reference_signal, find_peaks

import pickle
import argparse
import json
from tqdm import tqdm


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

def analyze_calibration_data(cal_model):
    results = {}
    for i, sensor in enumerate(cal_model.sensor_array):
        toa, corr, _ = find_peaks(sensor)

        plt.figure()
        plt.plot(sensor.timestamp, sensor.data, label=f"Sensor {i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Sensor Data")
        plt.title("Sensor Data Over Time")
        plt.legend()
        plt.savefig(f"sensor_{i}_data.png")

        plt.figure()
        plt.plot(sensor.timestamp, corr, label=f"Sensor {i} Correlation")
        plt.xlabel("Time (s)")
        plt.ylabel("Correlation")
        plt.title("Sensor Correlation Over Time")
        plt.legend()

        if len(toa) < 2:
            print(f"Warning: Sensor {i} has less than 2 detected peaks. Skipping pulse period analysis.")
            results[i] = {"num_samples": len(toa), "pulse_period_mean": None, "pulse_period_std": None}
            plt.savefig(f"sensor_{i}_correlation.png")
            continue
        
        pulse_period = np.diff(toa)
        pp_mean = np.mean(pulse_period)
        pp_std = np.std(pulse_period)
        results[i] = {"num_samples": len(toa), "pulse_period_mean": pp_mean, "pulse_period_std": pp_std}

        for j, t in enumerate(toa):
            lbl = "Detected Peaks" if j == 0 else None
            plt.axvline(x=t, color='r', linestyle=':', linewidth=1, label=lbl)
        plt.savefig(f"sensor_{i}_correlation.png")

    # convert dict to json serializable format
    results_serializable = {f"Sensor_{i}": {"pp_mean": res["pulse_period_mean"], "pp_std": res["pulse_period_std"], "num_samples": res["num_samples"]} for i, res in results.items()}
    with open("calibration_report.json", "w") as f:
        json.dump(results_serializable, f, indent=4)

    return results



def get_calibration_data(cache: bool):
    if cache and Path("model_calibration.pkl").exists():
        print("calibration cached results...")
        model_cal = pickle.load(open("model_calibration.pkl", "rb"))
    else:
        print("generating from calibration model...")

        config_path = Path("./config/cal_config.json")
        try:
            cal_config = _load_json_config(config_path)
        except Exception:
            fallback_path = Path("./config/test_config.json")
            if fallback_path.exists():
                print("Primary calibration config invalid/empty. Falling back to ./config/test_config.json")
                cal_config = _load_json_config(fallback_path)
            else:
                raise

        microphone_array = _build_sensor_array(cal_config)

        target_x = float(cal_config.get("target_x", 0.0))
        target_y = float(cal_config.get("target_y", 0.0))
        duration_s = float(cal_config.get("duration", 10.0))
        sampling_rate = float(cal_config.get("sampling_rate", 10000.0))

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
        print(f"Caching calibration results to model_calibration.pkl...")
        pickle.dump(model_cal, open("model_calibration.pkl", "wb"))

    return model_cal

def run_test(cache: bool):
    if cache and Path("model_calibration.pkl").exists():
        print("Loading calibration from cache...")
        model_test = pickle.load(open("model_test.pkl", "rb"))
    else:
        print("Calibration cache not found. Please run calibration first.")
        return
    print("Running test mode...")
    # Here you can add code to run tests using the loaded calibration model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Fusion Simulation")
    parser.add_argument(
        "--mode",
        choices=["calibration", "test"],
        required=True,
        help="Mode to run the simulation: 'calibration' or 'test'"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching"
    )
    args = parser.parse_args()

    if args.mode == "calibration":
        cal_data = get_calibration_data(args.cache)
        cal_results = analyze_calibration_data(cal_data)
        ## generate calibration results report

    elif args.mode == "test":
        print("Running test mode...")
        # Load the calibration model
        model_cal = pickle.load(open("model_calibration.pkl", "rb"))

        # Here you can add code to run tests using the loaded calibration model
