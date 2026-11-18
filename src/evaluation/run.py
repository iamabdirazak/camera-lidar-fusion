"""The evaluation on the test drive, 0020, which no stage of training has seen.

    python -m src.evaluation.run

Every drift is applied on purpose to the factory calibration, with up to 5 cm of
translation drift on top, which the system must look past. All draws use a fixed seed.

1. Monitor: how often it raises the alarm for drift of each size about each axis, and
   how often it raises it when nothing is wrong.
2. Corrector: the pitch and roll left after each pass.
3. System: what the full calibration watch reports for aligned, pitch and roll, and yaw
   drift, and whether the gate ever let a fix make things worse.
4. Yaw: what limits it (src/evaluation/yaw.py).
"""
import json

import numpy as np

from src.acquisition.kitti import Frames
from src.calibrator import Calibrator, session
from src.evaluation.metrics import tilt, window_pictures, windows
from src.evaluation.yaw import study
from src.geometry.transform import misalignment, rigid
from src.paths import PASSES, REPORTS, SEED, TEST_DRIVE, TRAINING_DRIVES, VALIDATION_DRIVE

AXES = {"pitch": 0, "yaw": 1, "roll": 2}
SIZES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
TRIALS = 40


def drift(rng, axis: int | None, size: float) -> np.ndarray:
    """A drift of a given size about one axis, plus the translation drift every sample carries."""
    offset = misalignment(rng, 0)
    if axis is not None:
        offset[axis] = size * rng.choice([-1, 1])
    return offset


def tilt_drift(rng, low: float = 0.0, high: float = 5.0) -> np.ndarray:
    """Pitch and roll drift together, in a random direction, of a size between the limits."""
    offset = misalignment(rng, 0)
    angle = rng.uniform(0, 2 * np.pi)
    offset[[0, 2]] = rng.uniform(low, high) * np.array([np.cos(angle), np.sin(angle)])
    return offset


def yaw_drift(rng) -> np.ndarray:
    offset = misalignment(rng, 0)
    offset[1] = rng.uniform(2, 5) * rng.choice([-1, 1])
    return offset


def monitor_study(frames, monitor, test_windows, rng) -> dict:
    def alarm(transform, window):
        votes = monitor.run(None, {"picture": window_pictures(frames, window, transform)})[0]
        return votes.mean() > 0.5, float((votes > 0.5).mean())

    aligned = [alarm(frames.transform, window) for window in test_windows]
    rates = {}
    for name, axis in AXES.items():
        rates[name] = []
        for size in SIZES:
            hits = [alarm(rigid(drift(rng, axis, size)) @ frames.transform,
                          test_windows[rng.integers(len(test_windows))])[0] for _ in range(TRIALS)]
            rates[name].append(round(float(np.mean(hits)), 3))
    return {"sizes_deg": SIZES, "alarm_rate": rates,
            "false_alarms": {"windows": round(float(np.mean([a for a, _ in aligned])), 3),
                             "frames": round(float(np.mean([f for _, f in aligned])), 3)}}


def corrector_study(frames, corrector, test_windows, rng) -> dict:
    left = [[] for _ in range(PASSES + 1)]
    for _ in range(TRIALS * 2):
        window = test_windows[rng.integers(len(test_windows))]
        transform = rigid(tilt_drift(rng)) @ frames.transform
        left[0].append(tilt(transform, frames.transform))
        for step in range(PASSES):
            rotation = corrector.run(None, {"picture": window_pictures(frames, window, transform)})[0].mean(0)
            rotation[1] = 0                                                   # yaw is never applied
            transform = np.linalg.inv(rigid(np.r_[rotation, 0, 0, 0])) @ transform
            left[step + 1].append(tilt(transform, frames.transform))
    return {"passes": list(range(PASSES + 1)),
            "median_deg": [round(float(np.median(v)), 2) for v in left],
            "p90_deg": [round(float(np.percentile(v, 90)), 2) for v in left]}


def system_study(frames, test_windows, rng) -> dict:
    """Start the calibration watch from a drifted calibration, give it five frames, and record its status."""
    cases = {"aligned": lambda: misalignment(rng, 0),
             "pitch and roll, 2 to 5 degrees": lambda: tilt_drift(rng, 2.0),
             "yaw, 2 to 5 degrees": lambda: yaw_drift(rng)}
    outcome = {}
    for name, draw in cases.items():
        counts = {"Aligned": 0, "Corrected": 0, "Recalibrate": 0}
        before, after = [], []
        for _ in range(TRIALS):
            drifted = rigid(draw()) @ frames.transform
            watch = Calibrator(drifted, frames.matrix)
            for frame in test_windows[rng.integers(len(test_windows))]:
                status = watch(frames.images[frame], frames.cloud(frame))
            counts[status] += 1
            before.append(tilt(drifted, frames.transform))
            after.append(tilt(watch.transform, frames.transform))
        before, after = np.array(before), np.array(after)
        outcome[name] = {"status": counts,
                         "tilt_median_deg": [round(float(np.median(before)), 2), round(float(np.median(after)), 2)],
                         "left_worse_by_over_quarter_degree": int(np.sum(after > before + 0.25))}
    return outcome


def main() -> None:
    rng = np.random.default_rng(SEED)
    frames = Frames()
    monitor, corrector = session("monitor"), session("corrector")
    test_windows = windows(frames.of(TEST_DRIVE))
    results = {
        "monitor": monitor_study(frames, monitor, test_windows, rng),
        "corrector": corrector_study(frames, corrector, test_windows, rng),
        "system": system_study(frames, test_windows, rng),
        "yaw": study(frames, corrector, {
            "training (0005)": windows(frames.of(next(iter(TRAINING_DRIVES)))),
            "validation (0039)": windows(frames.of(VALIDATION_DRIVE)),
            "test (0020)": test_windows}, rng),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
