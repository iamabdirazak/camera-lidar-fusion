"""The calibration watch as it would run on the vehicle: monitor, corrector, gate and status.

Every frame, the monitor scores the current calibration. Its votes are averaged over the
last five frames, because drift is persistent and a single odd frame is not. When the
average says drifting, the corrector wakes, estimates the rotation on each of those
frames, and averages the estimates. Only pitch and roll are applied: yaw is not reliably
observable on every kind of street (see the README), so it is never changed
automatically. The fix is kept only if the monitor finds the new calibration back
within the drift line. Otherwise the old calibration stays and the status asks for a
workshop recalibration, instead of applying a fix that has not solved the problem.

    from src.calibrator import Calibrator
    watch = Calibrator(transform, matrix)
    status = watch(grey, points)      # once per frame
    watch.transform                   # the calibration in use
"""
from collections import deque

import numpy as np
import onnxruntime

from src.geometry.projection import picture
from src.geometry.transform import rigid
from src.paths import MODELS, PASSES, THREADS, WINDOW

APPLIED = np.array([1.0, 0.0, 1.0])   # pitch and roll are corrected, yaw is left untouched


def session(name: str) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    return onnxruntime.InferenceSession(str(MODELS / f"{name}.onnx"), options, providers=["CPUExecutionProvider"])


class Calibrator:
    def __init__(self, transform: np.ndarray, matrix: np.ndarray):
        self.transform, self.matrix = transform, matrix
        self.monitor, self.corrector = session("monitor"), session("corrector")
        self.frames: deque = deque(maxlen=WINDOW)
        self.status = "Aligned"

    def pictures(self, transform: np.ndarray) -> np.ndarray:
        return np.stack([picture(grey, points, transform, self.matrix) for grey, points in self.frames])

    def drift(self, transform: np.ndarray) -> float:
        """The monitor's average drift probability over the window, for a given calibration."""
        return float(self.monitor.run(None, {"picture": self.pictures(transform)})[0].mean())

    def correct(self) -> np.ndarray:
        """Pitch and roll fixed in a few passes, each starting from the previous one's result."""
        transform = self.transform
        for _ in range(PASSES):
            rotation = self.corrector.run(None, {"picture": self.pictures(transform)})[0].mean(0) * APPLIED
            transform = np.linalg.inv(rigid(np.r_[rotation, 0, 0, 0])) @ transform
        return transform

    def __call__(self, grey: np.ndarray, points: np.ndarray) -> str:
        self.frames.append((grey, points))
        if len(self.frames) < WINDOW:
            return self.status
        before = self.drift(self.transform)
        if before <= 0.5:
            self.status = "Aligned"
            return self.status
        candidate = self.correct()
        if self.drift(candidate) <= 0.5:                     # the gate: apply a fix only if it ends the drift
            self.transform, self.status = candidate, "Corrected"
        else:
            self.status = "Recalibrate"                      # keep the old calibration and ask for a workshop
        self.frames.clear()                                  # judge the new calibration on fresh frames
        return self.status
