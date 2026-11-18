"""Small shared pieces of the evaluation: windows of frames, their pictures, and error measures."""
import numpy as np

from src.geometry.projection import picture
from src.geometry.transform import vector
from src.paths import WINDOW


def windows(indices: np.ndarray, step: int = 1) -> list[np.ndarray]:
    """Every run of consecutive frames the size of the monitor's window."""
    return [indices[start:start + WINDOW] for start in range(0, len(indices) - WINDOW + 1, step)]


def window_pictures(frames, window: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.stack([picture(frames.images[i], frames.cloud(i), transform, frames.matrix) for i in window])


def rotation_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The rotation left between two calibrations, as a rotation vector in degrees (pitch, yaw, roll)."""
    return vector(estimate[:3, :3] @ truth[:3, :3].T)


def tilt(estimate: np.ndarray, truth: np.ndarray) -> float:
    """The pitch and roll left over, in degrees: the part of the error the system corrects."""
    left = rotation_error(estimate, truth)
    return float(np.hypot(left[0], left[2]))
