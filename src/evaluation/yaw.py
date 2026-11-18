"""Why yaw is left uncorrected: measuring what limits it, on streets the corrector knows and on new ones.

Two estimators are compared on drift in yaw alone. The learned corrector shows whether
its yaw skill carries over from the streets it trained on to new ones. A classical
estimator with nothing to learn, the global correlation of upright image edges with
upright depth edges, shows whether the information is there to be found at all.
"""
import cv2
import numpy as np

from src.geometry.projection import depth_image
from src.geometry.transform import misalignment, rigid

REACH = 48            # pixels either way at the model's resolution, about 11 degrees
MARGIN = 48           # columns left out at each side, where a shifted depth image wraps


def upright_edges(image: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0))


def classical_yaw(frames, window: np.ndarray, transform: np.ndarray) -> float:
    """Yaw from the sideways shift at which upright depth edges best overlay upright image edges.

    Scan lines are joined into short columns first, so a pole reads as an upright edge.
    The match is summed over the window's frames, normalised per frame, and its best
    shift is turned into degrees with the camera's focal length.
    """
    profile = np.zeros(2 * REACH + 1)
    for frame in window:
        image = upright_edges(frames.images[frame])
        depth = cv2.dilate(depth_image(frames.cloud(frame), transform, frames.matrix), np.ones((5, 1)))
        depth = upright_edges(depth)
        match = np.array([(image[:, MARGIN:-MARGIN] * np.roll(depth, -shift, 1)[:, MARGIN:-MARGIN]).sum()
                          for shift in range(-REACH, REACH + 1)])
        profile += match / (match.mean() + 1e-9)
    pixels_per_degree = frames.matrix[0, 0] * np.pi / 180
    return (np.argmax(profile) - REACH) / pixels_per_degree


def study(frames, corrector, windows: dict[str, list[np.ndarray]], rng, trials: int = 30, limit: float = 4.0):
    """Per drive: how the learned and the classical estimates follow a known yaw drift."""
    from src.evaluation.metrics import window_pictures

    rows = {}
    for drive, candidates in windows.items():
        truth, learned, classical = [], [], []
        for _ in range(trials):
            window = candidates[rng.integers(len(candidates))]
            offset = misalignment(rng, 0)
            offset[1] = rng.uniform(-limit, limit)
            transform = rigid(offset) @ frames.transform
            truth.append(offset[1])
            learned.append(float(corrector.run(None, {"picture": window_pictures(frames, window, transform)})[0][:, 1].mean()))
            classical.append(classical_yaw(frames, window, transform))
        truth, learned, classical = map(np.array, (truth, learned, classical))
        rows[drive] = {
            "learned_slope": round(float(np.polyfit(truth, learned, 1)[0]), 2),
            "learned_error_deg": round(float(np.median(np.abs(learned - truth))), 2),
            "classical_error_deg": round(float(np.median(np.abs(classical - truth))), 2),
            "classical_within_1deg": round(float(np.mean(np.abs(classical - truth) < 1)), 2),
        }
    return rows
