"""Rigid transforms and the misalignments the system learns to find.

A misalignment is a small rotation and translation applied in the camera's frame on top
of the true LiDAR-to-camera transform. It is written as six numbers: a rotation vector
in degrees and a translation in centimetres, the units an engineer would read them in.
The system estimates the rotation; the translation is present in every sample as drift
the models must look past.
"""
import numpy as np

from src.paths import DRIFT_DEGREES, NUISANCE_CENTIMETRES, RANGE_DEGREES


def rotation(vector_degrees: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector in degrees (Rodrigues' formula)."""
    vector = np.radians(vector_degrees)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def vector(matrix: np.ndarray) -> np.ndarray:
    """Rotation vector in degrees from a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return np.degrees(angle) * axis / (2 * np.sin(angle))


def rigid(parameters: np.ndarray) -> np.ndarray:
    """4 x 4 transform from six numbers: rotation vector (degrees), translation (centimetres)."""
    transform = np.eye(4)
    transform[:3, :3] = rotation(parameters[:3])
    transform[:3, 3] = np.asarray(parameters[3:]) / 100
    return transform


def parameters(transform: np.ndarray) -> np.ndarray:
    return np.r_[vector(transform[:3, :3]), transform[:3, 3] * 100]


def error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """How far an estimated transform is from the true one: degrees and centimetres."""
    difference = estimate @ np.linalg.inv(truth)
    return float(np.linalg.norm(vector(difference[:3, :3]))), float(np.linalg.norm(difference[:3, 3]) * 100)


def direction(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def misalignment(rng, degrees: float = RANGE_DEGREES, centimetres: float = NUISANCE_CENTIMETRES) -> np.ndarray:
    """A random misalignment: random directions, sizes drawn evenly up to the given limits."""
    return np.r_[direction(rng) * rng.uniform(0, degrees), direction(rng) * rng.uniform(0, centimetres)]


def training_misalignment(rng) -> np.ndarray:
    """Half wide, for the first pass, and half narrow, so later passes learn to be precise."""
    if rng.random() < 0.1:
        return misalignment(rng, 0)                                 # aligned, apart from the translation
    if rng.random() < 0.5:
        return misalignment(rng)
    return misalignment(rng, RANGE_DEGREES / 5)


def mirror(offset: np.ndarray) -> np.ndarray:
    """The same misalignment seen in a left-right mirrored scene.

    Mirroring flips the camera's x axis. A rotation vector is an axial vector, so its x
    component keeps its sign and the other two flip; a translation flips only in x.
    """
    return offset * np.array([1, -1, -1, -1, 1, 1])
