"""Projecting LiDAR points into the camera, and the two-channel picture the models read.

The picture pairs the grey image with a sparse depth image: each LiDAR point drawn at
the pixel the calibration says it lands on. When the calibration is right, depth edges
fall on image edges. When it drifts, they slide apart, and that offset is what the
models learn to see.
"""
import numpy as np

from src.paths import HEIGHT, WIDTH

NEAREST_METRES = 1.0


def project(points: np.ndarray, transform: np.ndarray, matrix: np.ndarray, size=(WIDTH, HEIGHT)):
    """Pixel coordinates and depth of every point in view, with the indices of those points."""
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    depth = camera[:, 2]
    ahead = depth > NEAREST_METRES
    pixels = camera[ahead] @ matrix.T
    u, v = pixels[:, 0] / pixels[:, 2], pixels[:, 1] / pixels[:, 2]
    inside = (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
    index = np.flatnonzero(ahead)[inside]
    return u[inside], v[inside], depth[ahead][inside], index


def depth_image(points: np.ndarray, transform: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Inverse depth at each pixel a point lands on, the nearest point winning, 0 elsewhere.

    Inverse depth keeps near objects, whose edges matter most, bright and far ones dim,
    and it stays between 0 and 1 for anything beyond a metre.
    """
    u, v, depth, _ = project(points, transform, matrix)
    image = np.zeros((HEIGHT, WIDTH), np.float32)
    order = np.argsort(-depth)                                     # far first, so near points overwrite
    image[v[order].astype(int), u[order].astype(int)] = NEAREST_METRES / depth[order]
    return image


def picture(grey: np.ndarray, points: np.ndarray, transform: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """The two-channel model input: the normalised grey image and the sparse inverse depth."""
    return np.stack([(grey.astype(np.float32) / 255 - 0.45) / 0.25, depth_image(points, transform, matrix) * 4])
