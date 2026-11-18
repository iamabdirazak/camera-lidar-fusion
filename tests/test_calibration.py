"""Checks of the geometry the whole system rests on, and of the models' shapes."""
import numpy as np
import torch

from src.geometry.transform import error, mirror, misalignment, parameters, rigid
from src.model.corrector import Corrector
from src.model.monitor import Monitor
from src.paths import HEIGHT, WIDTH


def test_parameters_round_trip():
    offset = misalignment(np.random.default_rng(0))
    assert np.allclose(parameters(rigid(offset)), offset)


def test_error_of_a_known_rotation():
    degrees, centimetres = error(rigid(np.array([0, 3.0, 0, 0, 0, 4.0])), np.eye(4))
    assert np.isclose(degrees, 3.0) and np.isclose(centimetres, 4.0)


def test_mirrored_label_matches_mirrored_picture():
    """Projecting a mirrored scene with the mirrored label gives exactly the flipped pixels."""
    rng = np.random.default_rng(1)
    points = rng.uniform([-5, -2, 5], [5, 2, 30], (50, 3))
    matrix = np.array([[240.0, 0, 208], [0, 240, 64], [0, 0, 1]])
    offset = misalignment(rng)

    def columns(scene, drift):
        transform = rigid(drift)
        camera = scene @ transform[:3, :3].T + transform[:3, 3]
        return matrix[0, 0] * camera[:, 0] / camera[:, 2] + matrix[0, 2]

    flipped = 2 * matrix[0, 2] - columns(points, offset)
    assert np.allclose(flipped, columns(points * [-1, 1, 1], mirror(offset)))


def test_model_shapes():
    picture = torch.zeros(3, 2, HEIGHT, WIDTH)
    assert Monitor().eval()(picture).shape == (3,)
    assert Corrector().eval()(picture).shape == (3, 3)
