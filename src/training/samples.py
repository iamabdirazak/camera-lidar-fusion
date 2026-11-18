"""Self-supervised training samples: misalign the factory calibration on purpose, and keep the answer.

KITTI's calibration was measured carefully with targets, so it serves as the aligned
truth. Every sample takes a frame, applies a random misalignment to that truth, and
draws the two-channel picture the system would see with the drifted calibration. The
misalignment applied is the label, so there is nothing to annotate, and every frame
can be drifted in endless different ways.

Scenes are what the data is short of, so each sample also varies what the camera and
LiDAR see without changing the geometry: the scene may be mirrored left to right (with
the label mirrored to match), the image's exposure changes, and a share of the LiDAR
points is dropped, as rain, dark surfaces and range do on a real vehicle.
"""
import numpy as np
import torch

from src.acquisition.kitti import Frames
from src.geometry.projection import picture
from src.geometry.transform import mirror, misalignment, rigid, training_misalignment
from src.paths import DRIFT_DEGREES, TEST_DRIVE, TRAINING_DRIVES, VALIDATION_DRIVE


def split(frames: Frames) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Training frames grouped by drive, so every street is drawn equally often, and the validation drive."""
    return [frames.of(drive) for drive in TRAINING_DRIVES], [frames.of(VALIDATION_DRIVE)]


def monitor_misalignment(rng) -> np.ndarray:
    """Weighted towards the boundary: below it should stay quiet, just above it should raise the alarm."""
    if rng.random() < 0.35:
        return misalignment(rng, DRIFT_DEGREES * 0.8)
    if rng.random() < 0.5:
        return misalignment(rng, DRIFT_DEGREES * 3)
    return misalignment(rng)


class Samples(torch.utils.data.Dataset):
    """Pictures with drifted calibrations, and for each the misalignment and whether it counts as drift."""

    def __init__(self, frames: Frames, drives: list[np.ndarray], task: str, size: int, seed: int, fixed: bool = False):
        self.frames, self.drives, self.task, self.size, self.seed, self.fixed = frames, drives, task, size, seed, fixed
        self.draw = monitor_misalignment if task == "monitor" else training_misalignment

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int):
        # A fixed set draws the same samples every time it is read; a training set never repeats one.
        rng = np.random.default_rng((self.seed, item) if self.fixed else None)
        drive = self.drives[rng.integers(len(self.drives))]
        frame = drive[rng.integers(len(drive))]
        offset = self.draw(rng)
        grey, cloud = self.frames.images[frame], self.frames.cloud(frame)
        if not self.fixed:
            grey = np.clip(grey * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
            cloud = cloud[rng.random(len(cloud)) < rng.uniform(0.5, 1.0)]
        image = picture(grey, cloud, rigid(offset) @ self.frames.transform, self.frames.matrix)
        if not self.fixed and rng.random() < 0.5:
            image, offset = np.ascontiguousarray(image[:, :, ::-1]), mirror(offset)
        drifting = np.linalg.norm(offset[:3]) > DRIFT_DEGREES
        return torch.from_numpy(image), torch.tensor(offset[:3], dtype=torch.float32), torch.tensor(float(drifting))


def test_frames(frames: Frames) -> np.ndarray:
    return frames.of(TEST_DRIVE)
