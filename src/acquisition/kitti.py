"""KITTI raw: fetching only what the project needs, and reading it into one compact file.

Each drive is a zip archive of several gigabytes on KITTI's server. Range requests read
the archive's index and then only the files asked for: the left colour camera and the
LiDAR, every second or twentieth frame. The other cameras and the GPS/IMU data never leave the server.

Run as a module to download, then to prepare:
    python -m src.acquisition.kitti download
    python -m src.acquisition.kitti prepare
"""
import os
import sys
import time

import cv2
import numpy as np

from src.paths import DATE, FRAMES, HEIGHT, KITTI, STEPS, WIDTH

SERVER = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
FIELD_OF_VIEW = np.radians(55)  # half-angle kept around the camera axis: its 40 degrees plus room for drift


def download() -> None:
    from remotezip import RemoteZip

    with RemoteZip(f"{SERVER}/{DATE}_calib.zip", timeout=30) as archive:
        archive.extractall(KITTI)
    for drive, step in STEPS.items():
        with RemoteZip(f"{SERVER}/{DATE}_drive_{drive}/{DATE}_drive_{drive}_sync.zip", timeout=30) as archive:
            wanted = [info for info in archive.infolist() if not info.is_dir()
                      and ("image_02/data/" in info.filename or "velodyne_points/data/" in info.filename)
                      and int(info.filename[-14:-4]) % step == 0]
            for count, info in enumerate(wanted, 1):
                path = KITTI / info.filename
                if path.exists() and path.stat().st_size == info.file_size:
                    continue                                    # already complete, so a rerun resumes
                for attempt in range(5):
                    try:
                        archive.extract(info, KITTI)
                        break
                    except Exception:
                        time.sleep(3)
                print(f"\r{drive}  {count}/{len(wanted)}", end="", flush=True)
            print(f"\r{drive}  {len(wanted) // 2} frames")


def read_calibration(path) -> dict[str, np.ndarray]:
    values = {}
    for line in open(path):
        key, _, text = line.partition(":")
        try:
            values[key] = np.array(text.split(), float)
        except ValueError:
            pass                                                # the date line
    return values


def calibration() -> tuple[np.ndarray, np.ndarray]:
    """The factory LiDAR-to-camera transform and the camera matrix at the model's resolution.

    KITTI chains three steps: LiDAR to the reference camera, rectification, and the
    offset of the left colour camera from the reference one, which sits in P_rect_02's
    last column. Folding them gives one rigid transform into the colour camera.
    """
    lidar = read_calibration(KITTI / DATE / "calib_velo_to_cam.txt")
    camera = read_calibration(KITTI / DATE / "calib_cam_to_cam.txt")
    to_reference = np.eye(4)
    to_reference[:3, :3] = lidar["R"].reshape(3, 3)
    to_reference[:3, 3] = lidar["T"]
    rectify = np.eye(4)
    rectify[:3, :3] = camera["R_rect_00"].reshape(3, 3)
    projection = camera["P_rect_02"].reshape(3, 4)
    matrix = projection[:, :3]
    offset = np.eye(4)
    offset[:3, 3] = np.linalg.solve(matrix, projection[:, 3])
    transform = offset @ rectify @ to_reference

    width, height = camera["S_rect_02"]
    scale = np.diag([WIDTH / width, HEIGHT / height, 1.0])
    return transform, scale @ matrix


def read_frame(drive: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    folder = KITTI / DATE / f"{DATE}_drive_{drive}_sync"
    image = cv2.imread(str(folder / "image_02" / "data" / f"{index:010d}.png"))
    points = np.fromfile(folder / "velodyne_points" / "data" / f"{index:010d}.bin", np.float32).reshape(-1, 4)
    return image, points[:, :3]


def frames(drive: str) -> list[int]:
    """Frames that have both an image and a scan. KITTI's drive 0009 is missing a few scans."""
    folder = KITTI / DATE / f"{DATE}_drive_{drive}_sync"
    images, scans = ({int(name[:10]) for name in os.listdir(folder / sensor / "data")}
                     for sensor in ("image_02", "velodyne_points"))
    return sorted(images & scans)


def prepare() -> None:
    """One compact file: each frame's grey image at the model's size and the LiDAR points in front.

    Points stay in the LiDAR's own frame and in scan order, because the calibration is
    what is being estimated and the edge baseline needs neighbouring points of a ring.
    """
    transform, matrix = calibration()
    images, clouds, drives, indices = [], [], [], []
    for drive in STEPS:
        for index in frames(drive):
            image, points = read_frame(drive, index)
            ahead = (points[:, 0] > 1.0) & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < FIELD_OF_VIEW)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(cv2.resize(grey, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            clouds.append(points[ahead].astype(np.float32))
            drives.append(drive)
            indices.append(index)
    counts = np.array([len(cloud) for cloud in clouds])
    np.savez_compressed(FRAMES, images=np.stack(images), points=np.concatenate(clouds),
                        offsets=np.r_[0, np.cumsum(counts)], drives=np.array(drives), indices=np.array(indices),
                        transform=transform, matrix=matrix)
    print(f"{len(images)} frames, {counts.mean():.0f} points each, {FRAMES.stat().st_size / 1e6:.0f} MB")


class Frames:
    """The prepared frames, held in memory."""

    def __init__(self, path=FRAMES):
        data = np.load(path)
        self.images = data["images"]
        self.points = data["points"]
        self.offsets = data["offsets"]
        self.drives = data["drives"]
        self.indices = data["indices"]
        self.transform = data["transform"]
        self.matrix = data["matrix"]

    def __len__(self) -> int:
        return len(self.images)

    def cloud(self, frame: int) -> np.ndarray:
        return self.points[self.offsets[frame]:self.offsets[frame + 1]]

    def of(self, drive: str) -> np.ndarray:
        return np.flatnonzero(self.drives == drive)


if __name__ == "__main__":
    {"download": download, "prepare": prepare}[sys.argv[1]]()
