"""Time per frame on a CPU with ONNX Runtime: what the watch costs when all is well, and when it wakes.

    python -m src.evaluation.latency
"""
import json
import time

import numpy as np

from src.acquisition.kitti import Frames
from src.calibrator import session
from src.evaluation.metrics import window_pictures
from src.geometry.projection import picture
from src.paths import REPORTS, TEST_DRIVE, THREADS, WINDOW

RUNS = 100


def timed(work) -> dict:
    work()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        work()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(float(np.median(times)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1)}


def main() -> None:
    frames = Frames()
    test = frames.of(TEST_DRIVE)
    monitor, corrector = session("monitor"), session("corrector")
    frame, window = test[0], test[:WINDOW]
    one = picture(frames.images[frame], frames.cloud(frame), frames.transform, frames.matrix)[None]
    several = window_pictures(frames, window, frames.transform)
    results = {
        "threads": THREADS,
        "projection_per_frame": timed(lambda: picture(frames.images[frame], frames.cloud(frame), frames.transform,
                                                      frames.matrix)),
        "monitor_per_frame": timed(lambda: monitor.run(None, {"picture": one})),
        "corrector_per_pass_5_frames": timed(lambda: corrector.run(None, {"picture": several})),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "latency.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
