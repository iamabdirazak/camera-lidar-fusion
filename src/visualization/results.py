"""Results figures: detection, correction, the yaw constraint, and the system on one test frame."""
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition.kitti import Frames, read_frame
from src.calibrator import Calibrator
from src.geometry.projection import project
from src.geometry.transform import rigid
from src.paths import DRIFT_DEGREES, FIGURES, HEIGHT, REPORTS, TEST_DRIVE, WIDTH, WINDOW
from src.visualization import style

STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))


def load():
    return json.loads((REPORTS / "results.json").read_text())


def bars(axis, values, labels, colors, fmt, top):
    axis.set_xlim(-0.5, len(values) - 0.5)
    axis.set_ylim(0, top)
    style.rounded_bars(axis, range(len(values)), values, colors)
    for i, v in enumerate(values):
        axis.annotate(fmt(v), (i, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                      fontsize=7.5, color=style.NEUTRAL["ink"])
    axis.set_xticks(range(len(values)), labels)


def detection():
    style.apply()
    monitor = load()["monitor"]
    sizes = monitor["sizes_deg"]
    labels = [f"{s:g}°" for s in sizes]
    colors = [style.GREY if s < DRIFT_DEGREES else BASE for s in sizes]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.18})
    for axis, name in zip(axes, ("pitch", "yaw", "roll")):
        bars(axis, monitor["alarm_rate"][name], labels, colors, lambda v: f"{v:.0%}", 1.15)
        axis.set_title(name.capitalize())
        axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axis.set_yticks([0, 0.5, 1.0])
    axes[0].set_ylabel("Windows with the alarm raised")
    style.legend(figure, [(style.GREY, "Below the drift line"), (BASE, "Drift")], anchor=(0.9, 0.98))
    style.save(figure, FIGURES, "detection")


def correction():
    style.apply()
    corrector = load()["corrector"]
    figure, axis = plt.subplots(figsize=(5.2, 2.9))
    labels = ["Drifted"] + [f"Pass {p}" for p in corrector["passes"][1:]]
    colors = [style.GREY] + [BASE] * (len(labels) - 1)
    medians = corrector["median_deg"]
    bars(axis, medians, labels, colors, lambda v: f"{v:.2f}°", medians[0] * 1.3)
    axis.axhline(DRIFT_DEGREES, color=style.GREY, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    axis.annotate("Drift line", (axis.get_xlim()[1], DRIFT_DEGREES), xytext=(-2, 4), textcoords="offset points",
                  ha="right", va="bottom", fontsize=7.5, color=style.NEUTRAL["secondary"])
    axis.set_title("Pitch and roll left")
    axis.set_ylabel("Degrees")
    style.save(figure, FIGURES, "correction")


def yaw():
    style.apply()
    rows = load()["yaw"]
    drives = list(rows)
    labels = [d.replace(" (", "\n(") for d in drives]
    colors = [MUTED, SOFT, BASE]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), gridspec_kw={"wspace": 0.22})
    bars(axes[0], [max(rows[d]["learned_slope"], 0) for d in drives], labels, colors, lambda v: f"{v:.2f}", 1.15)
    axes[0].axhline(1.0, color=style.GREY, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    axes[0].set_title("Corrector: yaw followed (slope)")
    bars(axes[1], [rows[d]["classical_within_1deg"] for d in drives], labels, colors, lambda v: f"{v:.0%}", 1.15)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_title("Edge correlation: yaw within 1°")
    style.save(figure, FIGURES, "yaw")


def overlay():
    """One test frame at full resolution: factory calibration, drifted, and after the system."""
    style.apply()
    frames = Frames()
    window = frames.of(TEST_DRIVE)[16:16 + WINDOW]
    index = int(frames.indices[window[-1]])
    image, points = read_frame(TEST_DRIVE, index)
    full = np.diag([image.shape[1] / WIDTH, image.shape[0] / HEIGHT, 1.0]) @ frames.matrix
    drifted = rigid(np.array([2.0, 0.0, -2.5, 3.0, -2.0, 1.0])) @ frames.transform
    watch = Calibrator(drifted, frames.matrix)
    for frame in window:
        status = watch(frames.images[frame], frames.cloud(frame))
    panels = [("Factory calibration", frames.transform), ("Drifted 2° pitch, 2.5° roll", drifted),
              (f"After the system: {status}", watch.transform)]
    figure, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), gridspec_kw={"hspace": 0.22})
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for axis, (title, transform) in zip(axes, panels):
        u, v, depth, _ = project(points, transform, full, (image.shape[1], image.shape[0]))
        axis.imshow(rgb * 0.55 / 255)
        axis.scatter(u, v, s=0.35, c=np.clip(1 / depth, 0.02, 0.25), cmap=_ramp(), linewidths=0)
        axis.set_title(title)
        axis.axis("off")
    style.save(figure, FIGURES, "overlay")


def _ramp():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("", [FAINT, BASE, STRONG])


if __name__ == "__main__":
    detection()
    correction()
    yaw()
    overlay()
