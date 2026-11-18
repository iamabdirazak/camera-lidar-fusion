"""Documentation diagrams: the two models as files, and the system architecture.

The layout: a soft frame with a title, a symmetric row of raised cards, each with an
icon, a name and a short description, and small arrows between them.
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Wedge
from matplotlib.path import Path as Shape
from PIL import Image

from src.paths import MODELS, ROOT
from src.visualization import style

DOCS = ROOT / "docs"
STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41
MARGIN, CARD_Y = 0.4, 0.4
CARD_RADIUS, FRAME_RADIUS = 0.27, 0.38
HEIGHT = CARD_Y + CARD_H + 1.0
ICON_Y = CARD_Y + 2.25
NAME_Y, SUBTITLE_Y = CARD_Y + 1.05, CARD_Y + 0.65
MIDDLE = CARD_Y + CARD_H / 2


def rounded(axis, x, y, w, h, color, radius=0.12, z=3, alpha=1.0):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def scan(axis, cx, cy, width, rows, color, shift=0.0, z=6, size=2.2):
    """LiDAR points along level scan lines, as they land on the image."""
    xs = np.linspace(cx - width / 2, cx + width / 2, 13)
    for k, y in enumerate(rows):
        axis.scatter(xs + shift, np.full_like(xs, y), s=size, color=color, linewidths=0, zorder=z)


# Icons, each centred on (cx, cy) ------------------------------------------

def icon_input(axis, cx, cy):
    """Camera frames arriving one after another, LiDAR points laid over the newest."""
    for dx, dy, alpha in ((-0.3, -0.3, 0.3), (-0.16, -0.16, 0.55)):
        rounded(axis, cx - 0.5 + dx, cy - 0.34 + dy, 1.0, 0.7, BASE, 0.1, z=3, alpha=alpha)
    rounded(axis, cx - 0.5, cy - 0.34, 1.0, 0.7, BASE, 0.1, z=4)
    rounded(axis, cx - 0.12, cy - 0.16, 0.24, 0.34, MUTED, 0.03, z=5)          # an upright object
    scan(axis, cx, cy, 0.8, (cy - 0.2, cy - 0.08, cy + 0.04), STRONG)


def arrow(axis, start, end, color, z=6):
    """A straight arrow with a small solid head."""
    (x0, y0), (x1, y1) = start, end
    axis.plot([x0, x1], [y0, y1], color=color, linewidth=2.2, solid_capstyle="round", zorder=z)
    d = np.array([x1 - x0, y1 - y0])
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tip = np.array([x1, y1]) + d * 0.07
    axis.add_patch(Polygon([tip, tip - d * 0.16 + n * 0.08, tip - d * 0.16 - n * 0.08], facecolor=color,
                           edgecolor="none", zorder=z))


def icon_matching(axis, cx, cy):
    """Two directions of matching: sideways at an upright edge, up and down at a level one."""
    rounded(axis, cx - 0.5, cy - 0.42, 1.0, 0.84, MUTED, 0.1, z=3)
    rounded(axis, cx - 0.3, cy - 0.3, 0.07, 0.6, STRONG, 0.03, z=4)           # an upright edge
    rounded(axis, cx - 0.12, cy + 0.2, 0.52, 0.07, STRONG, 0.03, z=4)         # a level edge
    arrow(axis, (cx - 0.16, cy - 0.12), (cx + 0.06, cy - 0.12), STRONG)
    arrow(axis, (cx + 0.24, cy + 0.06), (cx + 0.24, cy - 0.22), STRONG)


def icon_monitor(axis, cx, cy):
    """A steady pulse with one beat: always on, and it notices."""
    t = np.linspace(-0.55, 0.55, 200)
    y = 0.03 * np.sin(t * 30) + 0.34 * np.exp(-((t - 0.1) / 0.05) ** 2) - 0.18 * np.exp(-((t - 0.18) / 0.04) ** 2)
    axis.plot(cx + t, cy + y, color=BASE, linewidth=2.4, solid_capstyle="round", zorder=5)
    axis.add_patch(Circle((cx + 0.47, cy + 0.38), 0.07, facecolor=STRONG, edgecolor="none", zorder=6))


def icon_corrector(axis, cx, cy):
    """Scan lines that had tilted off the scene, turned back level."""
    angle = np.radians(14)
    xs = np.linspace(-0.42, 0.42, 9)
    for y in (-0.2, 0.0, 0.2):
        axis.scatter(cx + 0.08 + xs * np.cos(angle), cy + y + xs * np.sin(angle), s=7, color=FAINT, linewidths=0, zorder=4)
        axis.scatter(cx + 0.08 + xs, np.full_like(xs, cy + y), s=9, color=STRONG, linewidths=0, zorder=5)
    t = np.radians(np.linspace(120, 230, 30))
    axis.plot(cx - 0.42 + 0.3 * np.cos(t) + 0.3, cy + 0.3 * np.sin(t), color=BASE, linewidth=2.2,
              solid_capstyle="round", zorder=6)
    end = np.array([cx - 0.12 + 0.3 * np.cos(t[-1]), cy + 0.3 * np.sin(t[-1])])
    arrow(axis, end + np.array([-0.05, 0.06]), end, BASE)


def icon_gate(axis, cx, cy):
    """A shield with a check: a fix passes only if it makes the match better."""
    points = [(cx - 0.36, cy + 0.36), (cx + 0.36, cy + 0.36), (cx + 0.36, cy - 0.02),
              (cx, cy - 0.44), (cx - 0.36, cy - 0.02)]
    axis.add_patch(Polygon(points, closed=True, facecolor=MUTED, edgecolor="none", joinstyle="round", zorder=4))
    axis.plot([cx - 0.16, cx - 0.03, cx + 0.18], [cy - 0.0, cy - 0.13, cy + 0.13], color=STRONG, linewidth=3,
              solid_capstyle="round", solid_joinstyle="round", zorder=5)


def icon_output(axis, cx, cy):
    """The frame with depth landing on the object again, and the status it reports."""
    rounded(axis, cx - 0.5, cy - 0.34, 1.0, 0.7, MUTED, 0.1, z=4)
    rounded(axis, cx - 0.12, cy - 0.2, 0.24, 0.34, FAINT, 0.03, z=5)
    scan(axis, cx, cy, 0.8, (cy - 0.14, cy - 0.02, cy + 0.1), STRONG)
    rounded(axis, cx + 0.2, cy + 0.22, 0.42, 0.24, BASE, 0.12, z=7)
    axis.plot([cx + 0.31, cx + 0.38, cx + 0.51], [cy + 0.34, cy + 0.28, cy + 0.4], color=style.NEUTRAL["ink"],
              linewidth=1.8, solid_capstyle="round", solid_joinstyle="round", zorder=8)


def diagram(title: str, cards: list[tuple], name: str) -> None:
    style.apply()
    width = len(cards) * CARD_W + (len(cards) - 1) * GAP + 2 * MARGIN
    figure, axis = plt.subplots(figsize=(width / 1.45, HEIGHT / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, HEIGHT, FRAME, FRAME_RADIUS, z=1)
    axis.text(width / 2, HEIGHT - 0.5, title, ha="center", va="center", fontsize=13, fontweight="medium",
              color=style.NEUTRAL["ink"], zorder=6)
    for index, (name_text, subtitle, draw) in enumerate(cards):
        x = MARGIN + index * (CARD_W + GAP)
        centre = x + CARD_W / 2
        rounded(axis, x, CARD_Y, CARD_W, CARD_H, CARD, CARD_RADIUS, z=2)
        draw(axis, centre, ICON_Y)
        axis.text(centre, NAME_Y, name_text, ha="center", va="center", fontsize=10, fontweight="medium",
                  color=style.NEUTRAL["ink"], zorder=6)
        axis.text(centre, SUBTITLE_Y, subtitle, ha="center", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        if index:
            mid = x - GAP / 2
            axis.add_patch(Polygon([[mid - 0.07, MIDDLE - 0.12], [mid - 0.07, MIDDLE + 0.12], [mid + 0.09, MIDDLE]],
                                   facecolor=ARROW, edgecolor="none", zorder=3))
    style.save(figure, DOCS, name, panel=False)


def architecture() -> None:
    diagram("System Architecture", [
        ("Input", "Image and projected LiDAR", icon_input),
        ("Matching", "Two directions of edges", icon_matching),
        ("Monitor", "Every frame, 5-frame vote", icon_monitor),
        ("Corrector", "Pitch and roll, 3 passes", icon_corrector),
        ("Gate", "Only if the match improves", icon_gate),
        ("Output", "Calibration and status", icon_output),
    ], "architecture")


# The models as files --------------------------------------------------------

def fill(axis, patch, colors, angle=90, z=3):
    axis.add_patch(patch)
    patch.set_facecolor("none")
    patch.set_edgecolor("none")
    x0, y0, w, h = patch.get_path().transformed(patch.get_patch_transform()).get_extents().bounds
    t = np.radians(angle)
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ramp = gx * np.cos(t) + gy * np.sin(t)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    image = axis.imshow(ramp, extent=(x0, x0 + w, y0, y0 + h), origin="lower", zorder=z,
                        cmap=LinearSegmentedColormap.from_list("", colors), interpolation="bicubic")
    image.set_clip_path(patch)


def corner(points, centre, radius, start, end):
    angles = np.linspace(np.radians(start), np.radians(end), 12)
    points += [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def page(x, y, w, h, fold, radius):
    """A document outline with rounded corners and a folded top-right corner."""
    points = []
    corner(points, (x + radius, y + radius), radius, 180, 270)
    corner(points, (x + w - radius, y + radius), radius, 270, 360)
    points += [(x + w, y + h - fold), (x + w - fold, y + h)]
    corner(points, (x + radius, y + h - radius), radius, 90, 180)
    return PathPatch(Shape(points + [points[0]], closed=True))


def onnx_mark() -> np.ndarray:
    """The ONNX mark from docs/onnx.png, without its wordmark, recoloured in the accent."""
    source = np.asarray(Image.open(DOCS / "onnx.png").convert("RGBA")).astype(float) / 255
    mark = source[:575]
    rows, columns = np.where(mark[..., 3] > 0.05)
    mark = mark[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    lightness, alpha = mark[..., :3].mean(axis=2), mark[..., 3]
    line = lightness < 0.5
    facets = LinearSegmentedColormap.from_list("", [MUTED, SOFT, BASE])
    shade = np.clip((lightness - 0.6) / 0.38, 0, 1)
    image = np.empty_like(mark)
    image[..., :3] = np.where(line[..., None], matplotlib.colors.to_rgb(STRONG), facets(shade)[..., :3])
    image[..., 3] = np.where(line, alpha, alpha * 0.9)
    return image


def document(axis, x, y, mark: np.ndarray) -> float:
    """One model file, drawn as in every other project: the page, the ONNX mark and its wordmark."""
    w, h, fold = 2.55, 3.4, 0.62
    for spread, alpha in ((0.32, 0.04), (0.2, 0.06), (0.1, 0.09), (0.03, 0.14)):
        shadow = page(x - spread, y - spread - 0.12, w + 2 * spread, h + 2 * spread, fold + spread, 0.2 + spread)
        shadow.set(facecolor="black", edgecolor="none", alpha=alpha, zorder=2)
        axis.add_patch(shadow)
    fill(axis, page(x, y, w, h, fold, 0.2), ["#1D1D20", "#2A2A2E", "#38383D"], z=3)
    rim = page(x, y, w, h, fold, 0.2)
    rim.set(facecolor="none", edgecolor="#4A4A50", linewidth=1, zorder=4)
    axis.add_patch(rim)
    flap = PathPatch(Shape([(x + w - fold, y + h), (x + w - fold, y + h - fold + 0.1), (x + w - fold + 0.1, y + h - fold),
                            (x + w, y + h - fold), (x + w - fold, y + h)], closed=True))
    fill(axis, flap, ["#3E3E44", "#5A5A61"], angle=45, z=4)
    size, cx, label = 1.3, x + w / 2, 0.3
    aspect = mark.shape[0] / mark.shape[1]
    block = size * aspect + 0.36 + label
    top = y + h * 0.47 + block / 2
    axis.imshow(mark, extent=(cx - size / 2, cx + size / 2, top - size * aspect, top), zorder=6, interpolation="lanczos")
    axis.text(cx, top - block + label / 2, "ONNX", ha="center", va="center", fontsize=15, fontweight="medium",
              color="#8E8E93", alpha=0.85, zorder=6)
    return cx


def details(task: str) -> str:
    entry = json.loads((MODELS / "model.json").read_text())[task]
    return f"{entry['file']}  \u00b7  {entry['parameters'] / 1e3:.0f} thousand parameters  \u00b7  {entry['size_kb']} KB"


def file_image(mark: np.ndarray) -> tuple[np.ndarray, tuple]:
    """One model file drawn on its own, with its shadow, as an image to place and turn."""
    import io

    margin = 0.5
    w, h = 2.55, 3.4
    figure, axis = plt.subplots(figsize=((w + 2 * margin) / 1.45, (h + 2 * margin) / 1.45), dpi=400)
    figure.subplots_adjust(0, 0, 1, 1)
    axis.set_xlim(-margin, w + margin)
    axis.set_ylim(-margin, h + margin)
    axis.set_aspect("equal")
    axis.axis("off")
    document(axis, 0, 0, mark)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", transparent=True)
    plt.close(figure)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGBA")), (-margin, w + margin, -margin, h + margin)


def fanned(axis, image: np.ndarray, extent: tuple, x: float, y: float, angle: float, pivot: tuple, z: int) -> None:
    """Place a file with its lower-left corner at (x, y), turned about the pivot, like a card in a hand."""
    from matplotlib.transforms import Affine2D

    left, right, bottom, top = extent
    shown = axis.imshow(image, extent=(x + left, x + right, y + bottom, y + top), zorder=z, interpolation="lanczos")
    shown.set_transform(Affine2D().rotate_deg_around(*pivot, angle) + axis.transData)


def model_files() -> None:
    """The released model shown as what it is: one model in two ONNX files, fanned out."""
    style.apply()
    width, height = 6 * CARD_W + 5 * GAP + 2 * MARGIN, 7.6
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)
    image, extent = file_image(onnx_mark())
    w, y = 2.55, 2.95
    pivot = (width / 2, y - 1.5)                                   # below the pages, like a hand of cards
    fanned(axis, image, extent, width / 2 - w / 2 - 0.5, y, 8, pivot, z=5)    # monitor, behind and to the left
    fanned(axis, image, extent, width / 2 - w / 2 + 0.5, y, -6, pivot, z=6)   # corrector, in front and to the right
    axis.text(width / 2, 2.0, "Calibration Monitor\u2013Corrector Model", ha="center", va="center", fontsize=14,
              fontweight="semibold", color=style.NEUTRAL["ink"], zorder=20)
    for row, task in enumerate(("monitor", "corrector")):
        axis.text(width / 2, 1.5 - row * 0.38, details(task), ha="center", va="center", fontsize=9,
                  color=style.NEUTRAL["secondary"], zorder=20)
    style.save(figure, DOCS, "model", panel=False)


if __name__ == "__main__":
    model_files()
    architecture()
