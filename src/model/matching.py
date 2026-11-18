"""Matching image and depth by correlation, in one direction, pooled over bands of the frame.

Both models are built from this. Each stream turns the grey image and the depth into
features beside their edges in one direction, compares them over a range of offsets in
that direction, and averages the comparison over bands of the frame, each cell weighted
by how much it trusts its own evidence. The result is a match profile per band: where
image and depth agree best, and how clearly. A correct calibration peaks at no offset.

Correlation, never position. The models see only these profiles, so they cannot learn
a rotation from where the LiDAR band sits in the frame, which is what a first version
did. The aperture problem decides the two directions: an edge only shows motion across
itself, so sideways slides are read from upright edges and up-down slides from level
ones. Upright edges are sparse, which is why evidence is pooled before any decision.
"""
import torch
from torch import nn

from src.paths import WIDTH


def band_weights(width: int, count: int) -> torch.Tensor:
    """Averaging weights from columns to bands, the same windows adaptive average pooling uses.

    Written as a matrix so the pooling exports to ONNX for any number of bands.
    """
    weights = torch.zeros(width, count)
    for band in range(count):
        start, end = (band * width) // count, -((-(band + 1) * width) // count)
        weights[start:end, band] = 1.0 / (end - start)
    return weights


def layer(inputs: int, outputs: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class Edges(nn.Module):
    """The signal beside its edges in one direction, from a fixed Sobel filter."""

    def __init__(self, direction: str):
        super().__init__()
        sobel = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
        self.register_buffer("kernel", (sobel if direction == "sideways" else sobel.T).reshape(1, 1, 3, 3))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.cat([signal, torch.abs(nn.functional.conv2d(signal, self.kernel, padding=1))], 1)


def encoder(direction: str, width: int) -> nn.Sequential:
    """Features at a quarter of the resolution, where one pixel is about one degree of rotation."""
    return nn.Sequential(Edges(direction), layer(2, width, 2), layer(width, width, 2),
                         nn.Conv2d(width, width, 3, padding=1))


class Stream(nn.Module):
    """Match profiles for slides in one direction, pooled over bands of the frame."""

    def __init__(self, direction: str, reach: int, bands: int, width: int):
        super().__init__()
        self.direction, self.reach, self.bands = direction, reach, bands
        self.image, self.depth = encoder(direction, width), encoder(direction, width)
        self.trust = nn.Sequential(nn.Conv2d(2 * width, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.register_buffer("pool", band_weights(WIDTH // 4, bands), persistent=False)

    @property
    def size(self) -> int:
        return (2 * self.reach + 1) * self.bands

    def forward(self, picture: torch.Tensor) -> torch.Tensor:
        depth = picture[:, 1:]
        if self.direction == "sideways":   # join each scan line's points into short columns, so upright edges read as edges
            depth = nn.functional.max_pool2d(depth, (5, 1), stride=1, padding=(2, 0))
        image, depth = self.image(picture[:, :1]), self.depth(depth)
        weight = torch.sigmoid(self.trust(torch.cat([image, depth], 1)))
        image, depth = nn.functional.normalize(image, dim=1), nn.functional.normalize(depth, dim=1)
        r, (height, width) = self.reach, image.shape[2:]
        if self.direction == "sideways":
            padded = nn.functional.pad(depth, (r, r, 0, 0))
            cost = torch.cat([(image * padded[:, :, :, k:k + width]).sum(1, keepdim=True) for k in range(2 * r + 1)], 1)
        else:
            padded = nn.functional.pad(depth, (0, 0, r, r))
            cost = torch.cat([(image * padded[:, :, k:k + height]).sum(1, keepdim=True) for k in range(2 * r + 1)], 1)
        # Trust-weighted average over each band of columns: one match profile per band.
        weighted = (cost * weight).mean(2) @ self.pool
        total = weight.mean(2) @ self.pool + 1e-4
        return (weighted / total).flatten(1)


