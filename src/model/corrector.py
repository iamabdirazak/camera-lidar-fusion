"""The corrector: a model woken only when the monitor reports drift, which estimates the rotation.

Three findings shaped it.

Position is a shortcut. A first version read image and depth side by side. Pitch and
roll move the band the LiDAR covers within the frame, and the model learned them from
the band's position alone, never from the scene, and so never learned yaw at all.

Edges only show motion across themselves (the aperture problem). The LiDAR's scan lines
are level, so up-down slides are visible everywhere, while a sideways slide, which is
what yaw causes, shows only at upright edges: poles, car sides, tree trunks.

Upright edges are sparse. Matched cell by cell, most cells have none and their sideways
slide is noise. Summed over the frame first, the same evidence gives a clear answer.

So the corrector reads match profiles from two streams (src/model/matching.py). The
sideways stream pools over four wide column bands. The up-down stream pools over eight
narrower ones, because roll tilts the slide from left to right, and the change across
bands is what separates roll from pitch. A small head reads the profiles and returns
the rotation as a rotation vector in degrees.
"""
import torch
from torch import nn

from src.model.matching import Stream
from src.paths import RANGE_DEGREES

SCALE = (RANGE_DEGREES,) * 3     # outputs are learned in units of the range


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.sideways = Stream("sideways", reach=12, bands=4, width=24)   # up to about 10 degrees of yaw
        self.upright = Stream("upright", reach=8, bands=8, width=24)      # up to about 7 degrees of pitch
        self.head = nn.Sequential(nn.Linear(self.sideways.size + self.upright.size, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 3))

    def forward(self, picture: torch.Tensor) -> torch.Tensor:
        """The estimated rotation misalignment, as a rotation vector in degrees."""
        profiles = torch.cat([self.sideways(picture), self.upright(picture)], 1)
        return self.head(profiles * 10) * RANGE_DEGREES
