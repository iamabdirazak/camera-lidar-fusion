"""The monitor: a small model that runs on every frame and asks one question, aligned or drifting.

It is built from the same matching streams as the corrector (src/model/matching.py),
made light: narrower features and a short reach of about 3.5 degrees either way, since it
only has to tell whether image and depth agree best at no offset, not by how much they
disagree. Its evidence is pooled into two bands for sideways slides and six for up-down ones,
because roll shows only as a tilt from left to right, and read by a single logistic unit. Keeping it this small is what lets it run on every frame.
"""
import torch
from torch import nn

from src.model.matching import Stream


class Monitor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sideways = Stream("sideways", reach=4, bands=2, width=12)
        self.upright = Stream("upright", reach=4, bands=6, width=12)
        self.score = nn.Linear(self.sideways.size + self.upright.size, 1)

    def forward(self, picture: torch.Tensor) -> torch.Tensor:
        """The probability that the calibration is drifting, one per picture."""
        profiles = torch.cat([self.sideways(picture), self.upright(picture)], 1)
        return torch.sigmoid(self.score(profiles * 10)).squeeze(1)
