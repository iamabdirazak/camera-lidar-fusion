"""Training the monitor and the corrector on self-supervised samples, on a CPU.

    python -m src.training.train monitor
    python -m src.training.train corrector

Each run is a fixed number of steps on fresh samples, checked against a fixed
validation set drawn from frames it never trains on. The weights with the best
validation score are kept.
"""
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from src.acquisition.kitti import Frames
from src.model.corrector import SCALE, Corrector
from src.model.monitor import Monitor
from src.paths import CACHE, SEED, THREADS
from src.training.samples import Samples, split

SETTINGS = {
    "monitor": {"steps": 1500, "batch": 32, "rate": 2e-3},
    "corrector": {"steps": 3000, "batch": 32, "rate": 1e-3},
}
CHECK_EVERY = 150
VALIDATION_SAMPLES = 384


def loss(task: str, output: torch.Tensor, offset: torch.Tensor, drifting: torch.Tensor) -> torch.Tensor:
    if task == "monitor":
        return nn.functional.binary_cross_entropy(output, drifting)
    scale = torch.tensor(SCALE)
    return nn.functional.smooth_l1_loss(output / scale, offset / scale, beta=0.05)


def validate(task: str, model: nn.Module, loader) -> float:
    """Monitor: the share of samples wrongly judged. Corrector: mean leftover error in units of the range."""
    model.eval()
    scores = []
    with torch.no_grad():
        for picture, offset, drifting in loader:
            output = model(picture)
            if task == "monitor":
                scores.append(((output > 0.5).float() != drifting).float())
            else:
                scores.append((torch.abs(output - offset) / torch.tensor(SCALE)).mean(1))
    model.train()
    return torch.cat(scores).mean().item()


def train(task: str) -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    settings = SETTINGS[task]
    frames = Frames()
    training, validation = split(frames)
    loader = torch.utils.data.DataLoader(Samples(frames, training, task, settings["steps"] * settings["batch"], SEED),
                                         batch_size=settings["batch"], num_workers=1, persistent_workers=True)
    held_out = torch.utils.data.DataLoader(Samples(frames, validation, task, VALIDATION_SAMPLES, SEED + 1, fixed=True),
                                           batch_size=64, num_workers=1)
    model = Monitor() if task == "monitor" else Corrector()
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings["rate"], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, settings["rate"], total_steps=settings["steps"])

    best, history, start = float("inf"), [], time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    for step, (picture, offset, drifting) in enumerate(loader, 1):
        value = loss(task, model(picture), offset, drifting)
        optimiser.zero_grad()
        value.backward()
        optimiser.step()
        schedule.step()
        if step % CHECK_EVERY == 0 or step == settings["steps"]:
            score = validate(task, model, held_out)
            history.append({"step": step, "loss": round(value.item(), 4), "validation": round(score, 4)})
            print(f"{task} {step}/{settings['steps']}  loss {value.item():.4f}  validation {score:.4f}"
                  f"  {time.time() - start:.0f} s", flush=True)
            if score < best:
                best = score
                torch.save(model.state_dict(), CACHE / f"{task}.pt")
    (CACHE / f"{task}-training.json").write_text(json.dumps(
        {"task": task, **settings, "best_validation": round(best, 4), "minutes": round((time.time() - start) / 60, 1),
         "history": history}, indent=2))


def load(task: str) -> nn.Module:
    model = Monitor() if task == "monitor" else Corrector()
    model.load_state_dict(torch.load(CACHE / f"{task}.pt"))
    return model.eval()


if __name__ == "__main__":
    train(sys.argv[1])
