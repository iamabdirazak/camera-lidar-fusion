"""Exporting the trained monitor and corrector to ONNX, the format the vehicle runs.

    python -m src.export.onnx
"""
import json

import onnx
import torch

from src.paths import HEIGHT, MODELS, REPORTS, WIDTH
from src.training.train import load


def export(task: str) -> None:
    model = load(task)
    example = torch.zeros(1, 2, HEIGHT, WIDTH)
    path = MODELS / f"{task}.onnx"
    torch.onnx.export(model, example, path, input_names=["picture"], output_names=["output"],
                      dynamic_axes={"picture": {0: "frames"}, "output": {0: "frames"}}, opset_version=17,
                      dynamo=False)
    onnx.checker.check_model(onnx.load(path))


def card() -> None:
    """The model card: what each file is, what it reads and returns, and the envelope it was tested in."""
    results = json.loads((REPORTS / "results.json").read_text())
    files = {}
    for task in ("monitor", "corrector"):
        model = load(task)
        files[task] = {"file": f"{task}.onnx", "size_kb": round((MODELS / f"{task}.onnx").stat().st_size / 1024),
                       "parameters": sum(p.numel() for p in model.parameters())}
    card = {
        "name": "Calibration Monitor–Corrector Model",
        "input": f"picture: frames x 2 x {HEIGHT} x {WIDTH}, normalised grey image and LiDAR inverse depth "
                 "projected with the current calibration",
        "monitor": {**files["monitor"], "output": "probability the calibration is drifting (rotation over 1 degree)"},
        "corrector": {**files["corrector"], "output": "rotation misalignment as a rotation vector in degrees "
                      "(camera frame); pitch and roll are applied, yaw is not"},
        "data": "KITTI raw, 2011_09_26: 9 training drives, drive 0039 for validation, drive 0020 for testing",
        "results": results,
    }
    (MODELS / "model.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    export("monitor")
    export("corrector")
    if (REPORTS / "results.json").exists():
        card()
