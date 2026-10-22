"""Why do roads-trained models output ~0 on DRIVE?  Probe input transforms.

Runs a roads checkpoint on DRIVE test images under: original, intensity-inverted
(vessels become bright like roads), upscaled (vessels as wide as roads), and both.
A roads test image is included as a positive control.

python scripts/diag_cross.py --run runs/roads_base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import load_model  # noqa: E402
from sam2unet.datasets import IMAGENET_MEAN, IMAGENET_STD, DataConfig, build_dataset  # noqa: E402
from sam2unet.engine import sliding_window_predict  # noqa: E402
from sam2unet.metrics import dice_score  # noqa: E402
from sam2unet.utils import get_device  # noqa: E402

MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def transform(img: torch.Tensor, fov: torch.Tensor, invert: bool, scale: float) -> torch.Tensor:
    """img: normalised (1,3,H,W).  Returns normalised, resampled image."""
    x = img * STD.to(img) + MEAN.to(img)              # back to [0, 1]
    if invert:
        x = 1 - x
        x = x * fov + (1 - fov) * x.new_tensor(0.0)   # keep the area outside the FOV black
    if scale != 1:
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    return (x - MEAN.to(x)) / STD.to(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/roads_base")
    p.add_argument("--n_images", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    model, a = load_model(ROOT / args.run, device)
    cfg = DataConfig()
    cfg.seed, cfg.drive_fold = a.get("seed", 42), a.get("drive_fold")

    roads = build_dataset("roads", "test", cfg)[0]
    pr = sliding_window_predict(model, roads["image"][None].to(device))[0, 0].cpu()
    gt = roads["mask"][0].bool()
    print(f"[control] {roads['name']}: max {pr.max():.3f} mean {pr.mean():.4f} "
          f"Dice@0.5 {dice_score((pr > 0.5).numpy(), gt.numpy()):.3f}", flush=True)

    drive = build_dataset("drive", "test", cfg)
    for i in range(min(args.n_images, len(drive))):
        s = drive[i]
        img, fov, gt = s["image"][None], s["fov"][None], s["mask"][0].numpy() > 0.5
        f = fov[0, 0].numpy() > 0.5
        for invert in (False, True):
            for scale in (1.0, 2.0, 3.0):
                x = transform(img, fov, invert, scale).to(device)
                pb = sliding_window_predict(model, x)
                if scale != 1:
                    pb = F.interpolate(pb, size=gt.shape, mode="bilinear", align_corners=False)
                pb = pb[0, 0].cpu().numpy()
                inside = pb[f]
                best = max((dice_score((pb > t) & f, gt), t) for t in (0.05, 0.1, 0.2, 0.3, 0.5))
                print(f"{s['name']} invert={invert!s:5} scale={scale}: max {inside.max():.3f} "
                      f"mean {inside.mean():.4f} p99 {np.percentile(inside, 99):.3f} "
                      f"Dice@0.5 {dice_score((pb > 0.5) & f, gt):.3f} "
                      f"best Dice {best[0]:.3f}@{best[1]}", flush=True)


if __name__ == "__main__":
    main()
