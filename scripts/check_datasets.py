"""Sanity-check every dataset split: counts, image/label/FOV shapes, foreground and FOV fractions.

python scripts/check_datasets.py [--full]     (--full reads every image, default: first 3 per split)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2unet.datasets import DATASETS, BinarySegDataset, DataConfig, build_records  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = p.parse_args()
    cfg = DataConfig(cache_images=False)
    for name in args.datasets:
        for split in ("train", "val", "test"):
            try:
                recs = build_records(name, split, cfg)
            except ValueError:
                continue
            ds = BinarySegDataset(recs, "eval", cfg)
            n = len(recs) if args.full else min(3, len(recs))
            fg, fv, shapes = [], [], set()
            for i in range(n):
                img, lbl, fov = ds._read(i)
                assert img.shape[:2] == lbl.shape == fov.shape, (recs[i], img.shape, lbl.shape, fov.shape)
                fg.append(lbl[fov > 0].mean())
                fv.append(fov.mean())
                shapes.add(lbl.shape)
            print(f"{name:13s} {split:5s} n={len(recs):5d}  shapes={sorted(shapes)[:2]}  "
                  f"fg/FOV={np.mean(fg):.3f}  FOV={np.mean(fv):.3f}  e.g. {recs[0].name}", flush=True)


if __name__ == "__main__":
    main()
