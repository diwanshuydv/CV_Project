"""
Stand-alone (cross-domain) evaluation of trained SAM2-UNet checkpoints.

Builds a source -> target matrix of IoU / Dice / HD95 for any number of runs:

python evaluate.py --runs runs/drive_bnd runs/roads_bnd runs/drive_base runs/roads_base \
                   --datasets drive roads --out runs/cross_domain.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sam2unet.datasets import DataConfig, build_dataset, build_loader
from sam2unet.engine import evaluate
from sam2unet.model import SAM2_VARIANTS, SAM2UNet
from sam2unet.utils import get_device, save_json

ROOT = Path("/scratch/b23cs1017/cv")


def load_model(run_dir: Path, device, ckpt_name: str = "best.pt") -> tuple:
    state = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    a = state["args"]
    sam_ckpt = a.get("sam2_ckpt") or str(ROOT / "checkpoints" / SAM2_VARIANTS[a["variant"]][1])
    model = SAM2UNet(a["variant"], sam_ckpt, a["lora_r"], a["lora_alpha"], a["lora_dropout"],
                     deep_supervision=not a.get("no_deep_supervision", False))
    model.load_trainable_state_dict(state["model"])
    return model.to(device).eval(), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run directories containing best.pt")
    p.add_argument("--datasets", nargs="+", default=["drive", "roads"])
    p.add_argument("--split", default="test")
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--test_scale", type=float, default=1.0,
                   help="resample target images (e.g. 0.5 to shrink 15 px roads toward vessel widths)")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "runs" / "cross_domain.md"))
    args = p.parse_args()

    device = get_device()
    dcfg = DataConfig(data_root=args.data_root)
    rows, full = [], {}
    for run in map(Path, args.runs):
        model, a = load_model(run, device)
        dcfg.seed, dcfg.drive_fold = a.get("seed", 42), a.get("drive_fold")  # same DRIVE split as training
        for name in args.datasets:
            loader = build_loader(build_dataset(name, args.split, dcfg), 1, args.num_workers)
            r = evaluate(model, loader, device, args.window, args.overlap,
                         save_dir=run / f"preds_{args.split}_{name}" if args.save_preds else None,
                         test_scale=args.test_scale)
            s = r["summary"]
            full[f"{run.name}->{name}"] = r
            rows.append((run.name, a["train_dataset"], name, a.get("lambda_max", 0), s))
            print(f"{run.name}: {a['train_dataset']} -> {name}  IoU {s['iou']:.4f}  Dice {s['dice']:.4f}  "
                  f"HD95 {s['hd95']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    lines = ["| run | train | test | domain | λ_max | IoU | Dice | HD95 (px) | ASSD (px) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run, src, tgt, lam, s in rows:
        dom = "in" if src == tgt else "cross"
        lines.append(f"| {run} | {src} | {tgt} | {dom} | {lam} | {s['iou']:.4f}±{s['iou_std']:.3f} | "
                     f"{s['dice']:.4f}±{s['dice_std']:.3f} | {s['hd95']:.2f}±{s['hd95_std']:.2f} | "
                     f"{s['assd']:.2f} |")
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    save_json(full, out.with_suffix(".json"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
