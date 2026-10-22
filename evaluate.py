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
from sam2unet.factory import build_model
from sam2unet.utils import get_device, save_json

ROOT = Path("/scratch/b23cs1017/cv")


def load_model(run_dir: Path, device, ckpt_name: str = None) -> tuple:
    """best.pt, or last.pt for train+val runs (no held-out validation to select on)."""
    if ckpt_name is None:
        a = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["args"]
        ckpt_name = "last.pt" if a.get("trainval") else "best.pt"
    state = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    a = state["args"]
    model = build_model(a)
    model.load_trainable_state_dict(state["model"])
    return model.to(device).eval(), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run directories containing best.pt")
    p.add_argument("--datasets", nargs="+", default=None, help="default: each run's training dataset")
    p.add_argument("--threshold", default="val",
                   help="'val' = F1-optimal threshold on the run's validation set, or a number")
    p.add_argument("--no_topology", action="store_true")
    p.add_argument("--tta", action="store_true", help="flip test-time augmentation (also on val)")
    p.add_argument("--memory_off", action="store_true",
                   help="tsm runs: predict with the pass-1 (no-memory) outputs of the same weights")
    p.add_argument("--tag", default="", help="suffix of the per-run eval_<split>_<dataset><tag>.json")
    p.add_argument("--split", default="test")
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--test_scale", type=float, default=None,
                   help="resample target images (e.g. 0.5 to shrink 15 px roads toward vessel widths)")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "runs" / "eval.md"))
    args = p.parse_args()

    device = get_device()
    dcfg = DataConfig(data_root=args.data_root)
    rows, full = [], {}
    for run in map(Path, args.runs):
        model, a = load_model(run, device)
        model.memory_off = args.memory_off
        dcfg.seed, dcfg.drive_fold = a.get("seed", 42), a.get("drive_fold")  # same split as training
        scale = args.test_scale or a.get("scale", 1.0)    # default: the run's training scale
        if args.threshold == "val" and not a.get("trainval"):
            val = build_loader(build_dataset(a["train_dataset"], "val", dcfg), 1, args.num_workers)
            thr = evaluate(model, val, device, args.window, args.overlap, topology=False,
                           test_scale=scale, tta=args.tta)["summary"]["ods_thr"]
        else:
            thr = 0.5 if args.threshold == "val" else float(args.threshold)
        for name in args.datasets or [a["train_dataset"]]:
            loader = build_loader(build_dataset(name, args.split, dcfg), 1, args.num_workers)
            r = evaluate(model, loader, device, args.window, args.overlap, threshold=thr,
                         save_dir=run / f"preds_{args.split}_{name}" if args.save_preds else None,
                         test_scale=scale, topology=not args.no_topology, tta=args.tta)
            s = r["summary"]
            full[f"{run.name}->{name}"] = r
            save_json(r, run / f"eval_{args.split}_{name}{args.tag}.json")
            rows.append((run.name, a["train_dataset"], name, s))
            print(f"{run.name}: {a['train_dataset']} -> {name}  thr {thr:.2f}  Dice {s['dice']:.4f}  "
                  f"clDice {s.get('cldice', float('nan')):.4f}  pooled F1 {s['px_f1']:.4f}  "
                  f"AUC {s['px_auc']:.4f}  HD95 {s['hd95']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    nan = float("nan")
    lines = ["| run | train | test | thr | Dice | IoU | clDice | β0 err/patch | seg-err frac | HD95 | "
             "pooled F1 | AUC | Se | Sp | Acc | ODS | OIS | relaxed F1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, src, tgt, s in rows:
        lines.append(f"| {run} | {src} | {tgt} | {s['threshold']:.2f} | {s['dice']:.4f}±{s['dice_std']:.3f} | "
                     f"{s['iou']:.4f} | {s.get('cldice', nan):.4f} | {s.get('betti0_err_patch', nan):.3f} | "
                     f"{s.get('seg_err_frac', nan):.3f} | {s['hd95']:.2f} | {s['px_f1']:.4f} | "
                     f"{s['px_auc']:.4f} | {s['px_se']:.4f} | {s['px_sp']:.4f} | {s['px_acc']:.4f} | "
                     f"{s['ods_f1']:.4f} | {s['ois_f1']:.4f} | {s.get('rel_f1', nan):.4f} |")
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    save_json(full, out.with_suffix(".json"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
