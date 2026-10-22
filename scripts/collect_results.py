"""Collect runs/<exp>/eval_test_<dataset>.json into a markdown table, mean ± std over seeds.

Runs named <group>_s<seed> are grouped; anything else is its own group.
python scripts/collect_results.py [--runs runs/drive_*] [--metrics dice cldice px_f1 ...]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["px_f1", "px_auc", "px_se", "px_sp", "dice", "iou", "cldice", "betti0_err_patch",
           "betti1_err_patch", "hd95", "seg_err_frac", "rel_f1", "ods_f1", "ois_f1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--metrics", nargs="+", default=DEFAULT)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = [Path(r) for r in args.runs] if args.runs else sorted((ROOT / "runs").iterdir())
    groups = defaultdict(list)   # (group, dataset) -> list of summaries
    for run in runs:
        for f in sorted(run.glob(f"eval_test_*{args.tag}.json")):
            ds = f.stem[len("eval_test_"):len(f.stem) - len(args.tag)]
            group = re.sub(r"_s\d+$", "", run.name)
            groups[(group, ds)].append(json.loads(f.read_text())["summary"])
    lines = ["| method | dataset | n seeds | " + " | ".join(args.metrics) + " |",
             "|---|---|---|" + "---|" * len(args.metrics)]
    for (group, ds), sums in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = []
        for m in args.metrics:
            v = np.array([s[m] for s in sums if m in s], dtype=float)
            cells.append("–" if not len(v) else (f"{v.mean():.4f}" if len(v) == 1 else
                                                f"{v.mean():.4f}±{v.std():.4f}"))
        lines.append(f"| {group} | {ds} | {len(sums)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
