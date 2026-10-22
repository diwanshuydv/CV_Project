"""
Step 4b -- Training / evaluation engine.

* ``build_optimizer``      AdamW with separate LR groups for LoRA and decoder
* ``build_scheduler``      linear warm-up -> CosineAnnealingLR (stepped per iteration)
* ``train_one_epoch``      AMP training loop with gradient clipping
* ``sliding_window_predict`` full-resolution inference with Gaussian-blended tiles
* ``evaluate``             IoU / Dice / HD95 / HD / ASSD over a whole split
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .metrics import PixelStats, aggregate, compute_metrics
from .utils import amp_dtype, to_device


# --------------------------------------------------------------------------- #
#                           Optimiser / scheduler                             #
# --------------------------------------------------------------------------- #
def build_optimizer(model, lr_lora: float = 2e-4, lr_decoder: float = 1e-3,
                    weight_decay: float = 1e-4) -> AdamW:
    groups = [
        {"params": model.lora_parameters(), "lr": lr_lora, "name": "lora"},
        {"params": model.decoder_parameters(), "lr": lr_decoder, "name": "decoder"},
    ]
    return AdamW(groups, weight_decay=weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer, total_iters: int, warmup_iters: int, min_lr_ratio: float = 0.01):
    """Linear warm-up followed by cosine annealing to ``min_lr_ratio * lr``."""
    warmup_iters = max(1, min(warmup_iters, total_iters - 1))
    eta_min = min(g["lr"] for g in optimizer.param_groups) * min_lr_ratio
    warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cos = CosineAnnealingLR(optimizer, T_max=max(1, total_iters - warmup_iters), eta_min=eta_min)
    return SequentialLR(optimizer, [warm, cos], milestones=[warmup_iters])


# --------------------------------------------------------------------------- #
#                                  Training                                   #
# --------------------------------------------------------------------------- #
def _frame_targets(batch, box, window: int):
    """Targets of one frame of a region: a crop for tiles, area-downsampled for the global view
    (soft mask; skeleton max-pooled so thin centrelines survive; no distance map)."""
    y, x, size = box
    if size == window:
        sl = (..., slice(y, y + size), slice(x, x + size))
        return (batch["mask"][sl], batch["dist"][sl], batch["fov"][sl],
                batch["skel"][sl] if "skel" in batch else None)
    f = size // window
    mask = F.avg_pool2d(batch["mask"], f)
    fov = (F.avg_pool2d(batch["fov"], f) > 0.5).float()
    skel = F.max_pool2d(batch["skel"], f) * fov if "skel" in batch else None
    return mask, None, fov, skel


def region_loss(res, batch, loss_fn, window: int, w_pass1: float = 0.5):
    """Mean loss over the memory-conditioned tiles + ``w_pass1`` x mean over pass-1 frames."""
    final = [loss_fn(out, *_frame_targets(batch, box, window)) for out, box in res["final"]]
    first = [loss_fn(out, *_frame_targets(batch, box, window))[0] for out, box in res["frames"]]
    loss = sum(l for l, _ in final) / len(final) + w_pass1 * sum(first) / len(first)
    logs = {k: sum(lg[k] for _, lg in final) / len(final) for k in final[0][1]}
    logs["pass1"] = (sum(first) / len(first)).detach()
    logs["total"] = loss.detach()
    return loss, logs


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device,
                    epoch: int, total_iters: int, logger=None, writer=None,
                    grad_clip: float = 1.0, log_every: int = 20, w_pass1: float = 0.5) -> Dict[str, float]:
    model.train()
    region = getattr(model, "region_training", False)
    dtype = amp_dtype(device)
    iters_per_epoch = len(loader)
    sums: Dict[str, float] = {}
    t0 = time.time()

    for it, batch in enumerate(loader):
        step = epoch * iters_per_epoch + it
        loss_fn.set_progress(step / max(total_iters, 1))   # lambda(t) ramp
        batch = to_device(batch, device)

        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = model.forward_region(batch["image"]) if region else model(batch["image"])
        if region:
            loss, logs = region_loss(out, batch, loss_fn, model.window, w_pass1)
        else:
            loss, logs = loss_fn(out, batch["mask"], batch["dist"], batch["fov"], batch.get("skel"))

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        params = [p for g in optimizer.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {logs}")
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        if writer is not None and step % log_every == 0:
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", float(v), step)
            writer.add_scalar("train/grad_norm", float(gnorm), step)
            for g in optimizer.param_groups:
                writer.add_scalar(f"lr/{g['name']}", g["lr"], step)
        if logger is not None and it % log_every == 0:
            logger.info(f"ep {epoch} it {it}/{iters_per_epoch} loss {float(loss):.4f} "
                        f"bce {float(logs['bce']):.4f} dice {float(logs['dice']):.4f} "
                        f"bnd {float(logs['boundary']):.4f} lam {float(logs['lambda']):.3f} "
                        + "".join(f"{k} {float(logs[k]):.4f} " for k in ("skel", "cldice", "pass1") if k in logs) +
                        f"lr {optimizer.param_groups[0]['lr']:.2e} ({time.time() - t0:.0f}s)")

    return {k: v / max(iters_per_epoch, 1) for k, v in sums.items()}


# --------------------------------------------------------------------------- #
#                                 Inference                                   #
# --------------------------------------------------------------------------- #
def _gaussian_window(size: int, sigma_scale: float = 0.25) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (sigma_scale * size) ** 2))
    w = g[:, None] * g[None, :]
    return (w / w.max()).clamp_min(1e-3)


def _starts(length: int, window: int, stride: int):
    if length <= window:
        return [0]
    s = list(range(0, length - window, stride))
    return s + [length - window]


@torch.no_grad()
def sliding_window_predict(model, image: torch.Tensor, window: int = 512, overlap: float = 0.25,
                           batch_size: int = 8) -> torch.Tensor:
    """Full-image foreground probabilities for one image (1,3,H,W) -> (1,1,H,W).

    The image is reflect-padded to >= window and to a multiple of 32 (Hiera's
    total stride), tiled with the given overlap, and tile predictions are
    blended with a Gaussian weight to suppress seams at tile borders.
    """
    model.eval()
    device = image.device
    dtype = amp_dtype(device)
    _, _, H, W = image.shape
    Hp = max(window, math.ceil(H / 32) * 32)
    Wp = max(window, math.ceil(W / 32) * 32)
    mode = "reflect" if (Hp - H) < H and (Wp - W) < W else "constant"
    x = F.pad(image, (0, Wp - W, 0, Hp - H), mode=mode)

    stride = max(32, int(window * (1 - overlap)))
    wmap = _gaussian_window(window).to(device)
    prob = torch.zeros(1, 1, Hp, Wp, device=device)
    norm = torch.zeros_like(prob)
    coords = [(y, xx) for y in _starts(Hp, window, stride) for xx in _starts(Wp, window, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        tiles = torch.cat([x[..., y:y + window, xx:xx + window] for y, xx in chunk])
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(tiles)
        p = torch.sigmoid(logits.float())
        for (y, xx), pt in zip(chunk, p):
            prob[..., y:y + window, xx:xx + window] += pt * wmap
            norm[..., y:y + window, xx:xx + window] += wmap
    return (prob / norm.clamp_min(1e-6))[..., :H, :W]


def predict_full(model, image: torch.Tensor, window: int = 512, overlap: float = 0.25,
                 tta: bool = False) -> torch.Tensor:
    """Full-image probabilities; models with their own whole-image inference (tile memory)
    provide ``predict_full``, everything else uses the Gaussian-blended sliding window.
    ``tta``: average over the identity and the horizontal / vertical / both flips."""
    def run(x):
        if hasattr(model, "predict_full"):
            return model.predict_full(x, window, overlap)
        return sliding_window_predict(model, x, window, overlap)

    if not tta:
        return run(image)
    flips = [(), (-1,), (-2,), (-2, -1)]
    return sum(torch.flip(run(torch.flip(image, f)), f) if f else run(image) for f in flips) / len(flips)


@torch.no_grad()
def evaluate(model, loader, device, window: int = 512, overlap: float = 0.25,
             threshold: float = 0.5, save_dir: Optional[Path] = None,
             max_images: Optional[int] = None, test_scale: float = 1.0,
             topology: bool = True, tta: bool = False) -> Dict[str, object]:
    """Per-image metrics inside the FOV + dataset aggregate + pixel-pooled metrics.

    ``threshold`` binarises the probabilities for the per-image metrics and the pooled
    Se/Sp/Acc/F1; the pooled summary also reports AUC, ODS/OIS and the F1-optimal
    threshold (``ods_thr``, used to pick the test threshold on the validation set).
    ``topology=False`` skips clDice / Betti / relaxed metrics (fast validation).
    ``test_scale`` != 1 resamples the input before inference (and the prediction
    back to the original size) -- handy for cross-domain tests where structure
    widths differ (e.g. 1-8 px vessels vs 10-20 px roads).
    """
    model.eval()
    rows = []
    stats = PixelStats()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for n, batch in enumerate(loader):
        if max_images is not None and n >= max_images:
            break
        batch = to_device(batch, device)
        img = batch["image"]
        if test_scale != 1.0:
            H, W = img.shape[-2:]
            img_s = F.interpolate(img, scale_factor=test_scale, mode="bilinear", align_corners=False)
            prob = F.interpolate(predict_full(model, img_s, window, overlap, tta),
                                 size=(H, W), mode="bilinear", align_corners=False)
        else:
            prob = predict_full(model, img, window, overlap, tta)
        prob = prob[0, 0].float().cpu().numpy()
        pred = prob > threshold
        gt = batch["mask"][0, 0].cpu().numpy() > 0.5
        fov = batch["fov"][0, 0].cpu().numpy() > 0.5
        stats.update(prob, gt, fov)
        m = compute_metrics(pred, gt, fov, topology=topology)
        m["name"] = batch["name"][0]
        rows.append(m)
        if save_dir is not None:
            Image.fromarray((pred & fov).astype(np.uint8) * 255).save(save_dir / f"{m['name']}_pred.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(save_dir / f"{m['name']}_prob.png")
    summary = aggregate({k: v for k, v in r.items() if k != "name"} for r in rows)
    summary.update(stats.summary(threshold))
    summary["threshold"] = threshold
    return {"summary": summary, "per_image": rows}
