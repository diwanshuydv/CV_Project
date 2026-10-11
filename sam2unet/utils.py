"""Device selection, seeding, logging and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """CUDA -> Apple MPS -> CPU (or an explicit ``prefer`` string)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """bf16 where supported (A30, RTX PRO 6000), fp16 on older CUDA GPUs, None = no AMP."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor in a batch dict to ``device`` (non-blocking with pinned memory)."""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def get_logger(out_dir: Path, name: str = "sam2unet") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False  # sam2/hydra configure the root logger -> avoid duplicate lines
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "log.txt")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, best: float = 0.0, args: Optional[dict] = None) -> None:
    """Saves only LoRA + decoder weights (the frozen SAM 2 trunk is reloaded from its .pt)."""
    state = {
        "model": model.trainable_state_dict(),
        "epoch": epoch,
        "best": best,
        "args": args or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a SLURM time-out never leaves a corrupt checkpoint
