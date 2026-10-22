"""Build a model from a run's argument dict (shared by train.py and evaluate.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .model import SAM2_VARIANTS, SAM2UNet

ROOT = Path("/scratch/b23cs1017/cv")


def build_model(a: Dict[str, Any]):
    """``a``: vars(args) of train.py (older runs lack the newer keys -> baseline SAM2-UNet)."""
    ckpt = a.get("sam2_ckpt") or str(ROOT / "checkpoints" / SAM2_VARIANTS[a["variant"]][1])
    common = dict(lora_r=a["lora_r"], lora_alpha=a["lora_alpha"], lora_dropout=a["lora_dropout"],
                  grad_checkpointing=a.get("grad_ckpt", False),
                  deep_supervision=not a.get("no_deep_supervision", False))
    if a.get("model", "sam2unet") == "sam2unet":
        return SAM2UNet(a["variant"], ckpt, **common)
    from .memory import SAM2MemUNet

    memory = a.get("memory", "both")
    model = SAM2MemUNet(a["variant"], ckpt, window=a["window"], region=a["region"],
                        use_global=memory in ("global", "both"), use_tiles=memory in ("tiles", "both"),
                        mem_lora_r=a.get("mem_lora_r", 8), **common)
    model.region_training = True
    return model
