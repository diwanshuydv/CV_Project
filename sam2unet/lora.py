"""
Step 2a -- Low-Rank Adaptation (LoRA) for the frozen SAM 2 Hiera encoder.

LoRA (Hu et al., 2021) re-parameterises the update of a frozen weight matrix
W0 in R^{d_out x d_in} as a low-rank product:

        h = W0 x + (alpha / r) * B A x,      A in R^{r x d_in},  B in R^{d_out x r}

Only A and B are trained (r << min(d_in, d_out)).  B is zero-initialised so the
adapted network is *exactly* the pre-trained SAM 2 encoder at step 0, which
keeps early optimisation stable.

In Hiera every ``MultiScaleBlock`` contains
    blk.attn.qkv          nn.Linear(dim, 3*dim_out)    -- fused Q/K/V projection
    blk.attn.proj         nn.Linear(dim_out, dim_out)  -- attention output projection
    blk.mlp.layers[0/1]   nn.Linear                    -- two-layer MLP
``inject_lora`` replaces any of these (selected by name suffix) with a
``LoRALinear`` wrapper.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():          # base weights stay frozen
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Kaiming-uniform for A (as in the reference implementation), zeros for B.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.merged = False

    # expose the attributes Hiera code may read from the original layer
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold BA into W0 for zero-overhead inference (irreversible in-place)."""
        if not self.merged:
            self.base.weight += (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.merged = True

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, scaling={self.scaling:g}"


DEFAULT_TARGETS: Tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.layers.0", "mlp.layers.1")


def inject_lora(model: nn.Module, targets: Iterable[str] = DEFAULT_TARGETS, r: int = 8,
                alpha: float = 16.0, dropout: float = 0.0) -> List[str]:
    """Replace every ``nn.Linear`` whose qualified name ends with one of ``targets``.

    Returns the list of replaced module names (useful for logging / sanity checks).
    """
    targets = tuple(targets)
    to_replace = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name.endswith(targets)
    ]
    for name in to_replace:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r=r, alpha=alpha, dropout=dropout))
    return to_replace


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
