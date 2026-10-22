"""
Step 2b -- SAM2-UNet: frozen SAM 2.1 Hiera encoder + LoRA + CNN U-Net decoder.

                       image (B,3,H,W)
                            |
          +-----------------+------------------------------+
          |                                                |
   Hiera trunk (frozen, LoRA in every block)        high-res CNN stem
          |                                         (stride 1 and 2 skips; thin
   f1: stride 4  (C1)  -----------------+            1-2 px vessels are lost at
   f2: stride 8  (C2)  ------------+    |            the encoder's stride 4)
   f3: stride 16 (C3)  -------+    |    |                  |
   f4: stride 32 (C4)         |    |    |                  |
          |                   |    |    |                  |
     adapter 1x1          adapters (1x1 conv-BN-ReLU)      |
          |                   |    |    |                  |
       Up+cat+ResBlock  <-----+    |    |                  |      (stride 16)
       Up+cat+ResBlock  <----------+    |                  |      (stride 8)
       Up+cat+ResBlock  <---------------+                  |      (stride 4)
       Up+cat+ResBlock  <----------------------------------+ s2   (stride 2)
       Up+cat+ResBlock  <----------------------------------+ s1   (stride 1)
          |
       1x1 conv -> logits (B,1,H,W)   (+ optional deep-supervision logits)

Hiera-L channel widths are [144, 288, 576, 1152]; they are read from the
trunk at run time so the same code works for the T/S/B+/L variants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .lora import DEFAULT_TARGETS, count_parameters, inject_lora

# config name (inside the installed sam2 package) and checkpoint file per variant
SAM2_VARIANTS: Dict[str, tuple] = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


# --------------------------------------------------------------------------- #
#                                   Encoder                                   #
# --------------------------------------------------------------------------- #
def _enable_grad_checkpointing(blocks: nn.ModuleList) -> None:
    """Activation checkpointing for each Hiera block (trades compute for memory).

    Useful on the 24 GB A30 nodes: gradients must still flow *through* the frozen
    blocks to reach the LoRA parameters, so their activations would otherwise be
    stored.  The block's ``forward`` is patched in place so state-dict keys are
    unchanged (checkpoints load with or without this option).
    """
    for blk in blocks:
        orig = blk.forward

        def fwd(x, _orig=orig):
            if torch.is_grad_enabled():
                return checkpoint(_orig, x, use_reentrant=False)
            return _orig(x)

        blk.forward = fwd


class SAM2HieraEncoder(nn.Module):
    """SAM 2.1 Hiera trunk with all base weights frozen and LoRA injected."""

    def __init__(self, variant: str = "large", checkpoint_path: Optional[str] = None,
                 lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 lora_targets: Sequence[str] = DEFAULT_TARGETS, grad_checkpointing: bool = False,
                 trunk: Optional[nn.Module] = None):
        super().__init__()
        if trunk is None:
            trunk = self._load_trunk(variant, checkpoint_path)
        self.trunk = trunk

        # 1) Freeze EVERYTHING in the pre-trained backbone ...
        self.trunk.requires_grad_(False)
        # 2) ... then add trainable low-rank adapters (only these receive gradients).
        self.lora_modules = inject_lora(self.trunk, lora_targets, r=lora_r,
                                        alpha=lora_alpha, dropout=lora_dropout)
        if grad_checkpointing:
            _enable_grad_checkpointing(self.trunk.blocks)

        # Hiera stores channel_list deepest-first (for the FPN neck); we want shallow-first.
        self.channels: List[int] = list(self.trunk.channel_list)[::-1]

    @staticmethod
    def _load_trunk(variant: str, checkpoint_path: Optional[str]) -> nn.Module:
        from sam2.build_sam import build_sam2  # imported lazily: tests run without sam2

        cfg, _ = SAM2_VARIANTS[variant]
        sam = build_sam2(cfg, checkpoint_path, device="cpu", mode="eval")
        trunk = sam.image_encoder.trunk
        del sam  # drop memory attention, prompt encoder, mask decoder, FPN neck
        return trunk

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns [f1, f2, f3, f4] at strides 4, 8, 16, 32 (NCHW)."""
        return self.trunk(x)


# --------------------------------------------------------------------------- #
#                                   Decoder                                   #
# --------------------------------------------------------------------------- #
def conv_bn_relu(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualConvBlock(nn.Module):
    """(conv3x3-BN-ReLU) x 2 with a 1x1 projection shortcut."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class UpBlock(nn.Module):
    """Bilinear x2 upsampling -> concat skip -> residual conv block."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.block = ResidualConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, enc_channels: Sequence[int], dec_channels: Sequence[int] = (64, 128, 256, 512),
                 stem_channels: Sequence[int] = (32, 48), deep_supervision: bool = True):
        super().__init__()
        assert len(enc_channels) == 4 and len(dec_channels) == 4
        # Adapters: compress the (large) Hiera channels before fusion.
        self.adapters = nn.ModuleList(conv_bn_relu(c, d, k=1) for c, d in zip(enc_channels, dec_channels))
        # High-resolution CNN stem on the raw image (strides 1 and 2).
        s1, s2 = stem_channels
        self.stem1 = ResidualConvBlock(3, s1)
        self.stem2 = nn.Sequential(conv_bn_relu(s1, s2, s=2), ResidualConvBlock(s2, s2))
        d1, d2, d3, d4 = dec_channels
        self.up3 = UpBlock(d4, d3, d3)      # 1/32 -> 1/16
        self.up2 = UpBlock(d3, d2, d2)      # 1/16 -> 1/8
        self.up1 = UpBlock(d2, d1, d1)      # 1/8  -> 1/4
        self.up0 = UpBlock(d1, s2, s2)      # 1/4  -> 1/2
        self.upf = UpBlock(s2, s1, s1)      # 1/2  -> 1/1
        self.head = nn.Conv2d(s1, 1, 1)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in (d3, d2, d1))

    def forward(self, image: torch.Tensor, feats: List[torch.Tensor],
                ctx16: Optional[torch.Tensor] = None):
        """``ctx16``: optional extra stride-16 context (tile memory), added to the adapted f3."""
        a1, a2, a3, a4 = (ad(f) for ad, f in zip(self.adapters, feats))
        if ctx16 is not None:
            a3 = a3 + ctx16
        x1 = self.stem1(image)
        x2 = self.stem2(x1)
        u3 = self.up3(a4, a3)
        u2 = self.up2(u3, a2)
        u1 = self.up1(u2, a1)
        u0 = self.up0(u1, x2)
        uf = self.upf(u0, x1)
        logits = self.head(uf)
        if self.deep_supervision and self.training:
            size = image.shape[-2:]
            aux = [F.interpolate(h(u), size=size, mode="bilinear", align_corners=False)
                   for h, u in zip(self.aux_heads, (u3, u2, u1))]
            return logits, aux
        return logits


# --------------------------------------------------------------------------- #
#                                 Full model                                  #
# --------------------------------------------------------------------------- #
class SAM2UNet(nn.Module):
    def __init__(self, variant: str = "large", checkpoint_path: Optional[str] = None,
                 lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 grad_checkpointing: bool = False, deep_supervision: bool = True,
                 trunk: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = SAM2HieraEncoder(variant, checkpoint_path, lora_r, lora_alpha, lora_dropout,
                                        grad_checkpointing=grad_checkpointing, trunk=trunk)
        self.decoder = UNetDecoder(self.encoder.channels, deep_supervision=deep_supervision)

    def forward(self, x: torch.Tensor):
        """x must be ImageNet-normalised with H, W divisible by 32."""
        return self.decoder(x, self.encoder(x))

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen trunk: keep any norm/dropout layers in eval mode.  (Hiera uses
        # LayerNorm, so this is mostly a safeguard.)  LoRA modules stay trainable.
        self.encoder.trunk.eval()
        for m in self.encoder.trunk.modules():
            if hasattr(m, "lora_A"):
                m.train(mode)
        return self

    # ---- PEFT bookkeeping ------------------------------------------------- #
    def lora_parameters(self):
        return [p for n, p in self.encoder.named_parameters() if "lora_" in n]

    def decoder_parameters(self):
        return list(self.decoder.parameters())

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only LoRA + decoder weights (a few MB instead of ~900 MB)."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items()
                if k in names or k.startswith("decoder.")}  # includes BN buffers

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
        bad = [k for k in missing if "lora_" in k or k.startswith("decoder.")]
        assert not bad, f"Missing trainable keys: {bad[:5]}"

    def parameter_summary(self) -> str:
        tr, tot = count_parameters(self)
        lora = sum(p.numel() for p in self.lora_parameters())
        dec = sum(p.numel() for p in self.decoder_parameters())
        return (f"params total={tot / 1e6:.2f}M  trainable={tr / 1e6:.2f}M ({100 * tr / tot:.2f}%)  "
                f"[LoRA={lora / 1e6:.2f}M in {len(self.encoder.lora_modules)} layers, "
                f"decoder={dec / 1e6:.2f}M]")
