"""
Tile-sequence memory (TSM): SAM 2's video memory re-used as a *spatial* memory.

Thin structures fail at the level of connectivity: a vessel or road that leaves
a 512 px tile is predicted without knowing where it goes next.  SAM 2 already
owns a mechanism that conditions one frame on others -- memory attention over
encoded (features, mask) memories -- but it is only used for video.  Here the
"frames" are views of one image region:

    frame 0   : the whole region, downsampled to one tile (global context)
    frames 1-K: the full-resolution tiles of the region (2 x 2 grid in training)

Two passes, both batched over frames (no sequential dependency):

  pass 1   trunk -> U-Net decoder for every frame (exactly the SAM2-UNet baseline);
           SAM 2's MemoryEncoder turns each frame's stride-16 FPN features plus its
           pass-1 mask into a 64-d memory.
  pass 2   each tile's stride-16 features cross-attend to the memories of *all*
           frames (pretrained SAM 2 MemoryAttention + LoRA).  The output enters the
           decoder at stride 16 through a zero-initialised 1x1 conv, and the
           decoder predicts the final mask.

Position: SAM 2 tells memories apart with a *temporal* encoding (frames ago).  We
keep its per-frame spatial sine encoding and its temporal slot for "conditioning
frame" (given to the global view; tiles get the 1-frame-ago slot), and add an
encoding of each token's *absolute coordinate in the region*, to both the queries
and the memory keys.  Both projections are zero-initialised, so at step 0 the
model is the pretrained SAM 2 memory plus a no-op context branch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import inject_lora
from .model import SAM2_VARIANTS, SAM2UNet

MEMORY_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2")


def fourier_coords(y: torch.Tensor, x: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    """Fourier features of normalised coordinates in [0, 1] -> (..., 4 * n_freq)."""
    freqs = (2.0 ** torch.arange(n_freq, device=y.device, dtype=torch.float32)) * math.pi
    ang = torch.stack([y, x], -1)[..., None] * freqs              # (..., 2, F)
    return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)      # (..., 4F)


def tile_grid(region: int, window: int) -> List[Tuple[int, int]]:
    """Top-left corners of the non-overlapping tiles covering a region (row-major)."""
    n = region // window
    return [(i * window, j * window) for i in range(n) for j in range(n)]


class TileMemory(nn.Module):
    """SAM 2 neck + memory encoder + memory attention, adapted to spatial memories."""

    def __init__(self, neck: nn.Module, memory_encoder: nn.Module, memory_attention: nn.Module,
                 maskmem_tpos_enc: torch.Tensor, no_mem_embed: torch.Tensor, dec_channels16: int,
                 lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 n_freq: int = 16, sigmoid_scale: float = 20.0, sigmoid_bias: float = -10.0):
        super().__init__()
        self.neck = neck.requires_grad_(False)                       # frozen pretrained FPN
        self.memory_encoder = memory_encoder                        # fully trainable (~1 M)
        self.memory_attention = memory_attention.requires_grad_(False)
        self.lora_modules = inject_lora(self.memory_attention, MEMORY_LORA_TARGETS, r=lora_r,
                                        alpha=lora_alpha, dropout=lora_dropout)
        self.maskmem_tpos_enc = nn.Parameter(maskmem_tpos_enc.detach().clone())  # (T, 1, 1, mem_dim)
        self.no_mem_embed = nn.Parameter(no_mem_embed.detach().clone())          # (1, 1, d_model)
        self.d_model = memory_attention.d_model
        self.mem_dim = maskmem_tpos_enc.shape[-1]
        self.sigmoid_scale, self.sigmoid_bias = sigmoid_scale, sigmoid_bias
        self.n_freq = n_freq
        # absolute-coordinate encodings (zero-init: start from the pretrained behaviour)
        self.coord_q = nn.Linear(4 * n_freq, self.d_model)
        self.coord_k = nn.Linear(4 * n_freq, self.mem_dim)
        for lin in (self.coord_q, self.coord_k):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        # memory-conditioned stride-16 features -> decoder (zero-init residual)
        self.to_decoder = nn.Conv2d(self.d_model, dec_channels16, 1)
        nn.init.zeros_(self.to_decoder.weight); nn.init.zeros_(self.to_decoder.bias)

    # ----------------------------------------------------------------------- #
    def features16(self, trunk_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stride-16 FPN features (B, 256, h, w) and their sine position encoding."""
        out, pos = self.neck(trunk_feats)
        return out[2], pos[2]

    def coords(self, boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Fourier features of token centres.  boxes: (N, 4) = (y0, x0, size_y, size_x) of each
        frame in normalised region coordinates -> (N, h*w, 4F)."""
        ty = (torch.arange(h, device=boxes.device, dtype=torch.float32) + 0.5) / h
        tx = (torch.arange(w, device=boxes.device, dtype=torch.float32) + 0.5) / w
        gy, gx = torch.meshgrid(ty, tx, indexing="ij")
        y = boxes[:, 0, None] + boxes[:, 2, None] * gy.flatten()[None]
        x = boxes[:, 1, None] + boxes[:, 3, None] * gx.flatten()[None]
        return fourier_coords(y, x, self.n_freq)

    def encode_memory(self, f16: torch.Tensor, mask_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAM 2 memory of one batch of frames: (B, 64, h, w) features + sine position encoding.
        ``mask_logits`` at frame resolution (16 x h); scaled like SAM 2 (sigmoid * 20 - 10)."""
        m = torch.sigmoid(mask_logits.float()) * self.sigmoid_scale + self.sigmoid_bias
        out = self.memory_encoder(f16, m.to(f16.dtype), skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][-1]

    def attend(self, q16: torch.Tensor, q_pos: torch.Tensor, q_coords: torch.Tensor,
               mem: torch.Tensor, mem_pos: torch.Tensor) -> torch.Tensor:
        """q16/q_pos: (B, 256, h, w); q_coords: (B, hw, 4F); mem/mem_pos: (B, M*hw, mem_dim)
        with coordinate + temporal encodings already added to mem_pos.  -> (B, 256, h, w)."""
        B, C, h, w = q16.shape
        curr = q16.flatten(2).permute(2, 0, 1)                                   # (hw, B, C)
        curr_pos = q_pos.flatten(2).permute(2, 0, 1) + self.coord_q(q_coords).permute(1, 0, 2).to(q16.dtype)
        out = self.memory_attention(curr=curr, curr_pos=curr_pos,
                                    memory=mem.permute(1, 0, 2), memory_pos=mem_pos.permute(1, 0, 2))
        return out.permute(1, 2, 0).reshape(B, C, h, w)

    def context(self, f16_with_mem: torch.Tensor) -> torch.Tensor:
        return self.to_decoder(f16_with_mem)


class SAM2MemUNet(SAM2UNet):
    """SAM2-UNet + tile-sequence memory.  ``forward`` on a single tile behaves like the
    baseline (no memory); ``forward_region`` / ``predict_full`` use the memory."""

    def __init__(self, variant: str = "large", checkpoint_path: Optional[str] = None,
                 lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 grad_checkpointing: bool = False, deep_supervision: bool = True,
                 window: int = 512, region: int = 1024, use_global: bool = True,
                 use_tiles: bool = True, mem_lora_r: int = 8, sam: Optional[nn.Module] = None,
                 trunk: Optional[nn.Module] = None, hf_prior: bool = False):
        if sam is None and trunk is None:
            from sam2.build_sam import build_sam2
            sam = build_sam2(SAM2_VARIANTS[variant][0], checkpoint_path, device="cpu", mode="eval")
        super().__init__(variant, checkpoint_path, lora_r, lora_alpha, lora_dropout,
                         grad_checkpointing, deep_supervision,
                         trunk=sam.image_encoder.trunk if sam is not None else trunk, hf_prior=hf_prior)
        assert region % window == 0 and window % 32 == 0
        self.window, self.region = window, region
        self.use_global, self.use_tiles = use_global, use_tiles
        self.memory = TileMemory(sam.image_encoder.neck, sam.memory_encoder, sam.memory_attention,
                                 sam.maskmem_tpos_enc, sam.no_mem_embed,
                                 dec_channels16=self.decoder.adapters[2][0].out_channels,
                                 lora_r=mem_lora_r, lora_alpha=2.0 * mem_lora_r, lora_dropout=lora_dropout)

    # ---- parameter groups -------------------------------------------------- #
    def lora_parameters(self):
        """Pretrained weights adapted at the low LR: trunk + memory-attention LoRA, memory
        encoder and SAM 2's memory embeddings."""
        mem = self.memory
        return (super().lora_parameters()
                + [p for n, p in mem.memory_attention.named_parameters() if "lora_" in n]
                + list(mem.memory_encoder.parameters()) + [mem.maskmem_tpos_enc, mem.no_mem_embed])

    def decoder_parameters(self):
        """New weights at the high LR: decoder, coordinate encodings, context projection."""
        mem = self.memory
        return (super().decoder_parameters() + list(mem.coord_q.parameters())
                + list(mem.coord_k.parameters()) + list(mem.to_decoder.parameters()))

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items()
                if k in names or k.startswith(("decoder.", "memory.memory_encoder."))}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
        bad = [k for k in missing if "lora_" in k or k.startswith(("decoder.", "memory.memory_encoder."))]
        assert not bad, f"Missing trainable keys: {bad[:5]}"

    def train(self, mode: bool = True):
        super().train(mode)
        self.memory.neck.eval()
        return self

    # ---- region forward ---------------------------------------------------- #
    def forward_region(self, region: torch.Tensor) -> Dict[str, object]:
        """region: (B, 3, R, R), R = self.region.  Returns
        ``frames``: list of (outputs, box) for pass 1 (global first, then tiles) and
        ``final`` : list of (outputs, box) for the memory-conditioned tiles;
        box = (y0, x0, size) in region pixels, the global frame has box (0, 0, R)."""
        B, _, R, _ = region.shape
        win = self.window
        grid = tile_grid(R, win)
        views, boxes = [], []
        if self.use_global:
            views.append(F.interpolate(region, size=(win, win), mode="bilinear",
                                       align_corners=False, antialias=True))
            boxes.append((0, 0, R))
        for y, x in grid:
            views.append(region[..., y:y + win, x:x + win])
            boxes.append((y, x, win))
        nF = len(views)
        frames = torch.cat(views)                                     # (nF*B, 3, win, win), frame-major
        feats = self.encoder(frames)

        # pass 1: plain decoder on every frame
        out1 = self.decoder(frames, feats)
        logits1 = out1[0] if isinstance(out1, tuple) else out1

        # memories of all frames (inputs detached: memory gradients reach only the memory modules)
        f16, pos16 = self.memory.features16(feats)
        mem, mpos = self.memory.encode_memory(f16.detach(), logits1.detach())
        h, w = f16.shape[-2:]
        bx = torch.tensor(boxes, dtype=torch.float32, device=region.device) / R
        bx = torch.cat([bx[:, :2], bx[:, 2:3], bx[:, 2:3]], 1)        # (nF, 4)
        coords = self.memory.coords(bx, h, w)                         # (nF, hw, 4F)
        tpos = self.memory.maskmem_tpos_enc
        T = tpos.shape[0]
        mem_tok, mem_pos = [], []
        for k in range(nF):
            sl = slice(k * B, (k + 1) * B)
            t = tpos[T - 1] if (self.use_global and k == 0) else tpos[T - 2]   # cond slot / 1 frame ago
            mem_tok.append(mem[sl].flatten(2).transpose(1, 2))       # (B, hw, 64)
            mem_pos.append(mpos[sl].flatten(2).transpose(1, 2) + t.view(1, 1, -1)
                           + self.memory.coord_k(coords[k]).to(mpos.dtype)[None])
        if not self.use_tiles:                                        # ablation: global memory only
            mem_tok, mem_pos = mem_tok[:1], mem_pos[:1]
        mem_tok, mem_pos = torch.cat(mem_tok, 1), torch.cat(mem_pos, 1)   # (B, M*hw, 64)

        # pass 2: tiles attend to the memory bank of the region
        t0 = 1 if self.use_global else 0
        nT = nF - t0
        q16 = f16[t0 * B:]
        q_pos = pos16[t0 * B:]
        q_coords = coords[t0:].repeat_interleave(B, 0)                # frame-major like q16
        mem_rep = mem_tok.repeat(nT, 1, 1)
        mpos_rep = mem_pos.repeat(nT, 1, 1)
        f16_mem = self.memory.attend(q16, q_pos, q_coords, mem_rep, mpos_rep)
        ctx = self.memory.context(f16_mem)
        tile_frames = frames[t0 * B:]
        tile_feats = [f[t0 * B:] for f in feats]
        out2 = self.decoder(tile_frames, tile_feats, ctx16=ctx)

        def split(out, n0, n):
            if isinstance(out, tuple):
                main, aux = out
                return [((main[k * B:(k + 1) * B], [a[k * B:(k + 1) * B] for a in aux]))
                        for k in range(n0, n)]
            return [out[k * B:(k + 1) * B] for k in range(n0, n)]

        return {"frames": list(zip(split(out1, 0, nF), boxes)),
                "final": list(zip(split(out2, 0, nT), boxes[t0:]))}

    # ---- whole-image inference --------------------------------------------- #
    @torch.no_grad()
    def predict_full(self, image: torch.Tensor, window: int = None, overlap: float = 0.5) -> torch.Tensor:
        """(1, 3, H, W) -> (1, 1, H, W) probabilities.

        Regions of ``self.region`` px are placed every half tile, and every tile prediction is
        blended with a Gaussian weight (as in the sliding-window baseline).  The image is padded
        to at least region + window/2 so that even a single-region image (DRIVE) is seen at
        two tile phases per axis -- otherwise the 2 x 2 tile seams would never be covered by
        a tile centre.  (``window`` / ``overlap`` are fixed by the model and ignored.)
        """
        from .engine import _gaussian_window
        from .utils import amp_dtype

        self.eval()
        R, win = self.region, self.window
        device = image.device
        dtype = amp_dtype(device)
        _, _, H, W = image.shape
        Hp = max(R + win // 2, math.ceil(H / 32) * 32)
        Wp = max(R + win // 2, math.ceil(W / 32) * 32)
        mode = "reflect" if (Hp - H) < H and (Wp - W) < W else "constant"
        x = F.pad(image, (0, Wp - W, 0, Hp - H), mode=mode)
        stride = win // 2

        def starts(L):
            s = list(range(0, L - R, stride))
            return s + [L - R]

        wmap = _gaussian_window(win).to(device)
        prob = torch.zeros(1, 1, Hp, Wp, device=device)
        norm = torch.zeros_like(prob)
        for y in starts(Hp):
            for xx in starts(Wp):
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    res = self.forward_region(x[..., y:y + R, xx:xx + R])
                outs = res["final"]
                if getattr(self, "memory_off", False):   # analysis: same weights, pass-1 tiles
                    outs = res["frames"][1 if self.use_global else 0:]
                for out, (ty, tx, s) in outs:
                    logit = out[0] if isinstance(out, tuple) else out
                    sl = (..., slice(y + ty, y + ty + s), slice(xx + tx, xx + tx + s))
                    prob[sl] += torch.sigmoid(logit.float()) * wmap
                    norm[sl] += wmap
        return (prob / norm.clamp_min(1e-6))[..., :H, :W]
