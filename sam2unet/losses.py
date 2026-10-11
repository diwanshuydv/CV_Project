"""
Step 3 -- Boundary-aware loss built on the ground-truth distance transform.

    L_total = L_standard + lambda(t) * L_boundary
    L_standard = BCE + Dice

Notation (per pixel x, inside the valid FOV only):
    p(x)    predicted foreground probability  sigmoid(logit)
    g(x)    ground truth in {0, 1}
    phi(x)  signed distance to the GT boundary from the dataloader,
            normalised to [-1, 1]: < 0 inside the structure, > 0 outside,
            |phi| = distance to the nearest boundary / max_dist.

L_boundary is a weighted sum of three complementary terms:

1. Near-edge error term (FP/FN weighted by boundary proximity)
        w(x)   = 1 + alpha * exp(-|phi(x)| / tau)
        L_err  = sum_x w(x) * [ p(1-g)  +  (1-p) g ] / sum_x w(x)
                              ^ soft FP    ^ soft FN
   Errors on/near the structure contour get up to (1+alpha)x more weight.  For
   thin vessels and roads almost every pixel is "near an edge", so this term
   effectively counters the background-dominated class imbalance.

2. Signed-distance region term (Kervadec et al., MIDL 2019)
        L_sdf  = mean_x phi(x) * p(x)
   A false positive at distance d outside the structure costs +d, a missed
   foreground pixel at depth d inside forgoes a reward of -d; its minimiser is
   exactly g.  It is a differentiable surrogate of the boundary distance
   between the predicted and GT contours, i.e. it targets HD-type errors.
   (Offset so that the perfect prediction has value 0, see code.)

3. Edge-alignment term using spatial gradients of the prediction
        E_p = || Sobel(p) ||,   E_g = || Sobel(g) ||   (gradient magnitude)
        L_edge = sum E_p |phi| / sum E_p            (predicted edges far from true contour)
               + 1 - sum min(E_p, E_g) / sum E_g    (true contour not reproduced)
   The first part is a soft "average surface distance" from the predicted
   contour to the GT contour; the second penalises broken/missing edges
   (discontinuities in thin structures).

lambda(t) is ramped up linearly from lambda_0 to lambda_max during the first
part of training: boundary terms are only informative once the network
produces a roughly correct region (Kervadec et al. use the same rebalancing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Per-image mean of x over valid pixels m, then mean over the batch."""
    num = (x * m).flatten(1).sum(1)
    den = m.flatten(1).sum(1).clamp_min(1.0)
    return (num / den).mean()


# --------------------------------------------------------------------------- #
#                              Standard losses                                #
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    """Soft Dice loss, computed per image (then averaged) inside the FOV."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target, fov):
        p, g = (prob * fov).flatten(1), (target * fov).flatten(1)
        inter = (p * g).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + g.sum(1) + self.smooth)
        return (1 - dice).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.bce_w, self.dice_w = bce_weight, dice_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, logits, target, fov):
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
        bce = _masked_mean(bce, fov)
        dice = self.dice(torch.sigmoid(logits), target, fov)
        return self.bce_w * bce + self.dice_w * dice, {"bce": bce.detach(), "dice": dice.detach()}


# --------------------------------------------------------------------------- #
#                               Boundary loss                                 #
# --------------------------------------------------------------------------- #
class SobelEdges(nn.Module):
    """Differentiable gradient-magnitude edge map of a 1-channel map."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        self.register_buffer("kernel", torch.stack([kx, kx.t()])[:, None])  # (2,1,3,3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.kernel.to(x.dtype))
        # max edge response of a 0->1 step is 1 after the /4 normalisation
        return torch.sqrt(g.pow(2).sum(1, keepdim=True) + EPS).clamp(max=1.0)


class BoundaryLoss(nn.Module):
    def __init__(self, w_err: float = 1.0, w_sdf: float = 1.0, w_edge: float = 1.0,
                 alpha: float = 4.0, tau: float = 0.1):
        """
        alpha, tau : near-edge weight w = 1 + alpha * exp(-|phi| / tau).  With the
                     default max_dist=20 px, tau=0.1 corresponds to a 2-pixel decay.
        """
        super().__init__()
        self.w_err, self.w_sdf, self.w_edge = w_err, w_sdf, w_edge
        self.alpha, self.tau = alpha, tau
        self.sobel = SobelEdges()

    def forward(self, prob, target, dist, fov):
        terms: Dict[str, torch.Tensor] = {}
        adist = dist.abs()

        # (1) near-edge weighted soft FP + FN
        w = (1 + self.alpha * torch.exp(-adist / self.tau)) * fov
        err = prob * (1 - target) + (1 - prob) * target
        terms["b_err"] = ((w * err).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(EPS)).mean()

        # (2) signed-distance region term, shifted by its optimum so perfect pred -> 0
        sdf = _masked_mean(dist * prob, fov) - _masked_mean(dist * target, fov)
        terms["b_sdf"] = sdf

        # (3) edge alignment using spatial gradients of the prediction
        e_p = self.sobel(prob) * fov
        e_g = self.sobel(target) * fov
        disp = (e_p * adist).flatten(1).sum(1) / e_p.flatten(1).sum(1).clamp_min(EPS)
        recall = torch.minimum(e_p, e_g).flatten(1).sum(1) / e_g.flatten(1).sum(1).clamp_min(EPS)
        has_edge = (e_g.flatten(1).sum(1) > EPS).float()       # ignore crops without GT contour
        terms["b_edge"] = (disp + has_edge * (1 - recall)).mean()

        total = self.w_err * terms["b_err"] + self.w_sdf * terms["b_sdf"] + self.w_edge * terms["b_edge"]
        return total, {k: v.detach() for k, v in terms.items()}


# --------------------------------------------------------------------------- #
#                                Total loss                                   #
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    lambda_0: float = 0.01
    lambda_max: float = 0.5          # 0 -> plain BCE+Dice baseline (ablation)
    ramp_fraction: float = 0.3       # fraction of training used to ramp lambda
    w_err: float = 1.0
    w_sdf: float = 1.0
    w_edge: float = 1.0
    alpha: float = 4.0
    tau: float = 0.1
    aux_weight: float = 0.4          # deep-supervision weight


class BoundaryAwareLoss(nn.Module):
    """L_total = (BCE + Dice) + lambda(t) * L_boundary, with deep supervision."""

    def __init__(self, cfg: LossConfig = LossConfig()):
        super().__init__()
        self.cfg = cfg
        self.standard = BCEDiceLoss()
        self.boundary = BoundaryLoss(cfg.w_err, cfg.w_sdf, cfg.w_edge, cfg.alpha, cfg.tau)
        self.lam = cfg.lambda_0 if cfg.lambda_max > 0 else 0.0

    def set_progress(self, progress: float) -> None:
        """progress in [0, 1] = fraction of total training iterations done."""
        c = self.cfg
        if c.lambda_max <= 0:
            self.lam = 0.0
            return
        r = min(1.0, progress / max(c.ramp_fraction, EPS))
        self.lam = c.lambda_0 + r * (c.lambda_max - c.lambda_0)

    def _single(self, logits, target, dist, fov):
        logits = logits.float()  # loss in fp32 under autocast
        std, logs = self.standard(logits, target, fov)
        if self.lam > 0:
            bnd, blogs = self.boundary(torch.sigmoid(logits), target, dist, fov)
            logs.update(blogs)
        else:
            bnd = logits.new_zeros(())
        logs["boundary"] = bnd.detach()
        return std + self.lam * bnd, logs

    def forward(self, outputs: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
                target, dist, fov):
        if isinstance(outputs, tuple):
            logits, aux = outputs
        else:
            logits, aux = outputs, []
        loss, logs = self._single(logits, target, dist, fov)
        for a in aux:  # deep supervision on intermediate decoder stages
            loss = loss + self.cfg.aux_weight * self._single(a, target, dist, fov)[0] / max(len(aux), 1)
        logs["lambda"] = torch.tensor(self.lam)
        logs["total"] = loss.detach()
        return loss, logs
