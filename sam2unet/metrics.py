"""
Step 4a -- Evaluation metrics: IoU, Dice, HD95 (+ HD, ASSD) for binary masks.

All metrics are computed per image on numpy boolean arrays restricted to the
valid FOV, then aggregated (mean / std) over a dataset.

Hausdorff distances
-------------------
Surfaces S_P, S_G = boundary pixels of the prediction / ground truth
(mask XOR its binary erosion).  For every boundary pixel we need the distance to
the nearest boundary pixel of the other mask; this is obtained exactly and in
O(N) via the Euclidean distance transform of the complement of the other
surface (``scipy.ndimage.distance_transform_edt``):

    d_{P->G} = EDT(~S_G)[S_P],      d_{G->P} = EDT(~S_P)[S_G]
    HD95 = max( P95(d_{P->G}), P95(d_{G->P}) )      (robust to outliers)
    HD   = max( max d_{P->G}, max d_{G->P} )
    ASSD = mean( concat(d_{P->G}, d_{G->P}) )

The full HD is cross-checked against ``scipy.spatial.distance.directed_hausdorff``
in the unit tests.  Pixel spacing is 1 (distances are in pixels).

Empty-mask convention: if exactly one of P, G is empty the surface distance is
undefined; we assign the image diagonal (worst case) so failures are penalised
rather than silently dropped.  If both are empty all distances are 0.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff

_STRUCT = generate_binary_structure(2, 1)  # 4-connectivity -> 1-px thick contours


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * inter / s)


def surface(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (pixels with a 4-neighbour outside)."""
    if not mask.any():
        return mask.copy()
    return np.logical_xor(mask, binary_erosion(mask, structure=_STRUCT, border_value=0))


def surface_distances(pred: np.ndarray, gt: np.ndarray):
    """(d_pred->gt, d_gt->pred) arrays of boundary-to-boundary distances (pixels)."""
    sp, sg = surface(pred), surface(gt)
    d_to_g = distance_transform_edt(~sg)
    d_to_p = distance_transform_edt(~sp)
    return d_to_g[sp], d_to_p[sg]


def hausdorff_metrics(pred: np.ndarray, gt: np.ndarray, percentile: float = 95.0) -> Dict[str, float]:
    if not pred.any() and not gt.any():
        return {"hd95": 0.0, "hd": 0.0, "assd": 0.0}
    if not pred.any() or not gt.any():
        diag = float(np.hypot(*pred.shape))
        return {"hd95": diag, "hd": diag, "assd": diag}
    d_pg, d_gp = surface_distances(pred, gt)
    return {
        "hd95": float(max(np.percentile(d_pg, percentile), np.percentile(d_gp, percentile))),
        "hd": float(max(d_pg.max(), d_gp.max())),
        "assd": float(np.concatenate([d_pg, d_gp]).mean()),
    }


def hausdorff_scipy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Reference symmetric HD via scipy's directed_hausdorff (used for validation)."""
    a, b = np.argwhere(surface(pred)), np.argwhere(surface(gt))
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, fov: Optional[np.ndarray] = None) -> Dict[str, float]:
    """All metrics for one image.  pred/gt/fov: HxW arrays (bool or {0,1})."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if fov is not None:
        fov = fov.astype(bool)
        pred, gt = pred & fov, gt & fov
    m = {"iou": iou_score(pred, gt), "dice": dice_score(pred, gt)}
    m.update(hausdorff_metrics(pred, gt))
    return m


def aggregate(per_image: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows: List[Dict[str, float]] = list(per_image)
    out: Dict[str, float] = {"n": len(rows)}
    if not rows:
        return out
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(v.mean())
        out[f"{k}_std"] = float(v.std())
    return out
