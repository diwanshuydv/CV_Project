"""
Step 4a -- Evaluation metrics for thin-structure binary masks.

Per image (inside the valid FOV, then mean / std over the dataset):
    IoU, Dice, HD95 / HD / ASSD, clDice, Betti-0/1 errors, relaxed precision /
    recall / F1 (roads) and the fringe-vs-segment error decomposition.
Per dataset (pixel-pooled, ``PixelStats``):
    Se / Sp / Acc / F1 at a threshold, ROC-AUC (retinal-vessel convention) and
    ODS / OIS F1 (crack-detection convention).

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


# --------------------------------------------------------------------------- #
#                         Topology / connectivity metrics                     #
# --------------------------------------------------------------------------- #
def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice (Shit et al., CVPR 2021) on hard masks.

    Tprec = |S(P) & G| / |S(P)|  (predicted centrelines lying inside the GT),
    Tsens = |S(G) & P| / |S(G)|  (GT centrelines covered by the prediction),
    clDice = harmonic mean.  A missed or broken branch lowers Tsens no matter how
    thin it is, so the score tracks connectivity rather than pixel overlap.
    """
    from skimage.morphology import skeletonize

    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    return 0.0 if tprec + tsens == 0 else float(2 * tprec * tsens / (tprec + tsens))


def betti_numbers(mask: np.ndarray) -> tuple:
    """(beta0, beta1) of a 2D binary mask: 8-connected components and enclosed holes.

    beta1 follows from the Euler characteristic chi = beta0 - beta1 (8-connectivity
    for the foreground, i.e. 4-connectivity for the background holes).
    """
    from scipy.ndimage import label
    from skimage.measure import euler_number

    if not mask.any():
        return 0, 0
    b0 = label(mask, structure=np.ones((3, 3), bool))[1]
    return b0, b0 - int(euler_number(mask, connectivity=2))


def betti_errors(pred: np.ndarray, gt: np.ndarray, patch: int = 64) -> Dict[str, float]:
    """Betti-number errors, whole image and averaged over a non-overlapping patch grid.

    Patch-level errors (as in clDice / TopoNet evaluations) localise topology
    mistakes: a missed branch in one corner cannot cancel a spurious one elsewhere.
    Only patches that contain foreground in P or G are averaged.
    """
    p0, p1 = betti_numbers(pred)
    g0, g1 = betti_numbers(gt)
    e0, e1 = [], []
    H, W = gt.shape
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            pp, gg = pred[y:y + patch, x:x + patch], gt[y:y + patch, x:x + patch]
            if not (pp.any() or gg.any()):
                continue
            (a0, a1), (b0, b1) = betti_numbers(pp), betti_numbers(gg)
            e0.append(abs(a0 - b0))
            e1.append(abs(a1 - b1))
    return {"betti0_err": float(abs(p0 - g0)), "betti1_err": float(abs(p1 - g1)),
            "betti0_err_patch": float(np.mean(e0)) if e0 else 0.0,
            "betti1_err_patch": float(np.mean(e1)) if e1 else 0.0}


def relaxed_metrics(pred: np.ndarray, gt: np.ndarray, rho: int = 3) -> Dict[str, float]:
    """Relaxed precision / recall / F1 (Mnih & Hinton 2010; standard for road extraction).

    A predicted pixel is correct if a GT pixel lies within ``rho`` pixels and vice versa,
    which forgives the ~1-3 px annotation offset of road centre-line labels.
    """
    if not pred.any() or not gt.any():
        v = 1.0 if not pred.any() and not gt.any() else 0.0
        return {"rel_prec": v, "rel_rec": v, "rel_f1": v}
    prec = float((distance_transform_edt(~gt)[pred] <= rho).mean())
    rec = float((distance_transform_edt(~pred)[gt] <= rho).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"rel_prec": prec, "rel_rec": rec, "rel_f1": f1}


def error_decomposition(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> Dict[str, float]:
    """Split pixel errors into *fringe* errors and *segment* errors.

    fringe : an FN pixel within ``tol`` px of the prediction, or an FP pixel within
             ``tol`` px of the GT -- the structure was found but its width/boundary is off.
    segment: errors farther away -- a whole branch missed (FN) or a spurious
             fragment (FP).  Boundary-precision losses can only fix the first kind.
    Returns the fraction of all error pixels that are segment-level, per error type.
    """
    fn, fp = gt & ~pred, pred & ~gt
    n_err = fn.sum() + fp.sum()
    if n_err == 0:
        return {"seg_err_frac": 0.0, "seg_fn_frac": 0.0, "seg_fp_frac": 0.0}
    d_to_pred = distance_transform_edt(~pred) if pred.any() else np.full(gt.shape, np.inf)
    d_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    seg_fn = (fn & (d_to_pred > tol)).sum()
    seg_fp = (fp & (d_to_gt > tol)).sum()
    return {"seg_err_frac": float((seg_fn + seg_fp) / n_err),
            "seg_fn_frac": float(seg_fn / max(fn.sum(), 1)),
            "seg_fp_frac": float(seg_fp / max(fp.sum(), 1))}


def compute_metrics(pred: np.ndarray, gt: np.ndarray, fov: Optional[np.ndarray] = None,
                    topology: bool = True, rho: int = 3) -> Dict[str, float]:
    """All per-image metrics.  pred/gt/fov: HxW arrays (bool or {0,1})."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if fov is not None:
        fov = fov.astype(bool)
        pred, gt = pred & fov, gt & fov
    m = {"iou": iou_score(pred, gt), "dice": dice_score(pred, gt)}
    m.update(hausdorff_metrics(pred, gt))
    if topology:
        m["cldice"] = cldice_score(pred, gt)
        m.update(betti_errors(pred, gt))
        m.update(relaxed_metrics(pred, gt, rho))
        m.update(error_decomposition(pred, gt))
    return m


# --------------------------------------------------------------------------- #
#                 Dataset-level (pixel-pooled) threshold metrics              #
# --------------------------------------------------------------------------- #
class PixelStats:
    """Accumulates per-image histograms of foreground probabilities inside the FOV.

    With ``bins`` equally spaced thresholds this gives, without storing any
    probability maps (HRF test = 30 x 8 Mpx):
      * pixel-pooled Se / Sp / Acc / F1 at a fixed threshold (retinal-vessel convention),
      * ROC-AUC (trapezoidal over the histogram thresholds),
      * ODS (best F1 at one dataset-wide threshold) and OIS (mean per-image best F1),
        the crack-detection convention,
      * the threshold maximising pooled F1 -- picked on the validation set.
    """

    def __init__(self, bins: int = 2000, logit_range: float = 16.0):
        # Bins are uniform in *logit* space: Dice-trained networks saturate, so uniform bins in
        # probability space lump most background pixels and many missed vessels into the
        # lowest bin and under-estimate the AUC (DRIVE: 0.967 vs 0.98+ with logit bins).
        self.bins = bins
        z = np.linspace(-logit_range, logit_range, bins + 1)[1:-1]
        self.edges = 1.0 / (1.0 + np.exp(-z))          # bin k = [edges[k-1], edges[k])
        self.pos: List[np.ndarray] = []
        self.neg: List[np.ndarray] = []

    def threshold_of(self, k: int) -> float:
        """Probability threshold at which 'predict fg if bin >= k' starts."""
        return 0.0 if k == 0 else float(self.edges[k - 1])

    def update(self, prob: np.ndarray, gt: np.ndarray, fov: Optional[np.ndarray] = None) -> None:
        gt = gt.astype(bool)
        valid = np.ones_like(gt) if fov is None else fov.astype(bool)
        idx = np.searchsorted(self.edges, prob, side="right")
        self.pos.append(np.bincount(idx[valid & gt], minlength=self.bins))
        self.neg.append(np.bincount(idx[valid & ~gt], minlength=self.bins))

    @staticmethod
    def _counts(pos: np.ndarray, neg: np.ndarray):
        """TP/FP/FN/TN for every threshold t_k = k / bins (predict fg if bin >= k)."""
        tp = np.cumsum(pos[..., ::-1], -1)[..., ::-1]
        fp = np.cumsum(neg[..., ::-1], -1)[..., ::-1]
        P, N = pos.sum(-1, keepdims=True), neg.sum(-1, keepdims=True)
        return tp, fp, P - tp, N - fp

    @staticmethod
    def _f1(tp, fp, fn):
        return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0)

    def summary(self, threshold: float = 0.5) -> Dict[str, float]:
        if not self.pos:
            return {}
        pos_i, neg_i = np.stack(self.pos).astype(np.float64), np.stack(self.neg).astype(np.float64)
        pos, neg = pos_i.sum(0), neg_i.sum(0)
        tp, fp, fn, tn = self._counts(pos, neg)
        k = int(np.searchsorted(self.edges, threshold, side="left")) + 1   # bins with p >= ~threshold
        k = min(k, self.bins - 1)
        f1 = self._f1(tp, fp, fn)
        tpr, fpr = tp / max(pos.sum(), 1), fp / max(neg.sum(), 1)
        auc = float(np.trapezoid(np.r_[tpr, 0.0][::-1], np.r_[fpr, 0.0][::-1]))
        best = int(f1.argmax())
        f1_i = self._f1(*self._counts(pos_i, neg_i)[:3])
        return {
            "px_f1": float(f1[k]),
            "px_se": float(tp[k] / max(tp[k] + fn[k], 1)),
            "px_sp": float(tn[k] / max(tn[k] + fp[k], 1)),
            "px_acc": float((tp[k] + tn[k]) / max(pos.sum() + neg.sum(), 1)),
            "px_auc": auc,
            "ods_f1": float(f1[best]),
            "ods_thr": self.threshold_of(best),
            "ois_f1": float(f1_i.max(-1).mean()),
        }


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
