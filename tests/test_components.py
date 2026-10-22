"""CPU unit tests for every component (no SAM 2 checkpoint needed).

Run:  python -m pytest -q tests/
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from sam2unet.datasets import DataConfig, build_dataset, signed_distance_map
from sam2unet.losses import BoundaryAwareLoss, BoundaryLoss, LossConfig, SobelEdges
from sam2unet.lora import LoRALinear, inject_lora
from sam2unet.metrics import (PixelStats, betti_errors, betti_numbers, cldice_score, compute_metrics,
                              dice_score, error_decomposition, hausdorff_metrics, hausdorff_scipy,
                              relaxed_metrics)
from sam2unet.engine import sliding_window_predict


def square(n=64, lo=20, hi=40, shift=0):
    m = np.zeros((n, n), np.uint8)
    m[lo:hi, lo + shift:hi + shift] = 1
    return m


# ------------------------------- EDT -------------------------------------- #
def test_signed_distance_signs_and_zero_level_set():
    m = square()
    phi = signed_distance_map(m, max_dist=20) * 20
    assert (phi[m == 1] <= 0).all() and (phi[m == 0] >= 1).all()
    assert phi[20, 30] == 0            # outermost ring of the square -> 0
    assert phi[19, 30] == 1            # one pixel outside -> 1
    assert phi[30, 30] < -5            # deep inside
    assert np.all(signed_distance_map(np.zeros((8, 8))) == 1)
    assert np.all(signed_distance_map(np.ones((8, 8))) == -1)


# ------------------------------- LoRA ------------------------------------- #
def test_lora_identity_at_init_and_trainable_params():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x))
    assert not base.weight.requires_grad
    assert {n for n, p in lora.named_parameters() if p.requires_grad} == {"lora_A.weight", "lora_B.weight"}
    lora.lora_B.weight.data.normal_()
    y = lora(x)
    lora.merge()
    assert torch.allclose(lora(x), y, atol=1e-5)


def tiny_hiera():
    from sam2.modeling.backbones.hieradet import Hiera
    return Hiera(embed_dim=16, num_heads=1, stages=(1, 1, 1, 1), global_att_blocks=(),
                 window_pos_embed_bkg_spatial_size=(4, 4), window_spec=(4, 4, 4, 4))


def test_model_with_tiny_hiera():
    pytest.importorskip("sam2")
    from sam2unet.model import SAM2UNet
    model = SAM2UNet(trunk=tiny_hiera(), lora_r=2)
    n_blocks = len(model.encoder.trunk.blocks)
    assert len(model.encoder.lora_modules) == 4 * n_blocks      # qkv, proj, mlp0, mlp1
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith("decoder.") for n in trainable)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    logits, aux = model(x)
    assert logits.shape == (2, 1, 128, 128) and len(aux) == 3
    logits.mean().backward()
    assert model.lora_parameters()[0].grad is not None
    # trainable state dict round-trip
    sd = model.trainable_state_dict()
    m2 = SAM2UNet(trunk=tiny_hiera(), lora_r=2)
    m2.encoder.trunk.load_state_dict(model.encoder.trunk.state_dict())
    m2.load_trainable_state_dict(sd)
    model.eval(); m2.eval()
    assert torch.allclose(model(x), m2(x), atol=1e-5)
    # sliding window on a non-multiple-of-32 image
    p = sliding_window_predict(model, torch.randn(1, 3, 150, 170), window=128)
    assert p.shape == (1, 1, 150, 170) and p.min() >= 0 and p.max() <= 1


def test_grad_checkpointing_matches():
    pytest.importorskip("sam2")
    from sam2unet.model import SAM2UNet
    torch.manual_seed(0)
    a = SAM2UNet(trunk=tiny_hiera(), lora_r=2, deep_supervision=False)
    b = SAM2UNet(trunk=tiny_hiera(), lora_r=2, deep_supervision=False, grad_checkpointing=True)
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        for p in m.lora_parameters():
            torch.manual_seed(1); p.data.normal_(0, 0.1)
    x = torch.randn(1, 3, 64, 64)
    a.eval(); b.eval()
    with torch.enable_grad():
        la, lb = a(x).sum(), b(x).sum()
        la.backward(); lb.backward()
    assert torch.allclose(la, lb, atol=1e-4)
    assert torch.allclose(a.lora_parameters()[0].grad, b.lora_parameters()[0].grad, atol=1e-4)


# ------------------------------- Loss ------------------------------------- #
def _tensors(m):
    g = torch.tensor(m, dtype=torch.float32)[None, None]
    d = torch.tensor(signed_distance_map(m), dtype=torch.float32)[None, None]
    return g, d, torch.ones_like(g)


def test_boundary_loss_zero_for_perfect_and_monotone_in_shift():
    g, d, fov = _tensors(square())
    bl = BoundaryLoss()
    perfect, logs = bl(g, g, d, fov)
    assert logs["b_err"] < 1e-6 and abs(logs["b_sdf"]) < 1e-6
    vals = []
    for s in (2, 5, 10):
        p = torch.tensor(square(shift=s), dtype=torch.float32)[None, None]
        vals.append(bl(p, g, d, fov)[0].item())
    assert perfect.item() < vals[0] < vals[1] < vals[2]


def test_total_loss_gradients_and_lambda_ramp():
    g, d, fov = _tensors(square())
    logits = torch.randn_like(g, requires_grad=True)
    lf = BoundaryAwareLoss(LossConfig(lambda_0=0.01, lambda_max=0.5, ramp_fraction=0.5))
    lf.set_progress(0.0); assert abs(lf.lam - 0.01) < 1e-9
    lf.set_progress(0.25); assert abs(lf.lam - 0.255) < 1e-9
    lf.set_progress(0.9); assert abs(lf.lam - 0.5) < 1e-9
    loss, logs = lf((logits, [logits * 0.5]), g, d, fov)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0
    base = BoundaryAwareLoss(LossConfig(lambda_max=0))
    base.set_progress(0.5)
    assert base.lam == 0


def test_sobel_unit_step():
    x = torch.zeros(1, 1, 8, 8); x[..., 4:] = 1
    e = SobelEdges()(x)
    assert torch.isclose(e[0, 0, 4, 3], torch.tensor(1.0), atol=1e-3)
    assert e[0, 0, 4, 0] < 1e-2


# ------------------------------ Metrics ----------------------------------- #
def test_metrics_shifted_square():
    gt, pred = square().astype(bool), square(shift=3).astype(bool)
    m = compute_metrics(pred, gt)
    assert 0 < m["iou"] < m["dice"] < 1
    assert m["hd"] == pytest.approx(3.0)
    assert m["hd95"] == pytest.approx(3.0)
    assert m["hd"] == pytest.approx(hausdorff_scipy(pred, gt))
    assert compute_metrics(gt, gt, topology=False) == {"iou": 1.0, "dice": 1.0, "hd95": 0.0, "hd": 0.0,
                                                      "assd": 0.0}
    full = compute_metrics(gt, gt)
    assert full["cldice"] == 1.0 and full["betti0_err"] == 0 and full["rel_f1"] == 1.0
    assert full["seg_err_frac"] == 0.0
    assert hausdorff_metrics(np.zeros_like(gt), gt)["hd95"] == pytest.approx(np.hypot(64, 64))


def test_random_masks_hd_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.random((40, 40)) > 0.7, rng.random((40, 40)) > 0.7
        assert hausdorff_metrics(a, b)["hd"] == pytest.approx(hausdorff_scipy(a, b))


def test_betti_numbers_and_cldice():
    ring = np.zeros((40, 40), bool); ring[5:30, 5:30] = True; ring[10:25, 10:25] = False
    two = np.zeros((40, 40), bool); two[5:10, 5:35] = True; two[20:25, 5:35] = True
    assert betti_numbers(ring) == (1, 1)
    assert betti_numbers(two) == (2, 0)
    assert betti_numbers(np.zeros((8, 8), bool)) == (0, 0)
    # a line broken in the middle: tiny pixel loss, but clDice / beta0 notice the gap
    line = np.zeros((20, 60), bool); line[9:12, 2:58] = True
    broken = line.copy(); broken[:, 29:32] = False
    assert dice_score(broken, line) > 0.94
    e = betti_errors(broken, line, patch=20)
    assert e["betti0_err"] == 1 and e["betti0_err_patch"] > 0
    assert cldice_score(line, line) == 1.0 and cldice_score(broken, line) < 1.0


def test_relaxed_and_error_decomposition():
    gt = np.zeros((40, 40), bool); gt[20, 5:35] = True
    near = np.zeros_like(gt); near[22, 5:35] = True          # 2 px offset
    assert relaxed_metrics(near, gt, rho=3)["rel_f1"] == 1.0
    assert relaxed_metrics(near, gt, rho=1)["rel_f1"] == 0.0
    assert error_decomposition(near, gt)["seg_err_frac"] == 0.0   # all fringe
    far = np.zeros_like(gt); far[2, 5:35] = True              # spurious + missed segment
    assert error_decomposition(far, gt)["seg_err_frac"] == 1.0


def test_pixel_stats_auc_and_threshold_metrics():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    gt = rng.random((2, 50, 50)) > 0.8
    prob = np.clip(gt * 0.3 + rng.random(gt.shape) * 0.7, 0, 1)
    ps = PixelStats()
    for p, g in zip(prob, gt):
        ps.update(p, g)
    s = ps.summary(0.5)
    assert s["px_auc"] == pytest.approx(roc_auc_score(gt.ravel(), prob.ravel()), abs=2e-3)
    pred = prob > 0.5
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    assert s["px_f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn), abs=1e-3)
    assert s["ods_f1"] >= s["px_f1"] - 1e-9 and s["ois_f1"] > 0.5   # OIS >= ODS is not guaranteed
    # saturated (Dice-like) probabilities: AUC must still match sklearn
    sat = 1 / (1 + np.exp(-(np.where(gt, 9.0, -9.0) + rng.normal(0, 4, gt.shape))))
    ps2 = PixelStats()
    for p, g in zip(sat, gt):
        ps2.update(p, g)
    assert ps2.summary()["px_auc"] == pytest.approx(roc_auc_score(gt.ravel(), sat.ravel()), abs=2e-3)


# ------------------------------ Datasets ---------------------------------- #
@pytest.mark.parametrize("name", ["drive", "chase", "roads", "deepglobe", "crack500"])
def test_real_datasets_if_present(name):
    cfg = DataConfig(crop_size=256, samples_per_epoch=4)
    try:
        ds = build_dataset(name, "train", cfg)
    except FileNotFoundError:
        pytest.skip(f"{name} not extracted")
    s = ds[0]
    assert s["image"].shape == (3, 256, 256)
    for k in ("mask", "dist", "fov"):
        assert s[k].shape == (1, 256, 256)
    assert set(s["mask"].unique().tolist()) <= {0.0, 1.0}
    assert s["dist"].min() >= -1 and s["dist"].max() <= 1
    if s["mask"].sum() > 0:  # sign convention consistent with the mask
        assert (s["dist"][s["mask"] > 0] <= 0).all() and (s["dist"][s["mask"] == 0] > 0).all()
    ev = build_dataset(name, "test", cfg)
    e = ev[0]
    assert e["image"].shape[-2:] == e["mask"].shape[-2:]


# --------------------------- Tile-sequence memory -------------------------- #
def tiny_sam(trunk):
    """A SAM-2-like container with tiny neck / memory modules built from SAM 2's classes."""
    from types import SimpleNamespace
    from sam2.modeling.backbones.image_encoder import FpnNeck
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.transformer import RoPEAttention

    d, m = 256, 64
    neck = FpnNeck(PositionEmbeddingSine(d), d, list(trunk.channel_list), fpn_top_down_levels=[2, 3],
                   fpn_interp_model="nearest")
    menc = MemoryEncoder(m, MaskDownSampler(kernel_size=3, stride=2, padding=1),
                         Fuser(CXBlock(d, 7, 3), 1), PositionEmbeddingSine(m))
    layer = MemoryAttentionLayer("relu", RoPEAttention(d, 1, 1, kv_in_dim=m, rope_k_repeat=True,
                                                       feat_sizes=[4, 4]),
                                 d, 64, 0.0, False, True, False,
                                 RoPEAttention(d, 1, 1, feat_sizes=[4, 4]))
    matt = MemoryAttention(d, True, layer, 1)
    return SimpleNamespace(image_encoder=SimpleNamespace(trunk=trunk, neck=neck), memory_encoder=menc,
                           memory_attention=matt, maskmem_tpos_enc=torch.randn(7, 1, 1, m) * 0.02,
                           no_mem_embed=torch.randn(1, 1, d) * 0.02)


def test_tile_memory_model():
    pytest.importorskip("sam2")
    from sam2unet.engine import region_loss
    from sam2unet.memory import SAM2MemUNet
    torch.manual_seed(0)
    model = SAM2MemUNet(sam=tiny_sam(tiny_hiera()), lora_r=2, window=64, region=128, mem_lora_r=2)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    res = model.forward_region(x)
    assert len(res["frames"]) == 5 and len(res["final"]) == 4
    (main, aux), box = res["final"][3]
    assert main.shape == (2, 1, 64, 64) and box == (64, 64, 64)
    assert res["frames"][0][1] == (0, 0, 128)
    # zero-initialised context: pass 2 == pass 1 at initialisation (eval mode: in training,
    # BatchNorm statistics differ because pass 1 also normalises over the global view)
    model.eval()
    with torch.no_grad():
        r0 = model.forward_region(x)
    assert torch.allclose(r0["final"][0][0], r0["frames"][1][0], atol=1e-5)
    model.train()
    # loss + gradients reach trunk LoRA, memory LoRA, memory encoder and the new projections
    g = (torch.rand(2, 1, 128, 128) > 0.9).float()
    batch = {"mask": g, "dist": torch.zeros_like(g), "fov": torch.ones_like(g), "skel": g}
    lf = BoundaryAwareLoss(LossConfig(lambda_max=0.0, w_skel=1.0, w_cldice=0.5))
    loss, logs = region_loss(res, batch, lf, window=64)
    loss.backward()
    assert torch.isfinite(loss) and "skel" in logs and "cldice" in logs
    mem = model.memory
    assert mem.to_decoder.weight.grad.abs().sum() > 0
    assert any(p.grad is not None for n, p in mem.memory_attention.named_parameters() if "lora_B" in n)
    assert all(p.requires_grad for p in model.lora_parameters() + model.decoder_parameters())
    n_opt = len(model.lora_parameters()) + len(model.decoder_parameters())
    assert n_opt == sum(1 for p in model.parameters() if p.requires_grad)   # every trainable param optimised
    # checkpoint round trip (frozen neck / memory attention come from the SAM 2 weights)
    m2 = SAM2MemUNet(sam=tiny_sam(tiny_hiera()), lora_r=2, window=64, region=128, mem_lora_r=2)
    m2.encoder.trunk.load_state_dict(model.encoder.trunk.state_dict())
    m2.memory.neck.load_state_dict(mem.neck.state_dict())
    m2.memory.memory_attention.load_state_dict(mem.memory_attention.state_dict())
    m2.load_trainable_state_dict(model.trainable_state_dict())
    model.eval(); m2.eval()
    img = torch.randn(1, 3, 150, 200)
    p1, p2 = model.predict_full(img), m2.predict_full(img)
    assert p1.shape == (1, 1, 150, 200) and torch.allclose(p1, p2, atol=1e-5)
    assert 0 <= p1.min() and p1.max() <= 1


def test_topology_losses():
    from sam2unet.losses import SkeletonRecallLoss, SoftClDiceLoss, soft_skeleton
    g = torch.zeros(1, 1, 32, 32); g[..., 14:18, 2:30] = 1
    sk = torch.zeros_like(g); sk[..., 15:17, 3:29] = 1
    fov = torch.ones_like(g)
    assert SkeletonRecallLoss()(g, sk, fov) < 1e-6
    half = g.clone(); half[..., :, 16:] = 0
    assert abs(SkeletonRecallLoss()(half, sk, fov).item() - 0.5) < 0.1
    assert SoftClDiceLoss()(g, g, sk, fov) < 0.1 < SoftClDiceLoss()(half, g, sk, fov)
    s = soft_skeleton(g)
    assert s.sum() > 0 and s.max() <= 1 + 1e-6
