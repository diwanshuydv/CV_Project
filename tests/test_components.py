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
from sam2unet.metrics import compute_metrics, hausdorff_metrics, hausdorff_scipy
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
    assert compute_metrics(gt, gt) == {"iou": 1.0, "dice": 1.0, "hd95": 0.0, "hd": 0.0, "assd": 0.0}
    assert hausdorff_metrics(np.zeros_like(gt), gt)["hd95"] == pytest.approx(np.hypot(64, 64))


def test_random_masks_hd_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.random((40, 40)) > 0.7, rng.random((40, 40)) > 0.7
        assert hausdorff_metrics(a, b)["hd"] == pytest.approx(hausdorff_scipy(a, b))


# ------------------------------ Datasets ---------------------------------- #
@pytest.mark.parametrize("name", ["drive", "roads"])
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
