"""
Train SAM2-UNet (frozen SAM 2.1 Hiera + LoRA + U-Net decoder) with the
boundary-aware loss, then evaluate the best checkpoint in-domain AND
cross-domain.

Examples
--------
# medical -> (medical, remote sensing)
python train.py --train_dataset drive --eval_datasets drive roads --exp_name drive_bnd

# remote sensing -> (remote sensing, medical), plain BCE+Dice baseline
python train.py --train_dataset roads --eval_datasets roads drive --lambda_max 0 --exp_name roads_base
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sam2unet.datasets import DATASETS, DEFAULT_SAMPLES_PER_EPOCH, DataConfig, build_dataset, build_loader
from sam2unet.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from sam2unet.factory import build_model
from sam2unet.losses import BoundaryAwareLoss, LossConfig
from sam2unet.model import SAM2_VARIANTS
from sam2unet.utils import amp_dtype, get_device, get_logger, save_checkpoint, save_json, seed_everything

ROOT = Path("/scratch/b23cs1017/cv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # experiment / data
    p.add_argument("--exp_name", required=True)
    p.add_argument("--out_root", default=str(ROOT / "runs"))
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--train_dataset", choices=DATASETS, required=True)
    p.add_argument("--eval_datasets", nargs="+", default=None,
                   help="datasets for the final test evaluation (default: the training dataset)")
    p.add_argument("--trainval", action="store_true",
                   help="train on train+val (final runs); the last epoch is evaluated")
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--scale", type=float, default=1.0,
                   help="resample images by this factor for training and inference (metrics at 1x)")
    p.add_argument("--samples_per_epoch", type=int, default=None,
                   help="random crops per epoch (default: DEFAULT_SAMPLES_PER_EPOCH[dataset])")
    p.add_argument("--max_dist", type=float, default=20.0, help="EDT clipping distance (px)")
    p.add_argument("--drive_fold", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    # model
    p.add_argument("--variant", choices=list(SAM2_VARIANTS), default="large")
    p.add_argument("--sam2_ckpt", default=None, help="default: checkpoints/<variant file>")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing in Hiera")
    p.add_argument("--no_deep_supervision", action="store_true")
    p.add_argument("--hf_prior", action="store_true",
                   help="concatenate fixed Sobel / Haar-DWT high-frequency maps into the stride-4/2/1 skips")
    p.add_argument("--model", choices=["sam2unet", "tsm"], default="sam2unet",
                   help="tsm = SAM2-UNet + tile-sequence memory (trains on regions of --region px)")
    p.add_argument("--memory", choices=["both", "global", "tiles"], default="both",
                   help="tsm memory bank: global view + tiles, global view only, tiles only")
    p.add_argument("--region", type=int, default=None, help="tsm region size (default 2 x window)")
    p.add_argument("--mem_lora_r", type=int, default=8)
    p.add_argument("--w_pass1", type=float, default=0.5, help="tsm: weight of the pass-1 (no-memory) loss")
    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr_lora", type=float, default=2e-4)
    p.add_argument("--lr_decoder", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # loss
    p.add_argument("--lambda_0", type=float, default=0.01)
    p.add_argument("--lambda_max", type=float, default=0.5, help="0 = BCE+Dice baseline")
    p.add_argument("--ramp_fraction", type=float, default=0.3)
    p.add_argument("--w_err", type=float, default=1.0)
    p.add_argument("--w_sdf", type=float, default=1.0)
    p.add_argument("--w_edge", type=float, default=1.0)
    p.add_argument("--edge_alpha", type=float, default=4.0)
    p.add_argument("--edge_tau", type=float, default=0.1)
    p.add_argument("--w_skel", type=float, default=0.0, help="skeleton recall loss weight")
    p.add_argument("--w_cldice", type=float, default=0.0, help="soft clDice loss weight")
    # evaluation
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--val_max_images", type=int, default=None)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--threshold", default="val", choices=["val", "0.5"],
                   help="test threshold: F1-optimal on the validation set, or fixed 0.5")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true", help="resume from <run>/last.pt if present")
    p.add_argument("--max_iters_debug", type=int, default=None, help="smoke tests only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.model == "tsm":
        args.region = args.region or 2 * args.window
        args.crop_size = args.region          # one training sample = one region
    seed_everything(args.seed)
    out_dir = Path(args.out_root) / args.exp_name
    if args.resume and (out_dir / "results.json").exists():
        print(f"{out_dir} already finished (results.json exists) -- nothing to do")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name()} | AMP dtype: {amp_dtype(device)}")
    logger.info(f"args: {vars(args)}")
    save_json(vars(args), out_dir / "args.json")

    # ---------------- data ----------------
    spe = args.samples_per_epoch or DEFAULT_SAMPLES_PER_EPOCH[args.train_dataset]
    dcfg = DataConfig(data_root=args.data_root, crop_size=args.crop_size, samples_per_epoch=spe,
                      max_dist=args.max_dist, seed=args.seed, drive_fold=args.drive_fold,
                      trainval=args.trainval, scale=args.scale)
    train_ds = build_dataset(args.train_dataset, "train", dcfg)
    val_ds = build_dataset(args.train_dataset, "val", dcfg)
    train_loader = build_loader(train_ds, args.batch_size, args.num_workers)
    val_loader = build_loader(val_ds, 1, max(1, args.num_workers // 2))
    logger.info(f"train: {args.train_dataset} {len(train_ds.records)} images, {len(train_ds)} crops/epoch "
                f"| val: {len(val_ds)} images")

    # ---------------- model ----------------
    model = build_model(vars(args)).to(device)
    logger.info(model.parameter_summary())

    # ---------------- optimisation ----------------
    loss_fn = BoundaryAwareLoss(LossConfig(
        lambda_0=args.lambda_0, lambda_max=args.lambda_max, ramp_fraction=args.ramp_fraction,
        w_err=args.w_err, w_sdf=args.w_sdf, w_edge=args.w_edge, alpha=args.edge_alpha,
        tau=args.edge_tau, w_skel=args.w_skel, w_cldice=args.w_cldice)).to(device)
    optimizer = build_optimizer(model, args.lr_lora, args.lr_decoder, args.weight_decay)
    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    scheduler = build_scheduler(optimizer, total_iters, int(args.warmup_epochs * iters_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(out_dir / "tb")

    start_epoch, best = 0, -1.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        logger.info(f"Resumed from {last_path} at epoch {start_epoch} (best val dice {best:.4f})")

    # ---------------- training loop ----------------
    if args.max_iters_debug:  # smoke test: shorten everything
        args.epochs = start_epoch + 1
        train_loader = [b for _, b in zip(range(args.max_iters_debug), train_loader)]
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        stats = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device,
                                epoch, total_iters, logger, writer, args.grad_clip, w_pass1=args.w_pass1)
        logger.info(f"epoch {epoch} done in {time.time() - t0:.0f}s | " +
                    " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        if device.type == "cuda" and epoch == start_epoch:
            logger.info(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            res = evaluate(model, val_loader, device, args.window, args.overlap,
                           max_images=args.val_max_images, topology=False, test_scale=args.scale)["summary"]
            for k in ("iou", "dice", "hd95"):
                writer.add_scalar(f"val/{k}", res[k], epoch)
            logger.info(f"[val] epoch {epoch} IoU {res['iou']:.4f} Dice {res['dice']:.4f} "
                        f"HD95 {res['hd95']:.2f}")
            if res["dice"] > best:
                best = res["dice"]
                save_checkpoint(best_path, model, epoch=epoch, best=best, args=vars(args))
                logger.info(f"  new best (Dice {best:.4f}) -> {best_path}")
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best, vars(args))

    # ---------------- final in-domain + cross-domain test ----------------
    if best_path.exists() and not args.trainval:
        model.load_trainable_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    # Test threshold: 0.5, or the pooled-F1-optimal threshold on the validation set.
    thr = 0.5
    if args.threshold == "val" and not args.trainval:
        thr = evaluate(model, val_loader, device, args.window, args.overlap,
                       max_images=args.val_max_images, topology=False,
                       test_scale=args.scale)["summary"]["ods_thr"]
        logger.info(f"threshold selected on val: {thr:.3f}")
    results = {}
    for name in args.eval_datasets or [args.train_dataset]:
        test_loader = build_loader(build_dataset(name, "test", dcfg), 1, max(1, args.num_workers // 2))
        r = evaluate(model, test_loader, device, args.window, args.overlap, threshold=thr, test_scale=args.scale,
                     save_dir=out_dir / "preds" / name,
                     max_images=args.val_max_images if args.max_iters_debug else None)
        results[name] = r
        save_json(r, out_dir / f"eval_test_{name}.json")
        s = r["summary"]
        tag = "in-domain" if name == args.train_dataset else "CROSS-domain"
        logger.info(f"[test {tag}] {args.train_dataset} -> {name} (thr {thr:.2f}, n={s['n']}): "
                    f"Dice {s['dice']:.4f} IoU {s['iou']:.4f} clDice {s['cldice']:.4f} "
                    f"HD95 {s['hd95']:.2f} b0err {s['betti0_err_patch']:.3f} | pooled F1 {s['px_f1']:.4f} "
                    f"AUC {s['px_auc']:.4f} Se {s['px_se']:.4f} Sp {s['px_sp']:.4f} Acc {s['px_acc']:.4f}")
    save_json(results, out_dir / "results.json")
    writer.close()


if __name__ == "__main__":
    main()
