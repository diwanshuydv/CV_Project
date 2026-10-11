# SAM2-UNet + LoRA + Distance-Transform Boundary Loss

Thin-structure segmentation (DRIVE retinal vessels, Massachusetts Roads) with a frozen
SAM 2.1 Hiera-L encoder, LoRA adapters, a residual U-Net decoder, and
`L = BCE + Dice + λ(t)·L_boundary`.

| Step | File |
|---|---|
| 1. Dataset (+ CPU EDT signed distance map) | `sam2unet/datasets.py` |
| 2. LoRA / model | `sam2unet/lora.py`, `sam2unet/model.py` |
| 3. Boundary-aware loss | `sam2unet/losses.py` |
| 4. Train / eval / metrics (IoU, Dice, HD95) | `sam2unet/engine.py`, `sam2unet/metrics.py`, `train.py`, `evaluate.py` |

Everything (code, data, `.venv`, checkpoints, runs, logs) lives in `/scratch/b23cs1017/cv`.

## Setup (login node, once)
```bash
bash scripts/setup_env.sh          # uv venv (.venv) + torch 2.7.1 cu128 + sam2 + Hiera-L ckpt
sbatch slurm/prepare_data.sbatch   # unzip data/*.zip -> data/DRIVE, data/roads
```

## Cluster notes
- `small` (36 h) / `btech` (12 h): cn04–05, **A30 24 GB** → real training.
- `test` (1 h): includes cn07 (RTX PRO 6000 96 GB) → debugging only.
- QoS: 1 GPU per user. Jobs resume from `runs/<exp>/last.pt` (`--resume` is always on in `train.sbatch`).

## Run
```bash
sbatch slurm/smoke.sbatch                                        # tests + 20-iter end-to-end check
# boundary loss
sbatch slurm/train.sbatch --exp_name drive_bnd --train_dataset drive --eval_datasets drive roads
sbatch slurm/train.sbatch --exp_name roads_bnd --train_dataset roads --eval_datasets roads drive
# baselines (BCE + Dice only)
sbatch slurm/train.sbatch --exp_name drive_base --train_dataset drive --lambda_max 0
sbatch slurm/train.sbatch --exp_name roads_base --train_dataset roads --lambda_max 0
# cross-domain matrix -> runs/cross_domain.md
sbatch slurm/eval_cross.sbatch --runs runs/drive_bnd runs/roads_bnd runs/drive_base runs/roads_base
```
Monitor: `tail -f logs/sam2unet-<jobid>.out`, `tensorboard --logdir runs`.

## Notes
- DRIVE's public test set has no vessel labels → the 20 labelled images are split 14/2/4
  (seeded, saved to `data/splits/`; `--drive_fold k` for 5-fold CV).
- Evaluation is full-resolution sliding-window (512 px tiles, Gaussian blending), metrics inside the FOV.
- HD95 empty-mask convention: image diagonal.
