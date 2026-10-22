#!/bin/bash
# Smoke test of the tile-sequence-memory model on the A30 nodes: memory footprint + speed.
#   sbatch slurm/smoke_tsm.sbatch
#SBATCH --job-name=smoke_tsm
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python train.py --exp_name smoke_tsm_drive --train_dataset drive --model tsm --window 384 \
    --batch_size 2 --grad_ckpt --w_skel 1 --max_iters_debug 30 --val_max_images 1 --num_workers 6
python train.py --exp_name smoke_tsm_roads --train_dataset roads --model tsm --window 512 \
    --batch_size 2 --grad_ckpt --w_skel 1 --max_iters_debug 30 --val_max_images 1 --num_workers 6
