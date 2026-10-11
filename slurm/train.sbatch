#!/bin/bash
# Full training run; all CLI args are forwarded to train.py, e.g.
#   sbatch slurm/train.sbatch --exp_name drive_bnd --train_dataset drive --eval_datasets drive roads
# 'small' allows 36 h; job auto-resumes from <run>/last.pt if re-submitted
# (chain with: sbatch --dependency=afterany:<jobid> slurm/train.sbatch <same args>).
#SBATCH --job-name=sam2unet
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python train.py --resume "$@"
