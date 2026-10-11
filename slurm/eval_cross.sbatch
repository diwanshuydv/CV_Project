#!/bin/bash
# Cross-domain evaluation matrix; args forwarded to evaluate.py, e.g.
#   sbatch slurm/eval_cross.sbatch --runs runs/drive_bnd runs/roads_bnd runs/drive_base runs/roads_base
#SBATCH --job-name=eval_cross
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python evaluate.py --save_preds "$@"
