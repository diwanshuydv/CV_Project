#!/bin/bash
# One <=1 h chunk of a training run on the `test` partition (no per-user GPU QoS there).
# Runs resume from <run>/last.pt (checkpointed every epoch) and exit at once when finished,
# so a long run is a chain of these:  bash scripts/chain.sh <n_chunks> <train.py args...>
#SBATCH --job-name=sam2unet_t
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python train.py --resume "$@"
