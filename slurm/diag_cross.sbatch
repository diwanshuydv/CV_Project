#!/bin/bash
# Probe why roads-trained models output ~0 on DRIVE; args forwarded to scripts/diag_cross.py.
#SBATCH --job-name=diag_cross
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python scripts/diag_cross.py "$@"
