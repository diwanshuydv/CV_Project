#!/bin/bash
# Unit tests + end-to-end smoke test (Hiera-L, 20 iters, 1-image eval on both datasets)
# on the A30 nodes where real training runs.   sbatch slurm/smoke.sbatch
#SBATCH --job-name=smoke
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
set -eo pipefail
source /scratch/b23cs1017/cv/slurm/env.sh
python -m pytest -q tests/
for DS in drive roads; do
  python train.py --exp_name smoke_$DS --train_dataset $DS --eval_datasets drive roads \
      --batch_size ${BS:-4} --max_iters_debug 20 --val_max_images 1 --num_workers 6 $EXTRA
done
