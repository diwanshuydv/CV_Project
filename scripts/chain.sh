#!/bin/bash
# Submit a training run as a dependency chain of 1-hour chunks on the `test` partition.
#   bash scripts/chain.sh <n_chunks> --exp_name x --train_dataset drive ...
# Every chunk resumes from runs/<exp>/last.pt; chunks after the run has finished exit at once.
cd /scratch/b23cs1017/cv
n=$1; shift
name=$(echo "$@" | sed -n 's/.*--exp_name \([^ ]*\).*/\1/p')
dep=""; ids=""
for i in $(seq 1 $n); do
  id=$(sbatch --parsable --job-name="${name}" $dep slurm/train_test.sbatch "$@") || exit 1
  dep="--dependency=afterany:$id"; ids="$ids $id"
done
echo "$(date '+%F %T') chain $name ($n x 1h):$ids" | tee -a logs/queue.log
