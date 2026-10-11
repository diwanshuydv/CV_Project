#!/bin/bash
# Feeds jobs to SLURM respecting QOSMaxSubmitJobPerUserLimit (2 jobs/user).
# Runs on the login node (nohup); log: logs/submit_queue.log
# The final cross-domain eval is submitted with afterok on any still-active training job
# and only if every finished training job COMPLETED.
cd /scratch/b23cs1017/cv
MAX=2
ALL_IDS="43092 43093"   # already submitted: drive_bnd, drive_base
QUEUE=(
  "--job-name=roads_bnd  slurm/train.sbatch --exp_name roads_bnd  --train_dataset roads --eval_datasets roads drive"
  "--job-name=roads_base slurm/train.sbatch --exp_name roads_base --train_dataset roads --eval_datasets roads drive --lambda_max 0"
)
log() { echo "$(date '+%F %T') $*"; }
wait_slot() { while [ "$(squeue -h -u $USER | wc -l)" -ge $MAX ]; do sleep 120; done; }

for args in "${QUEUE[@]}"; do
  wait_slot
  id=$(sbatch --parsable $args) || { log "FAILED to submit: $args"; exit 1; }
  ALL_IDS="$ALL_IDS $id"; log "submitted $id: $args"
done

wait_slot
active=""; failed=""
for id in $ALL_IDS; do
  if squeue -h -j $id 2>/dev/null | grep -q .; then active="$active:$id"
  else
    st=$(sacct -j $id -X -n -o State | head -1 | tr -d ' ')
    [ "$st" = "COMPLETED" ] || failed="$failed $id($st)"
  fi
done
if [ -n "$failed" ]; then log "NOT submitting eval_cross: training jobs failed:$failed"; exit 1; fi
dep=""; [ -n "$active" ] && dep="--dependency=afterok$active"
id=$(sbatch --parsable $dep slurm/eval_cross.sbatch --runs runs/drive_bnd runs/roads_bnd runs/drive_base runs/roads_base)
log "submitted eval_cross $id $dep"
