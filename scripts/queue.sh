#!/bin/bash
# Job feeder for the 2-submitted-jobs-per-user QoS.  Runs on the login node:
#   nohup bash scripts/queue.sh > /dev/null 2>&1 &
# Each non-empty, non-comment line of logs/queue.txt is the argument list of one `sbatch` call
# (e.g. "--job-name=x slurm/train.sbatch --exp_name x --train_dataset drive").  The first line
# is submitted whenever fewer than MAX jobs are queued/running and is then moved to
# logs/queue.done.  Edit logs/queue.txt at any time to add, drop or reorder jobs.
cd /scratch/b23cs1017/cv
Q=logs/queue.txt; DONE=logs/queue.done; LOG=logs/queue.log; MAX=${MAX:-2}
touch $Q $DONE
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "feeder started (pid $$)"
while true; do
  if [ "$(squeue -h -u $USER -p small,btech | wc -l)" -lt $MAX ]; then
    line=$(grep -v -e '^\s*$' -e '^\s*#' $Q | head -1)
    if [ -n "$line" ]; then
      if id=$(sbatch --parsable $line 2>>$LOG); then
        log "submitted $id: $line"; echo "$id $line" >> $DONE
      else
        log "FAILED: $line"; echo "FAILED $line" >> $DONE
      fi
      # drop the first matching line only
      awk -v l="$line" 'BEGIN{d=0} {if(!d && $0==l){d=1; next} print}' $Q > $Q.tmp && mv $Q.tmp $Q
    fi
  fi
  sleep 60
done
