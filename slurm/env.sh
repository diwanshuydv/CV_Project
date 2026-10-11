# Sourced by every sbatch script: activate the uv venv and keep all caches on /scratch.
export ROOT=/scratch/b23cs1017/cv
source $ROOT/.venv/bin/activate
export UV_CACHE_DIR=$ROOT/.cache/uv TORCH_HOME=$ROOT/.cache/torch HF_HOME=$ROOT/.cache/hf
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $ROOT
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
