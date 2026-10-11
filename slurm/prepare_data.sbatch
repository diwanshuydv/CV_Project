#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/b23cs1017/cv
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
# CPU-only job: unzip DRIVE + Massachusetts Roads (~6 GB) off the login node.
/usr/bin/python3 scripts/prepare_data.py --data_root /scratch/b23cs1017/cv/data
