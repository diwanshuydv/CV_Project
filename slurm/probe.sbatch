#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/scratch/b23cs1017/cv
#SBATCH --output=/scratch/b23cs1017/cv/logs/%x-%j.out
# Probe the compute node: GPU model, driver/CUDA version, CPUs, RAM, internet access.
hostname; nvidia-smi; nproc; free -g
timeout 10 curl -sI https://pypi.org | head -1 || echo "NO INTERNET"
