#!/bin/bash

#SBATCH --partition gpu-2080ti
#SBATCH --job-name=RL-EVAL
#SBATCH --time=2-00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

source ~/.bashrc
source $PREAMBLE
conda activate wb

srun --mpi=pmix python ice/deploy_remote.py --username "Tübingen Ice Q-Learners" --password fie9Amai6r "$@"