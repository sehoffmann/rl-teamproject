#!/bin/bash

#SBATCH --partition gpu-2080ti
#SBATCH --job-name=EV-CRPS
#SBATCH --time=2-00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

source ~/.bashrc
source $PREAMBLE
conda activate wb

while true
do
    srun python -u ice/deploy_remote.py --preset crps "$@"
done