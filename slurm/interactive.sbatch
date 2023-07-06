#!/bin/bash

#SBATCH --partition gpu-2080ti-interactive
#SBATCH --job-name=Interactive
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

#source $WORK/slurm/preamble.sh
while true; do sleep 1; done
