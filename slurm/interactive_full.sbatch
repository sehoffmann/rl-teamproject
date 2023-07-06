#!/bin/bash

#SBATCH --partition gpu-2080ti
#SBATCH --job-name=Interactive
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#ARGS+=" --cpus-per-task=8"
#ARGS+=" -m block:block:block"
#SBATCH --ntasks=8

#source $WORK/slurm/preamble.sh
while true; do sleep 1; done
