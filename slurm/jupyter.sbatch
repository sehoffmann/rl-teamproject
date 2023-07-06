#!/bin/bash

#SBATCH --partition gpu-2080ti-interactive
#SBATCH --job-name=Jupyter
#SBATCH --time=0
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

#source $WORK/slurm/preamble.sh
#source ~/.bashrc
conda activate wb

cd $WORK2/jupyter
jupyter notebook
