#!/bin/bash

#SBATCH --job-name=ImageNet
#SBATCH --time=16:00:00

source ~/.bashrc
source $PREAMBLE
conda activate wb

ARGS+=" --project-name=imagenet"
ARGS+=" --data=/mnt/qb/datasets/ImageNet2012/"
ARGS+=" --direct-path"
srun --mpi=pmix dmlcloud-train imagenet $ARGS ${@:1}