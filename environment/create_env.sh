#!/bin/bash
set -e

if (( $# < 1 )); then
    echo "Usage: ./scripts/create_conda_env ENV_NAME"
    exit 1
fi

if ! mamba -V &> /dev/null; then
    echo "mamba not found! Make sure that you have mamba installed. \"conda install -c conda-forge mamba\""
    exit 1
fi

CONDA_ENVS_DIR=`conda info | grep "envs directories"`
CONDA_ENVS_DIR="/${CONDA_ENVS_DIR#*/}"

export HOROVOD_CUDA_HOME="$CONDA_ENVS_DIR/$1"
export HOROVOD_NCCL_HOME="$CONDA_ENVS_DIR/$1"
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_MPI=1

mamba env create --name $1 --file=environment.yml --force
conda activate $1
horovodrun --check-build
