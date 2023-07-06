#!/bin/bash
N_GPUS=$1
PARTITION="gpu-2080ti"

if (( $# < 2 )); then
    echo "Usage: slurm_job N_GPUS SCRIPT [ARGS...]"
    exit 1
fi

# logging
LOG_DIR="$WORK2/slurm-logs/`date '+%Y/%b/%d'`"
LOG_FILE="$LOG_DIR/`date '+%H_%M'`.%x.%j"

mkdir -p $LOG_DIR

# basic task setup
let "MAX_NODES = (N_GPUS+7) / 8"
ARGS+=" -p $PARTITION"
ARGS+=" -N $MAX_NODES" # important to prevent getting too many nodes
ARGS+=" -n $N_GPUS"

# allocate full node
ARGS+=" --gpus-per-node=8"
ARGS+=" --cpus-per-gpu=8"
ARGS+=" -B 2:18:2"

# cpu <-> task distribution
ARGS+=" --cpus-per-task=8"
ARGS+=" -m block:block:block"

# misc flags
ARGS+=" --gres-flags=enforce-binding"
ARGS+=" --exclusive=user"
ARGS+=" --output=$LOG_FILE"
ARGS+=" --open-mode=append"  # important for requeued jobs
ARGS+=" -v"

# lets go
echo "Allocating $N_GPUS gpus across $MAX_NODES $PARTITION nodes"
echo "> sbatch $ARGS ${@:2}"
echo
sbatch $ARGS ${@:2}

echo
echo "Writing log to $LOG_FILE"
