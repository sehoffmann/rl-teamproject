#!/usr/bin/env bash

PROJECT_DIR="$( dirname -- "$( readlink -f -- "$0"; )"; )/.."
SCRIPT="$PROJECT_DIR/scripts/imagenet_train.sbatch"
ARGS="--project-name=imagenet_benchmark --epochs 3"

$PROJECT_DIR/slurm/full_node.sh 1 --job-name="ImageNet_N1" --time="06:00:00" $SCRIPT $ARGS --experiment-name=imagenet_N1 
$PROJECT_DIR/slurm/full_node.sh 4 --job-name="ImageNet_N4" --time="04:00:00" $SCRIPT $ARGS --experiment-name=imagenet_N4
$PROJECT_DIR/slurm/full_node.sh 8 --job-name="ImageNet_N8" --time="02:00:00" $SCRIPT $ARGS --experiment-name=imagenet_N8
$PROJECT_DIR/slurm/full_node.sh 16 --job-name="ImageNet_N16" --time="01:00:00" $SCRIPT $ARGS --experiment-name=imagenet_N16