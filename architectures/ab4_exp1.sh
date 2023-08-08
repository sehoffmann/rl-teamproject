#!/bin/bash

python ice/train.py \
    -n "ab4-baseline1" \
    --model "baseline1" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
