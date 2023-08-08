#!/bin/bash

python ice/train.py \
    -n "ab5-baseline1_ln_big_stack" \
    --model "baseline1_ln_big" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
    --frame-stacks 3
