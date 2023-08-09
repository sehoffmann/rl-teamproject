#!/bin/bash

python ice/train.py \
    -n "final-baseline1-LN-stacked" \
    --model "baseline1_layernorm" \
    --schedule "basic" \
    -f 15000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
    --frame-stacks 3
