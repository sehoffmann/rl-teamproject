#!/bin/bash

python ice/train.py \
    -n "ab4-baseline1_layernorm" \
    --model "baseline1_layernorm" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
