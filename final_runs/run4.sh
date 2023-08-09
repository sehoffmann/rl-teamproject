#!/bin/bash

python ice/train.py \
    -n "final-BBLN-crps" \
    --model "baseline1_ln_big" \
    --schedule "basic" \
    -f 15000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
    --crps
