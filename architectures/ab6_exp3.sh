#!/bin/bash

python ice/train.py \
    -n "ab6-base1lnbig" \
    --model "baseline1_ln_big" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --bootstrap-frames 300_000 \
    --crps
