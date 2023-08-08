#!/bin/bash

python ice/train.py \
    -n "lilith-strong-bt150k-rampup" \
    --model "lilith" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 1 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --softactions \
    --bootstrap-frames 150000 \
    --rampup 200000