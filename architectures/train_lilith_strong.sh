#!/bin/bash

python ice/train.py \
    -n "lilith-strong" \
    --model "lilith" \
    --schedule "basic" \
    -f 15000000 \
    --nsteps 1 \
    --eps-decay 4_000_000 \
    --warmup-frames 300000 \
    --cosine-annealing