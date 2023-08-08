#!/bin/bash

python ice/train.py \
    -n "ab2-nsteps5" \
    --model "lilith" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 5 \
    --eps-decay 2_000_000 \
    --cosine-annealing
