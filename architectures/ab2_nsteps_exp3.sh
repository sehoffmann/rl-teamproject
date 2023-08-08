#!/bin/bash

python ice/train.py \
    -n "ab2-nsteps3" \
    --model "lilith" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 3 \
    --eps-decay 2_000_000 \
    --cosine-annealing