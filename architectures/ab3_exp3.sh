#!/bin/bash

python ice/train.py \
    -n "ab3-PER-Dueling" \
    --model "lilith" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 4 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --per
