#!/bin/bash

python ice/train.py \
    -n "ab1-lilith-strong-plain-nosoft" \
    --model "lilith" \
    --schedule "basic" \
    -f 5000000 \
    --nsteps 1 \
    --eps-decay 2_000_000 \
    --cosine-annealing
