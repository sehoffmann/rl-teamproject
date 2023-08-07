#!/bin/bash

python ice/train.py \
    -n "lilith-weak" \
    --model "lilith" \
    --schedule "lilith" \
    -f 3000000 \
    --nsteps 1 \
    --eps-decay 1_000_000 \
    --no-lilith-bootstrap