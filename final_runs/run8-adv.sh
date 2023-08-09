#!/bin/bash

python ice/train.py \
    -n "final-BBLN-DoubleQ-adv" \
    --model "baseline1_ln_big" \
    --advanced-schedule \
    --double-q \
    --no-dueling
