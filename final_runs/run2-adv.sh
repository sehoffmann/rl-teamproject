#!/bin/bash

python ice/train.py \
    -n "final-BBLN-stacked-adv" \
    --model "baseline1_ln_big" \
    --advanced-schedule \
    --frame-stacks 3
