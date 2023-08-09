#!/bin/bash

python ice/train.py \
    -n "final-baseline1-LN-stacked-adv" \
    --model "baseline1_layernorm" \
    --advanced-schedule \
    --frame-stacks 3
