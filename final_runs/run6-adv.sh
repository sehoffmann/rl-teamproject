#!/bin/bash

python ice/train.py \
    -n "final-baseline1-LN-adv" \
    --model "baseline1_layernorm" \
    --advanced-schedule
