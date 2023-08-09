#!/bin/bash

python ice/train.py \
    -n "final-BBLN-crps-adv" \
    --model "baseline1_ln_big" \
    --advanced-schedule \
    --crps
