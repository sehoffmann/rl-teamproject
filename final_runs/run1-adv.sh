#!/bin/bash

python ice/train.py \
    -n "final-BBLN-adv" \
    --model "baseline1_ln_big" \
    --advanced-schedule