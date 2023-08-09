#!/bin/bash

python ice/train.py \
    -n "final-BBLN-crps-explore-adv" \
    --model "baseline1_ln_big" \
    --advanced-schedule \
    --crps \
    --crps-explore
