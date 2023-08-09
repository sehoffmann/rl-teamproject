#!/bin/bash

python ice/train.py \
    -n "final-lstm-big-adv" \
    --model "LSTM-big" \
    --advanced-schedule \
    --frame-stacks 3
