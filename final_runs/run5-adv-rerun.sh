#!/bin/bash

python ice/train.py \
    -n "final-BBLN-crps-explore-adv-RERUN" \
    --model "baseline1_ln_big" \
    --checkpoint "/mnt/qb/work2/goswami0/gkd021/code/rl-teamproject/models/final-BBLN-crps-explore-adv-phase2_20230809_06:30/frame_0000200000.pt" \
    --schedule "self-play" \
    -f 10000000 \
    --nsteps 4 \
    --rampup 100_000 \
    --eps-decay 2_000_000 \
    --cosine-annealing \
    --crps \
    --crps-explore \
    --no-shaping
