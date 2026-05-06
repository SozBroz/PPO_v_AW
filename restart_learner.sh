#!/bin/bash
# restart_learner_with_push_gradients.sh
# Run this on 192.168.0.160 (ai_machine / learner)

cd D:/awbw
D:/python3.12.exe scripts/train_rhea_value_parallel.py \
  --checkpoint D:/awbw/checkpoints/value_rhea_latest.pt \
  --map-id 171596 \
  --co-p0 14,8,28,7 \
  --co-p1 14,8,28,7 \
  --max-days 30 \
  --rhea-autotune \
  --save-every-transitions 1000 \
  --reward-weight 0.8 \
  --value-weight 0.2 \
  --value-lr 1e-4 \
  --replay-size 50000 \
  --min-replay-before-train 1000 \
  --updates-per-turn 1 \
  --gamma-turn 0.99 \
  --target-update-interval 1000 \
  --grad-clip 1.0 \
  --device cuda \
  --n-envs 20 \
  --gpu-actors 6 \
  --phi-capture-phase-weighting \
  --phi-safe-neutral-opening-mult 1.50 \
  --phi-safe-neutral-early-mid-mult 1.30 \
  --phi-safe-neutral-mid-mult 1.15 \
  --phi-safe-neutral-late-mult 1 \
  --phi-safe-neutral-endgame-mult 0.50 \
  --phi-contested-neutral-opening-mult 1.25 \
  --phi-contested-neutral-mid-mult 1.00 \
  --phi-contested-neutral-late-mult 0.90 \
  --phi-capture-opening-end-day 5 \
  --phi-capture-early-mid-end-day 8 \
  --phi-capture-mid-end-day 12 \
  --phi-capture-late-end-day 15 \
  --dual-gradient-hist-prob 0.2 \
  --dual-gradient-self-play \
  --pairwise-zero-sum-reward \
  --push-gradients \
  --gradient-shared-root Z: \
  2>&1 | tee D:/awbw/logs/learner_output.log
