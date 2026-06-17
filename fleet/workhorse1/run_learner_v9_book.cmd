@echo off
setlocal
cd /d D:\awbw
if not exist logs mkdir logs
echo [%date% %time%] learner_v9_book start>> logs\train_learner_v9_book.log
python -m scripts.train_rhea_value_parallel ^
  --map-id 171596 ^
  --co-p0 14,8,28,7 ^
  --co-p1 14,8,28,7 ^
  --max-days 30 ^
  --rhea-autotune ^
  --save-every-transitions 1000 ^
  --reward-weight 0.6 ^
  --value-weight 0.4 ^
  --value-lr 1e-4 ^
  --replay-size 50000 ^
  --min-replay-before-train 1000 ^
  --updates-per-turn 1 ^
  --gamma-turn 0.9925 ^
  --target-update-interval 1000 ^
  --grad-clip 1.0 ^
  --device cuda ^
  --n-envs 9 ^
  --gpu-actors 3 ^
  --phi-capture-phase-weighting ^
  --phi-safe-neutral-opening-mult 1.50 ^
  --phi-safe-neutral-early-mid-mult 1.30 ^
  --phi-safe-neutral-mid-mult 1.15 ^
  --phi-safe-neutral-late-mult 1 ^
  --phi-safe-neutral-endgame-mult 0.50 ^
  --phi-contested-neutral-opening-mult 1.25 ^
  --phi-contested-neutral-mid-mult 1.00 ^
  --phi-contested-neutral-late-mult 0.90 ^
  --phi-capture-opening-end-day 5 ^
  --phi-capture-early-mid-end-day 8 ^
  --phi-capture-mid-end-day 12 ^
  --phi-capture-late-end-day 15 ^
  --dual-gradient-hist-prob 0.2 ^
  --dual-gradient-self-play ^
  --pairwise-zero-sum-reward ^
  --machine-id learner ^
  --checkpoint D:/awbw/checkpoints/value_rhea_latest.pt ^
  --push-gradients ^
  --buy-mode exhaustive ^
  --rhea-tactical-beam-max-width 96 ^
  --rhea-tactical-beam-max-depth 28 ^
  --rhea-pv-max-followup-pairs 4 ^
  --rhea-pv-inner-budget-scale 0.25 ^
  --rhea-adaptive-hard-turn-wall-s 900 ^
  --rhea-adaptive-extend ^
  --rhea-adaptive-max-extra-generations 20 ^
  --rhea-adaptive-patience-generations 4 ^
  --rhea-adaptive-min-improvement 0.0003 ^
  --capture-completion-bonus 0.03 ^
  --capture-progress-bonus 0.02 ^
  --neutral-income-gap-weight 0.04 ^
  --blunder-exposure-weight 0.01 ^
  --hq-defense-weight 0.01 ^
  --capture-interrupt-bonus 0.01 ^
  --buy-air-context-penalty 0.05 ^
  --opening-book-path D:/awbw/data/designed_desires_opening_book.jsonl ^
  --opening-book-prob 1.0 ^
  >> logs\train_learner_v9_book.log 2>&1
set EC=%ERRORLEVEL%
echo [%date% %time%] learner_v9_book exit=%EC%>> logs\train_learner_v9_book.log
exit /b %EC%
