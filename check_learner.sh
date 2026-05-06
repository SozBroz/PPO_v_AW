#!/bin/bash
# check_learner.sh - Run on learner (192.168.0.160)
cd D:/awbw || cd /mnt/awbw || exit 1
hostname
pwd
ls -la checkpoints/value_rhea_latest.pt 2>&1 || echo "Checkpoint not found"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>&1 || echo "nvidia-smi not available"
