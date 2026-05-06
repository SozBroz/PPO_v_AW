#!/bin/bash
# Check learner training status
ssh sshuser@192.168.0.160 << 'EOF'
cd D:\awbw
echo "=== Learner Processes ==="
tasklist | findstr python
echo ""
echo "=== Recent Training Events ==="
powershell -Command "Get-Content logs\\games_log.jsonl -First 100 2>&1 | Select-String -Pattern 'event.*(transition|value_loss|gradients_applied|heartbeat)' | Select-Object -First 50"
EOF'
