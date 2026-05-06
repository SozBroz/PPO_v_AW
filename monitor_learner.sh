#!/bin/bash
# monitor_learner.sh - Monitor learner training on 192.168.0.160
ssh sshuser@192.168.0.160 << 'EOF'
cd D:\awbw
echo "=== Processes ==="
tasklist | findstr python
echo ""
echo "=== Recent Log Entries ==="
powershell -Command "Get-Content logs\games_log.jsonl -Tail 20" 2>&1
EOF
