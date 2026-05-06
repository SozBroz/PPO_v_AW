#!/usr/bin/env python3
"""Check CO state mismatch examples"""
import json

with open('D:/awbw/logs/desync_register_state_mismatch_co_state.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        row = json.loads(line)
        print(f"games_id={row['games_id']}, day={row['approx_day']}, msg={row['message'][:120]}")
        print()
