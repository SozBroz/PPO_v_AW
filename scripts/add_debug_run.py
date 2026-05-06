#!/usr/bin/env python3
"""Add debug print at start of _run_replay_instrumented"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "progress.envelopes_total = len(envelopes)" (line 697)
# and add debug print before it
for i, line in enumerate(lines):
    if 'progress.envelopes_total = len(envelopes)' in line:
        print(f'Found target line at index {i} (line {i+1})')
        debug_line = '    print(f"[DEBUG] _run_replay_instrumented: entered, n_envelopes={len(envelopes)}", file=sys.stderr)\n'
        lines.insert(i, debug_line)
        break

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done - added debug print at start of _run_replay_instrumented')
