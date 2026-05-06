#!/usr/bin/env python3
"""Add try/except around lines 1507-1755 in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line numbers (0-indexed)
# Line 1507 (1-indexed) = index 1506
# We want to add try: before line 1507 and except after line 1755

# Actually, let me just add try: before line 1507 (index 1506)
# and except after line 1755 (index 1754)

# First, let me find where the outer try/except is
outer_try_line = None
outer_except_line = None

for i, line in enumerate(lines):
    if 'for gid, zpath, meta in targets:' in line:
        print(f'Found targets loop at line {i+1}')
    if i > 1750 and 'except Exception as exc:  # safety net' in line:
        outer_except_line = i
        print(f'Found outer except at line {i+1}')
        break

# Let me add try: before line 1507 (progress = _ReplayProgress())
# and add except after line 1755 (before the outer except)

# Actually, let me just add a simple debug print before each major operation
# between line 1507 and 1755

debug_prints = []
for i in range(1506, 1755):
    line = lines[i]
    if 'exc = _run_replay_instrumented(' in line:
        debug_prints.append((i, '    print(f"[DEBUG] About to call _run_replay_instrumented", file=sys.stderr)\n'))
    elif 'base.envelopes_total = progress.envelopes_total' in line:
        debug_prints.append((i, '    print(f"[DEBUG] About to set envelopes_total", file=sys.stderr)\n'))
    elif 'if exc is None:' in line:
        debug_prints.append((i, '    print(f"[DEBUG] Checking exc is None", file=sys.stderr)\n'))

# Insert in reverse order to preserve indices
for idx, debug_line in reversed(debug_prints):
    lines.insert(idx, debug_line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Added {len(debug_prints)} debug prints')
