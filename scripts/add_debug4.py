#!/usr/bin/env python3
"""Add debug prints to find exact failing line in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Add debug prints before each major function call between line 1507 and 1758
debug_prints = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if 1506 <= i <= 1758:  # Between inner try end and outer except
        if stripped.startswith('exc = _run_replay_instrumented('):
            debug_prints.append((i, '    print("[DEBUG] About to call _run_replay_instrumented", file=sys.stderr)\n'))
        elif stripped.startswith('base.envelopes_total = progress.envelopes_total'):
            debug_prints.append((i, '    print("[DEBUG] Setting envelopes_total", file=sys.stderr)\n'))
        elif stripped.startswith('if exc is None:'):
            debug_prints.append((i, '    print("[DEBUG] Checking exc is None", file=sys.stderr)\n'))
        elif stripped.startswith('if isinstance(exc, StateMismatchError):'):
            debug_prints.append((i, '    print("[DEBUG] Checking StateMismatchError", file=sys.stderr)\n'))

# Insert in reverse order
for idx, debug_line in reversed(debug_prints):
    lines.insert(idx, debug_line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Added {len(debug_prints)} debug prints')
