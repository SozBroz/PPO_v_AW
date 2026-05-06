#!/usr/bin/env python3
"""Add try/except around lines 1507-1758 in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Add try: before line 1507 (index 1506)
# Add except: after line 1758 (index 1757)
# But first, let me just add debug prints before each major function call

debug_prints = []
for i, line in enumerate(lines):
    if i > 1506 and i < 1758:  # Between inner try end and outer except
        stripped = line.strip()
        if stripped.startswith('exc = _run_replay_instrumented('):
            debug_prints.append((i, f'    print("[DEBUG] About to call _run_replay_instrumented", file=sys.stderr)\n'))
        elif stripped.startswith('base.envelopes_total = progress.envelopes_total'):
            debug_prints.append((i, f'    print("[DEBUG] Setting envelopes_total", file=sys.stderr)\n'))
        elif stripped.startswith('if exc is None:'):
            debug_prints.append((i, f'    print("[DEBUG] Checking exc is None", file=sys.stderr)\n'))
        elif stripped.startswith('if isinstance(exc, StateMismatchError):'):
            debug_prints.append((i, f'    print("[DEBUG] Checking StateMismatchError", file=sys.stderr)\n'))

# Insert in reverse order
for idx, debug_line in reversed(debug_prints):
    lines.insert(idx, debug_line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Added {len(debug_prints)} debug prints')
