#!/usr/bin/env python3
"""Add debug prints to trace the exact failing line in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Debug prints to add (line_number, debug_message)
# Line numbers are 1-indexed, so index = line_number - 1
debug_inserts = []

for i, line in enumerate(lines):
    stripped = line.strip()
    # Add debug before _run_replay_instrumented call
    if stripped.startswith('exc = _run_replay_instrumented('):
        debug_inserts.append((i, '    print("[DEBUG] About to call _run_replay_instrumented", file=sys.stderr)\n'))
    # Add debug before setting envelopes_total
    elif stripped.startswith('base.envelopes_total = progress.envelopes_total'):
        debug_inserts.append((i, '    print("[DEBUG] Setting envelopes_total", file=sys.stderr)\n'))
    # Add debug before checking exc is None
    elif stripped.startswith('if exc is None:'):
        debug_inserts.append((i, '    print("[DEBUG] Checking exc is None", file=sys.stderr)\n'))
    # Add debug before checking StateMismatchError
    elif stripped.startswith('if isinstance(exc, StateMismatchError):'):
        debug_inserts.append((i, '    print("[DEBUG] Checking StateMismatchError", file=sys.stderr)\n'))
    # Add debug before returning base (first_divergence path)
    elif stripped.startswith('        return base') and i > 1500:
        debug_inserts.append((i, '        print("[DEBUG] Returning base (first_divergence)", file=sys.stderr)\n'))

# Insert in reverse order to preserve indices
for idx, debug_line in reversed(debug_inserts):
    lines.insert(idx, debug_line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Added {len(debug_inserts)} debug prints')
