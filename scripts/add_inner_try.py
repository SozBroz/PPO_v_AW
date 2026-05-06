#!/usr/bin/env python3
"""Add try/except around _run_replay_instrumented call in _audit_one"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "exc = _run_replay_instrumented(" (line 1515)
# and add try: before it and except: after the closing ))

for i, line in enumerate(lines):
    if 'exc = _run_replay_instrumented(' in line:
        print(f'Found _run_replay_instrumented call at line {i+1}')
        # Add try: before this line
        lines.insert(i, '    try:\n')
        # Now find the line with closing ) for this call
        # It's lines 1515-1522 (indices i to i+7)
        # Add except: after line 1522 (index i+7)
        except_line = i + 8  # After the call and its closing )
        lines.insert(except_line, '    except Exception as inner_exc:\n')
        lines.insert(except_line + 1, '        import traceback as _tb3\n')
        lines.insert(except_line + 2, '        print(f"[DEBUG] Inner exception in _run_replay_instrumented: {inner_exc}", file=sys.stderr)\n')
        lines.insert(except_line + 3, '        _tb3.print_exc(file=sys.stderr)\n')
        lines.insert(except_line + 4, '        raise\n')
        break

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done - added try/except around _run_replay_instrumented')
