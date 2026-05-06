#!/usr/bin/env python3
"""Add debug print after line 1506 in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Insert after line 1506 (which is index 1505 - the empty line after return base)
# Line 1506 is empty, so insert after that
insert_idx = 1506  # 0-indexed position after line 1506
debug_line = '    print("[DEBUG] _audit_one: past inner try block", file=sys.stderr)\n'

lines.insert(insert_idx, debug_line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done - added debug print after line 1506')
