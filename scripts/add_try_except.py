#!/usr/bin/env python3
"""Add try/except around lines 1507-1758 in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Add try: before line 1507 (index 1506)
try_line = '    try:\n'
lines.insert(1506, try_line)

# Add except after line 1758 (index 1757) - but we need to adjust for the inserted line
except_lines = [
    '    except Exception as inner_exc:\n',
    '        import traceback as _tb2\n',
    '        print(f"[DEBUG] Inner exception: {inner_exc}", file=sys.stderr)\n',
    '        _tb2.print_exc(file=sys.stderr)\n',
    '        raise\n',
]

# Insert after line 1758 (now index 1758 because we added 1 line)
for i, line in enumerate(except_lines):
    lines.insert(1758 + i + 1, line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done - added try/except around lines 1507-1758')
