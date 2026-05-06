#!/usr/bin/env python3
"""Add try/except around lines 1507-1758 in desync_audit.py"""
with open('d:/awbw/tools/desync_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Add try: before line 1507 (index 1506)
try_line = '    try:\n'
lines.insert(1506, try_line)

# Add except: after line 1758 (index 1757) - but we need to account for the inserted line
# Original line 1506 (empty) becomes index 1506
# Original line 1507 (progress = ...) becomes index 1508 (because we inserted 1 line)
# So original line 1758 (the outer except) becomes index 1759
# We want to add except: BEFORE that, at index 1758

except_lines = [
    '    except Exception as inner_exc:\n',
    '        import traceback as _tb4\n',
    '        print(f"[DEBUG] Inner exception: {inner_exc}", file=sys.stderr)\n',
    '        _tb4.print_exc(file=sys.stderr)\n',
    '        raise\n',
]

# Insert after line 1757 (original line 1757, now index 1758 because of inserted line)
for i, line in enumerate(except_lines):
    lines.insert(1758 + i, line)

with open('d:/awbw/tools/desync_audit.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done - added try/except around lines 1507-1758')
