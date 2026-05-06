#!/usr/bin/env python3
"""Remove all [DEBUG] print statements from desync_audit.py and other files."""
import re

files = [
    'd:/awbw/tools/desync_audit.py',
    'd:/awbw/engine/game.py',
    'd:/awbw/engine/co.py',
]

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove all [DEBUG] print lines (handle both single and double quotes)
    pattern = r'^\s*print\(f?["\'].*?\[DEBUG\].*?file=sys\.stderr\)\s*$'
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Cleaned: {filepath}")
    else:
        print(f"No DEBUG prints found in: {filepath}")

print("Done.")
