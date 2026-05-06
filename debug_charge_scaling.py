"""
Debug: Check what PHP co_power values look like vs engine power_bar.
Read a few state_mismatch rows to understand the correct scaling.
"""
import json

# Read the state_mismatch_multi file
with open('d:/awbw/logs/desync_register_state_mismatch_multi.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Look at first few rows with charge_mismatch
for l in lines[:5]:
    row = json.loads(l)
    msg = row.get('message', '')
    if 'charge_mismatch' in msg:
        print(f"\ngames_id={row['games_id']}")
        print(f"message: {msg}")
        sm = row.get('state_mismatch', {}).get('diff_summary', {})
        print(f"diff_summary: {json.dumps(sm, indent=2)}")
        
        # Extract PHP and engine star values
        import re
        matches = re.findall(r'P(\d+) charge_mismatch: php=([\d.]+) stars \(raw=(\d+)\) engine=([\d.]+) stars \(power_bar=(\d+)\)', msg)
        for m in matches:
            p, php_s, php_raw, eng_s, eng_bar = m
            print(f"  P{p}: php_raw={php_raw}, php_stars={php_s}, eng_bar={eng_bar}, eng_stars={eng_s}")
            # What if we divide by 9000?
            print(f"    php_raw/9000 = {int(php_raw)/9000:.2f} stars")
            print(f"    php_raw/1000 = {int(php_raw)/1000:.2f} stars")
            print(f"    eng_bar/9000 = {int(eng_bar)/9000:.2f} stars")
