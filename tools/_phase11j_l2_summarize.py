#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
for case in d['cases']:
    print('===', case['games_id'], 'env', case.get('env_i'), 'fail at action_idx', case.get('action_idx_in_env'), '===')
    for a in case.get('envelope_action_summaries', [])[:40]:
        print(f"  [{a['i']}] kind={a['kind']} player={a['player']}")
