#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
for case in d['cases']:
    print('===', case['games_id'], 'env', case.get('env_i'), '===')
    for de in case.get('envelope_delete_actions_full', []):
        print(f"  [{de['i']}] DELETE OBJ:")
        print(json.dumps(de['obj'], indent=4))
