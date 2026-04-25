#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
for c in d['cases']:
    print(c['games_id'], 'captured=', c.get('captured'),
          'completed=', c.get('completed_no_capture'),
          'env=', c.get('env_i'),
          'actions=', c.get('actions_applied'),
          'envelopes=', c.get('envelopes_applied'),
          'err=', c.get('error_msg'),
          'pyexc=', c.get('python_exception'))
