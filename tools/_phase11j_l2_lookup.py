#!/usr/bin/env python3
import json, sys
paths = sys.argv[1:-1]
target = int(sys.argv[-1])
for path in paths:
    with open(path, encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            if j.get('games_id') == target:
                print(path)
                print('  class=', j.get('class'))
                print('  msg=', j.get('message'))
                break
