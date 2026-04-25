import json
with open('data/co_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
cos = data.get('cos', data)
for cid in sorted([int(k) for k in cos.keys()]):
    name = cos[str(cid)].get('name', '?')
    print(f"  engine id {cid:3} = {name}")
