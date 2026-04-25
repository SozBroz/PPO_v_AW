#!/usr/bin/env python3
import json, sys
def summary(path):
    ok=eg=bug=0
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            cl = j.get('class')
            if cl == 'ok': ok += 1
            elif cl == 'oracle_gap': eg += 1
            elif cl == 'engine_bug': bug += 1
            rows.append(j)
    return ok, eg, bug, rows

if __name__ == '__main__':
    a_path = sys.argv[1]
    b_path = sys.argv[2] if len(sys.argv) > 2 else None
    okA, egA, bugA, rowsA = summary(a_path)
    print(f'{a_path}: ok={okA} oracle_gap={egA} engine_bug={bugA} total={len(rowsA)}')
    if b_path is not None:
        okB, egB, bugB, rowsB = summary(b_path)
        print(f'{b_path}: ok={okB} oracle_gap={egB} engine_bug={bugB} total={len(rowsB)}')
        a_by_gid = {r['games_id']: r for r in rowsA}
        b_by_gid = {r['games_id']: r for r in rowsB}
        common = set(a_by_gid) & set(b_by_gid)
        flips_ok_to_gap = []
        flips_ok_to_bug = []
        flips_gap_to_bug = []
        flips_gap_to_ok = []
        flips_bug_to_ok = []
        flips_bug_to_gap = []
        for g in common:
            a = a_by_gid[g]['class']; b = b_by_gid[g]['class']
            if a == 'ok' and b == 'oracle_gap': flips_ok_to_gap.append(g)
            elif a == 'ok' and b == 'engine_bug': flips_ok_to_bug.append(g)
            elif a == 'oracle_gap' and b == 'engine_bug': flips_gap_to_bug.append(g)
            elif a == 'oracle_gap' and b == 'ok': flips_gap_to_ok.append(g)
            elif a == 'engine_bug' and b == 'ok': flips_bug_to_ok.append(g)
            elif a == 'engine_bug' and b == 'oracle_gap': flips_bug_to_gap.append(g)
        print(f'A->B flips common gids={len(common)}:')
        print(f'  ok -> oracle_gap   (regress): {len(flips_ok_to_gap)}  {flips_ok_to_gap[:20]}')
        print(f'  ok -> engine_bug   (regress): {len(flips_ok_to_bug)}  {flips_ok_to_bug[:20]}')
        print(f'  oracle_gap -> engine_bug     : {len(flips_gap_to_bug)}  {flips_gap_to_bug[:20]}')
        print(f'  oracle_gap -> ok   (progress): {len(flips_gap_to_ok)}  {flips_gap_to_ok[:20]}')
        print(f'  engine_bug -> ok   (progress): {len(flips_bug_to_ok)}  {flips_bug_to_ok[:20]}')
        print(f'  engine_bug -> oracle_gap     : {len(flips_bug_to_gap)}  {flips_bug_to_gap[:20]}')
