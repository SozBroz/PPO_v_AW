import json
d = json.load(open('logs/phase11j_funds_drill.json', encoding='utf-8'))
for case in d['cases']:
    gid = case['gid']
    print(f"\n--- gid {gid} matchup={case['matchup']} co_p0={case['co_p0']} co_p1={case['co_p1']} map={case['map_id']} ---")
    print(f"  awbw_to_engine={case['awbw_to_engine']}")
    first_drift = None
    for r in case['per_envelope']:
        delta = r.get('delta_engine_minus_php')
        if delta and any(v != 0 for v in delta.values()) and first_drift is None:
            first_drift = r
            print(f"  FIRST DRIFT: env={r['env_i']} day={r['day']} pid={r['pid']} delta={delta}")
            print(f"    engine_funds={r['engine_funds']} php_funds={r['php_funds']}")
            print(f"    engine_props={r['engine_props']}")
            break
    for r in case['per_envelope']:
        if 'fail_msg' in r:
            print(f"  FAIL: env={r['env_i']} day={r['day']} action_idx={r['fail_at_action_idx']} kind={r['fail_action_kind']}")
            print(f"    msg={r['fail_msg']}")
            print(f"    engine_funds={r['engine_funds']}")
            print(f"    php_funds_pre_env={r.get('php_funds_pre_env')} php_funds_post_env={r.get('php_funds_post_env')}")
