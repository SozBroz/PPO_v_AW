#!/usr/bin/env python3
"""
Deep field-level diff of turn-0 snapshot between two AWBW replay zips.
Prints side-by-side for every scalar top-level and per-player field.
"""
import gzip, io, re, sys, zipfile
from pathlib import Path

REF  = Path(r"D:\AWBW\replays\replay_1630459_GL_STD_[T1]__Gronktastic_vs_justbored_2026-04-16.zip")
CAND = Path(r"D:\AWBW\replays\127254.zip")

def load_turn0(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            try:
                with gzip.open(io.BytesIO(raw)) as gz:
                    text = gz.read().decode("utf-8")
            except OSError:
                text = raw.decode("utf-8", errors="replace")
            if 'awbwGame' in text:
                return text.split("\n")[0]
    raise RuntimeError(f"No awbwGame entry in {path}")

def extract(line: str, key: str) -> str:
    """Pull first value after PHP s:N:"key"; """
    kpat = f's:{len(key.encode())}:"{key}";'
    idx = line.find(kpat)
    if idx < 0:
        return "<MISSING>"
    rest = line[idx + len(kpat):]
    # Return up to 60 chars of the raw PHP value
    return rest[:60].split(";")[0] + ";" if ";" in rest[:60] else rest[:60]

def count_objects(line: str, cls: str) -> int:
    return line.count(f'O:{len(cls)}:"{cls}"')

def extract_all_players(line: str) -> list[dict]:
    players = []
    pat = re.compile(r'O:10:"awbwPlayer":\d+:\{')
    for m in pat.finditer(line):
        chunk = line[m.end():]
        d = {}
        for field in ["id","users_id","countries_id","co_id","funds","co_power",
                       "co_max_power","co_max_spower","co_power_on","eliminated","order","team"]:
            fp = f's:{len(field.encode())}:"{field}";'
            fi = chunk.find(fp)
            if fi >= 0:
                rest = chunk[fi+len(fp):]
                d[field] = rest[:40].split(";")[0]
            else:
                d[field] = "<MISSING>"
        players.append(d)
    return players

ref_line  = load_turn0(REF)
cand_line = load_turn0(CAND)

TOP_FIELDS = [
    "id","maps_id","weather_type","weather_code","weather_start","win_condition",
    "active","turn","day","funds","capture_win","fog","type","boot_interval",
    "starting_funds","official","min_rating","max_rating","league","team",
    "aet_interval","use_powers",
]

print(f"{'FIELD':25s}  {'REFERENCE':35s}  {'CANDIDATE':35s}  MATCH")
print("-" * 105)
for f in TOP_FIELDS:
    rv = extract(ref_line,  f)
    cv = extract(cand_line, f)
    ok = "OK" if rv == cv else "DIFF"
    print(f"  {f:23s}  {rv:35s}  {cv:35s}  {ok}")

print()
print(f"{'OBJECT':20s}  {'REF COUNT':12s}  {'CAND COUNT':12s}")
print("-" * 50)
for cls in ["awbwPlayer","awbwBuilding","awbwUnit"]:
    rc = count_objects(ref_line, cls)
    cc = count_objects(cand_line, cls)
    ok = "OK" if rc == cc else "DIFF"
    print(f"  {cls:18s}  {rc:<12d}  {cc:<12d}  {ok}")

print()
ref_players  = extract_all_players(ref_line)
cand_players = extract_all_players(cand_line)
PLAYER_FIELDS = ["id","users_id","countries_id","co_id","funds","co_power",
                 "co_max_power","co_max_spower","co_power_on","eliminated","order","team"]

for pi in range(max(len(ref_players), len(cand_players))):
    print(f"\n--- Player {pi} ---")
    print(f"  {'FIELD':20s}  {'REFERENCE':30s}  {'CANDIDATE':30s}  MATCH")
    print("  " + "-" * 90)
    rp = ref_players[pi]  if pi < len(ref_players)  else {}
    cp = cand_players[pi] if pi < len(cand_players) else {}
    for f in PLAYER_FIELDS:
        rv = rp.get(f, "<MISSING>")
        cv = cp.get(f, "<MISSING>")
        # Allow any positive integer for co_max_power/spower
        if f in ("co_max_power","co_max_spower") and rv != "<MISSING>" and cv != "<MISSING>":
            match = "OK(~)" if rv != cv else "OK"
        else:
            match = "OK" if rv == cv else "DIFF"
        print(f"  {f:20s}  {rv:30s}  {cv:30s}  {match}")
