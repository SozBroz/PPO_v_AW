"""Round-trip verify a p: action stream inside a replay zip.

Parses the envelope like the AWBW viewer does (p:<pid>;d:<day>;a:a:3:{...})
and validates every embedded action JSON. Prints counts by action type.
"""
import gzip
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path


def _read(text: str, i: int):
    """Return (token, new_index). Subset of PHP: i,s,a,{,},;"""
    c = text[i]
    if c == "i":
        j = text.index(";", i)
        return ("i", int(text[i + 2 : j])), j + 1
    if c == "s":
        j = text.index(":", i + 2)
        n = int(text[i + 2 : j])
        if text[j + 1] != '"':
            raise ValueError(f"bad s: at {i}")
        val = text[j + 2 : j + 2 + n]
        return ("s", val), j + 2 + n + 2
    raise ValueError(f"unhandled token {c!r} at {i}")


def parse_envelope(line: str):
    """p:<pid>;d:<day>;a:a:3:{i:0;i:<pid>;i:1;i:<turn>;i:2;a:<N>:{i:i;s:...;...}}"""
    # p:pid;d:day;
    assert line.startswith("p:")
    semi = line.index(";")
    pid = int(line[2:semi])
    i = semi + 1
    assert line[i : i + 2] == "d:"
    j = line.index(";", i)
    day = int(line[i + 2 : j])
    i = j + 1
    assert line[i : i + len("a:a:3:{")] == "a:a:3:{"
    i += len("a:a:3:{")
    # skip i:0;i:<pid>;i:1;i:<turnN>;i:2;
    for _ in range(5):
        _, i = _read(line, i)
    # then a:<N>:{ ... }
    assert line[i] == "a"
    j = line.index(":", i + 2)
    n = int(line[i + 2 : j])
    assert line[j + 1] == "{"
    i = j + 2
    actions = []
    for _ in range(n):
        idx, i = _read(line, i)   # i:<k>;
        tok, i = _read(line, i)   # s:<len>:"...";
        if tok[0] == "s":
            actions.append(tok[1])
    return pid, day, actions


def main(zip_path: Path) -> int:
    game_id = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        action_entry = f"a{game_id}"
        if action_entry not in names:
            print(f"[FAIL] zip missing action entry {action_entry!r}. entries={names}")
            return 2
        blob = zf.read(action_entry)

    text = gzip.decompress(blob).decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"envelopes: {len(lines)}")

    counts: Counter = Counter()
    bad = 0
    for lineno, ln in enumerate(lines):
        try:
            pid, day, actions = parse_envelope(ln)
        except Exception as e:
            print(f"[BAD ENV] line {lineno}: {e}")
            bad += 1
            if bad >= 5:
                return 3
            continue

        for i, ajs in enumerate(actions):
            try:
                obj = json.loads(ajs)
            except Exception as e:
                print(f"[BAD JSON] env {lineno} action {i}: {e}")
                bad += 1
                continue
            counts[obj.get("action", "<no action>")] += 1

    print(f"action counts: {dict(counts)}")
    if bad:
        print(f"[FAIL] {bad} malformed entries")
        return 1

    # Sanity: last envelope should end in End action
    _, _, last_actions = parse_envelope(lines[-1])
    last_obj = json.loads(last_actions[-1]) if last_actions else {}
    print(f"final envelope last action: {last_obj.get('action')}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
