"""
Structural diff between two AWBW Replay Player .zip replays.

Usage:
    python tools/diff_replay_zips.py <ours.zip> <oracle.zip>
    python tools/diff_replay_zips.py <ours.zip>             # diff vs default oracle

Decompresses the gzipped replay inside each .zip, then compares:
  * Player field set and a summary of player[0]/player[1] values
    (id, users_id, team, countries_id, co_id, funds, order).
  * Buildings: terrain_id histogram and total count per turn.
  * Units: players_id distribution per turn.

Prints a concise report, flagging any divergence that would likely change how
the AWBW Replay Player renders the match.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ORACLE = Path(
    r"C:\Users\phili\AppData\Roaming\AWBWReplayPlayer\ReplayData\Replays\1630459.zip"
)


# ---------------------------------------------------------------------------
# Minimal PHP-serialized reader
# ---------------------------------------------------------------------------

class _Reader:
    __slots__ = ("s", "i")

    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0

    def peek(self, n: int = 1) -> str:
        return self.s[self.i : self.i + n]

    def eat(self, c: str) -> None:
        if self.s[self.i] != c:
            raise ValueError(f"expected {c!r} at index {self.i}, got {self.s[self.i]!r}")
        self.i += 1

    def read_until(self, c: str) -> str:
        j = self.s.index(c, self.i)
        out = self.s[self.i : j]
        self.i = j
        return out


def _parse(r: _Reader) -> Any:
    tok = r.peek()
    if tok == "N":
        r.i += 2  # N;
        return None
    if tok == "i":
        r.eat("i"); r.eat(":")
        v = r.read_until(";")
        r.i += 1
        return int(v)
    if tok == "d":
        r.eat("d"); r.eat(":")
        v = r.read_until(";")
        r.i += 1
        return float(v)
    if tok == "b":
        r.eat("b"); r.eat(":")
        v = r.read_until(";")
        r.i += 1
        return bool(int(v))
    if tok == "s":
        r.eat("s"); r.eat(":")
        n_str = r.read_until(":")
        n = int(n_str)
        r.eat(":"); r.eat('"')
        v = r.s[r.i : r.i + n]
        r.i += n
        r.eat('"'); r.eat(";")
        return v
    if tok == "a":
        r.eat("a"); r.eat(":")
        n = int(r.read_until(":"))
        r.eat(":"); r.eat("{")
        out: dict[Any, Any] = {}
        for _ in range(n):
            k = _parse(r)
            v = _parse(r)
            out[k] = v
        r.eat("}")
        return out
    if tok == "O":
        r.eat("O"); r.eat(":")
        clen = int(r.read_until(":"))
        r.eat(":"); r.eat('"')
        cname = r.s[r.i : r.i + clen]
        r.i += clen
        r.eat('"'); r.eat(":")
        n = int(r.read_until(":"))
        r.eat(":"); r.eat("{")
        fields: dict[str, Any] = {}
        for _ in range(n):
            k = _parse(r)
            v = _parse(r)
            fields[k] = v
        r.eat("}")
        fields["__class__"] = cname
        return fields
    raise ValueError(f"unknown token {tok!r} at index {r.i}")


def parse_php(s: str) -> Any:
    return _parse(_Reader(s))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_replay(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        blob = zf.read(name)
    # PHP serialize uses byte lengths for `s:` strings; UTF-8 decoding breaks indices
    # when player/map names contain non-ASCII. Latin-1 preserves byte positions 1:1.
    text = gzip.decompress(blob).decode("latin-1")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return [parse_php(ln) for ln in lines]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

_PLAYER_SUMMARY_FIELDS = (
    "id", "users_id", "team", "countries_id", "co_id", "funds", "order",
)


def summarize_player(pl: dict) -> dict[str, Any]:
    return {k: pl.get(k) for k in _PLAYER_SUMMARY_FIELDS}


def summarize_turn(game: dict, turn_idx: int) -> dict[str, Any]:
    players = game.get("players", {})
    buildings = game.get("buildings", {})
    units = game.get("units", {})

    p_sum = {}
    for k, pl in players.items():
        p_sum[k] = summarize_player(pl)

    b_hist = Counter(b.get("terrain_id") for b in buildings.values())
    u_hist = Counter(u.get("players_id") for u in units.values())

    return {
        "turn": turn_idx,
        "day": game.get("day"),
        "active_turn": game.get("turn"),
        "n_players": len(players),
        "n_buildings": len(buildings),
        "n_units": len(units),
        "players": p_sum,
        "building_terrain_hist": dict(sorted(b_hist.items(), key=lambda x: -x[1])),
        "units_players_id_hist": dict(u_hist),
    }


# ---------------------------------------------------------------------------
# Diff output
# ---------------------------------------------------------------------------

def _fmt_hist(h: dict[int, int], top: int = 12) -> str:
    items = list(h.items())[:top]
    return ", ".join(f"{k}:{v}" for k, v in items) + ("" if len(h) <= top else f"  (+{len(h)-top} more)")


def print_report(label: str, frames: list[dict]) -> None:
    print(f"\n=== {label}  ({len(frames)} turns) ===")
    if not frames:
        return
    t0 = summarize_turn(frames[0], 0)
    print(f"  turn0 day={t0['day']} active={t0['active_turn']} players={t0['n_players']} "
          f"buildings={t0['n_buildings']} units={t0['n_units']}")
    for pk, ps in t0["players"].items():
        print(f"    player[{pk}]: {ps}")
    print(f"  turn0 buildings histogram (terrain_id:count): {_fmt_hist(t0['building_terrain_hist'])}")
    print(f"  turn0 units players_id histogram: {t0['units_players_id_hist']}")

    # Mid and last turns to catch drift
    for pos, idx in (("mid", len(frames) // 2), ("last", len(frames) - 1)):
        t = summarize_turn(frames[idx], idx)
        print(f"  {pos} turn[{idx}] day={t['day']} active={t['active_turn']} "
              f"units={t['n_units']} units_players_id_hist={t['units_players_id_hist']}")


def compare(ours: list[dict], oracle: list[dict]) -> None:
    print("\n=== structural comparison ===")
    if not ours or not oracle:
        print("  [skipped] one side empty")
        return
    o0 = summarize_turn(ours[0], 0)
    r0 = summarize_turn(oracle[0], 0)

    our_pl_fields = {k for pl in o0["players"].values() for k in pl}
    ora_pl_fields = {k for pl in r0["players"].values() for k in pl}
    missing = ora_pl_fields - our_pl_fields
    extra = our_pl_fields - ora_pl_fields
    if missing:
        print(f"  [WARN] player fields present in oracle but missing in ours: {sorted(missing)}")
    if extra:
        print(f"  [info] player fields present in ours but not in oracle: {sorted(extra)}")
    if not missing and not extra:
        print("  player field sets match")

    our_cids = {o0["players"][k].get("countries_id") for k in o0["players"]}
    ora_cids = {r0["players"][k].get("countries_id") for k in r0["players"]}
    print(f"  ours countries_id set: {sorted(our_cids)}   oracle countries_id set: {sorted(ora_cids)}")

    # Buildings: diverging terrain_ids?
    our_bts = set(o0["building_terrain_hist"].keys())
    ora_bts = set(r0["building_terrain_hist"].keys())
    print(f"  ours terrain_ids in turn0 buildings: {sorted(our_bts)}")
    print(f"  oracle terrain_ids in turn0 buildings: {sorted(ora_bts)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours", type=Path, help="path to our exported .zip replay")
    ap.add_argument("oracle", type=Path, nargs="?", default=DEFAULT_ORACLE,
                    help=f"path to oracle .zip replay (default: {DEFAULT_ORACLE})")
    args = ap.parse_args()

    if not args.ours.exists():
        print(f"error: ours replay not found: {args.ours}", file=sys.stderr)
        return 2
    if not args.oracle.exists():
        print(f"error: oracle replay not found: {args.oracle}", file=sys.stderr)
        return 2

    ours = load_replay(args.ours)
    oracle = load_replay(args.oracle)

    print_report(f"OURS  {args.ours.name}", ours)
    print_report(f"ORACLE  {args.oracle.name}", oracle)
    compare(ours, oracle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
