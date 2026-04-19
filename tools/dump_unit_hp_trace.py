"""Dump HP transitions for a single unit (or all units on a tile) across a trace.

Use this to diagnose display HP glitches in the AWBW Replay Player, e.g. the
"infantry 5 bars -> 6 bars after capture" symptom on replay 166901 day 19.

For each step that touches the tracked unit, prints:
    <turn.day> <player> <action>  hp=<pre>/<post>  display=<pre>/<post>
      move_pos=...  target_pos=...

Example:
    python -m tools.dump_unit_hp_trace replays/166901.trace.json \
        --player 0 --day 19 --near 5 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.action import Action, ActionType, ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UnitType, UNIT_STATS


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_trace(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _event_to_action(entry: dict) -> Action:
    atype = ActionType[entry["type"]]
    kwargs = {"action_type": atype}
    for field in ("unit_pos", "move_pos", "target_pos", "unload_pos"):
        val = entry.get(field)
        if val is not None:
            kwargs[field] = tuple(val)
    if entry.get("unit_type") is not None:
        kwargs["unit_type"] = UnitType[entry["unit_type"]]
    return Action(**kwargs)


def _display(hp: int) -> int:
    return (max(hp, 0) + 9) // 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path, help="replays/<id>.trace.json")
    ap.add_argument("--map-pool", type=Path,
                    default=REPO_ROOT / "data" / "gl_map_pool.json")
    ap.add_argument("--maps-dir", type=Path,
                    default=REPO_ROOT / "data" / "maps")
    ap.add_argument("--player", type=int, default=None,
                    help="Only dump events for this player (0 or 1)")
    ap.add_argument("--day", type=int, default=None,
                    help="Only dump events on this day")
    ap.add_argument("--unit-id", type=int, default=None,
                    help="Track a single unit by its engine unit_id")
    ap.add_argument("--near", nargs=2, type=int, metavar=("ROW", "COL"),
                    default=None,
                    help="Track any unit whose unit_pos or move_pos touches this tile")
    args = ap.parse_args()

    record = _load_trace(args.trace)
    map_data = load_map(record["map_id"], args.map_pool, args.maps_dir)
    state = make_initial_state(map_data, record["co0"], record["co1"],
                               tier_name=record.get("tier", "T2"))

    day = 1
    turn_pid = 0
    near = tuple(args.near) if args.near else None

    def _match_tile(pos) -> bool:
        return near is not None and pos is not None and tuple(pos) == near

    def _match(entry: dict) -> bool:
        if args.player is not None and turn_pid != args.player:
            return False
        if args.day is not None and day != args.day:
            return False
        if args.unit_id is not None:
            # Resolve unit at unit_pos pre-step; match on unit_id.
            pos = entry.get("unit_pos")
            if pos is None:
                return False
            u = state.get_unit_at(*pos)
            return u is not None and u.unit_id == args.unit_id
        if near is not None:
            return any(_match_tile(entry.get(k))
                       for k in ("unit_pos", "move_pos", "target_pos", "unload_pos"))
        return True

    print(f"# trace={args.trace.name}  map={record['map_id']}  co0={record['co0']} co1={record['co1']}")
    print("# day.player  action       unit_pos -> move_pos   hp=pre/post  display=pre/post   extra")

    for entry in record["full_trace"]:
        atype_name = entry["type"]

        # Capture pre-step HP context for any unit on the relevant tiles.
        pre_units: dict[tuple[int, int], tuple[int, int]] = {}
        for key in ("unit_pos", "target_pos"):
            pos = entry.get(key)
            if pos is None:
                continue
            u = state.get_unit_at(*pos)
            if u is not None:
                pre_units[tuple(pos)] = (u.unit_id, u.hp)

        try:
            state.step(_event_to_action(entry))
        except Exception as exc:
            # Show where the replay diverges but keep walking the trace.
            print(f"! step failed at day {day} pid {turn_pid} on {atype_name}: {exc}")

        if atype_name == "END_TURN":
            turn_pid = 1 - turn_pid
            if turn_pid == 0:
                day += 1
            continue

        if not _match(entry):
            continue

        # Post-step HP lookup: unit may have moved so re-find at move_pos.
        post_pos = entry.get("move_pos") or entry.get("unit_pos")
        post_unit = state.get_unit_at(*post_pos) if post_pos is not None else None
        post_hp = post_unit.hp if post_unit is not None else None
        post_id = post_unit.unit_id if post_unit is not None else None

        src_pos = entry.get("unit_pos")
        pre_hp = pre_units.get(tuple(src_pos), (None, None))[1] if src_pos else None
        pre_id = pre_units.get(tuple(src_pos), (None, None))[0] if src_pos else None

        pre_d = _display(pre_hp) if pre_hp is not None else "-"
        post_d = _display(post_hp) if post_hp is not None else "-"
        unit_id = post_id if post_id is not None else pre_id

        extra = []
        if entry.get("target_pos"):
            tpos = tuple(entry["target_pos"])
            tgt_pre = pre_units.get(tpos)
            tgt_post = state.get_unit_at(*tpos)
            if tgt_pre is not None or tgt_post is not None:
                pre_thp = tgt_pre[1] if tgt_pre is not None else None
                post_thp = tgt_post.hp if tgt_post is not None else None
                extra.append(f"target_hp={pre_thp}->{post_thp}")

        print(
            f"d{day:>2}.p{turn_pid}  {atype_name:<9s}  "
            f"{src_pos} -> {entry.get('move_pos')}   "
            f"id={unit_id}  hp={pre_hp}->{post_hp}  display={pre_d}->{post_d}   "
            f"{' '.join(extra)}"
        )


if __name__ == "__main__":
    main()
