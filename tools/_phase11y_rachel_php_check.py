"""Phase 11Y-RACHEL-IMPL — PHP snapshot cross-check for Rachel +1 HP repair.

Walks PHP turn snapshots from a list of Rachel zips and, for each Rachel-owned
unit that was on a Rachel-owned property at the END of the opponent's turn AND
remained on the same tile at the END of Rachel's NEXT turn (i.e. did not move
or fight), computes the displayed-HP delta. The delta is the property-day
heal as the PHP oracle records it.

Expected per AWBW chart + AWBW Fandom Wiki Rachel page:
  delta == +3 visual bars (or capped at HP 10), unless the unit was already
  at 10 (delta == 0).

If the delta is consistently +2 (standard) instead of +3, the wiki claim is
WRONG and the engine fix must be ROLLED BACK (Kindle precedent).

Usage:
    python tools/_phase11y_rachel_php_check.py 1622501 1623772 1625211
"""
from __future__ import annotations

import math
import sys
import zipfile
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reuse the parser inside diff_replay_zips.
from tools.diff_replay_zips import load_replay  # noqa: E402
from engine.terrain import get_country, is_property  # noqa: E402


PROPERTY_TERRAIN_NAMES = {
    "City", "Base", "Airport", "Port", "HQ",
}


def _bars(hp_field: Any) -> int:
    """Convert PHP ``hit_points`` (float on 1..10 scale, internal/10) to displayed bars (1..10)."""
    if hp_field is None:
        return 0
    return int(math.ceil(float(hp_field)))


def _is_property_terrain_id(terrain_id: int, buildings: dict[str, Any]) -> bool:
    """Quick check: tile id appears in the buildings dict means it's a property."""
    # buildings is a flat dict keyed by building id, each entry has 'terrain_id'
    # plus row/col coords. Easier path: look up by tile coords (see below).
    return False  # unused — we use coord lookup instead


def _build_prop_index(buildings: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """(row, col) -> building dict, for the current frame."""
    out = {}
    for _k, b in buildings.items():
        if not isinstance(b, dict):
            continue
        try:
            row = int(b["y"]) if "y" in b else int(b["buildings_y"])
            col = int(b["x"]) if "x" in b else int(b["buildings_x"])
        except (KeyError, ValueError, TypeError):
            continue
        out[(row, col)] = b
    return out


def _unit_pos(u: dict[str, Any]) -> tuple[int, int] | None:
    """Extract (row, col) from a PHP unit dict — fields vary between zips."""
    for ry, cx in (("y", "x"), ("units_y", "units_x")):
        if ry in u and cx in u:
            try:
                return (int(u[ry]), int(u[cx]))
            except (ValueError, TypeError):
                continue
    return None


def _unit_owner(u: dict[str, Any]) -> int | None:
    for k in ("players_id", "units_players_id"):
        if k in u:
            try:
                return int(u[k])
            except (ValueError, TypeError):
                continue
    return None


def _building_country(b: dict[str, Any]) -> int | None:
    """Building owner is encoded in ``terrain_id`` (country-specific tile ids).
    Returns the engine country id of the owner, or None for neutral.
    """
    tid = b.get("terrain_id")
    if tid is None:
        return None
    try:
        return get_country(int(tid))
    except Exception:
        return None


def _building_is_property(b: dict[str, Any]) -> bool:
    tid = b.get("terrain_id")
    if tid is None:
        return False
    try:
        return is_property(int(tid))
    except Exception:
        return False


def _frame_active_pid(frame: dict[str, Any]) -> int | None:
    """``frame['turn']`` is the awbwPlayer id whose turn this frame represents."""
    t = frame.get("turn")
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def _rachel_player(frame: dict[str, Any]) -> tuple[int, int] | None:
    """Returns (rachel_player_id, rachel_countries_id) or None."""
    return _player_with_co(frame, 28)


def _player_with_co(frame: dict[str, Any], co_id: int) -> tuple[int, int] | None:
    """Returns (player_id, countries_id) for the player whose ``co_id`` matches."""
    players = frame.get("players") or {}
    for _k, pl in players.items():
        if isinstance(pl, dict) and int(pl.get("co_id", -1)) == co_id:
            try:
                return int(pl["id"]), int(pl["countries_id"])
            except (KeyError, ValueError, TypeError):
                continue
    return None


def inspect_zip(zip_path: Path, co_id: int = 28) -> dict[str, Any]:
    frames = load_replay(zip_path)
    if not frames:
        return {"zip": str(zip_path), "error": "no frames"}

    rp = _player_with_co(frames[0], co_id)
    if rp is None:
        return {"zip": str(zip_path), "error": f"co_id {co_id} not in frame 0"}
    rachel_id, _rachel_php_country = rp

    # Build per-frame indexes once.
    per_frame: list[dict[str, Any]] = []
    for fr in frames:
        units = fr.get("units") or {}
        props = _build_prop_index(fr.get("buildings") or {})
        active_pid = _frame_active_pid(fr)
        # Index Rachel-owned units by units_id with their pos and HP bars.
        rachel_units: dict[int, dict[str, Any]] = {}
        for _k, u in units.items():
            if not isinstance(u, dict):
                continue
            if _unit_owner(u) != rachel_id:
                continue
            pos = _unit_pos(u)
            if pos is None:
                continue
            try:
                uid_raw = u.get("id") if u.get("id") is not None else u.get("units_id")
                uid = int(uid_raw)
            except (TypeError, ValueError):
                continue
            rachel_units[uid] = {
                "pos": pos,
                "bars": _bars(u.get("hit_points")),
                "name": u.get("name") or u.get("units_name"),
            }
        per_frame.append({
            "active_pid": active_pid,
            "day": fr.get("day"),
            "rachel_units": rachel_units,
            "props": props,
        })

    # Find consecutive frame pairs where Rachel's turn just started.
    # A frame K with active_pid == rachel_id, after a frame K-1 with
    # active_pid != rachel_id (opponent end-of-turn).
    deltas: list[dict[str, Any]] = []
    for k in range(1, len(per_frame)):
        prev = per_frame[k - 1]
        curr = per_frame[k]
        # Detect rachel turn-start boundary
        if curr["active_pid"] != rachel_id:
            continue
        if prev["active_pid"] == rachel_id:
            continue
        # For each Rachel unit present in BOTH frames at the SAME tile, on a
        # Rachel-owned property in the current frame, record the bar delta.
        for uid, cu in curr["rachel_units"].items():
            pu = prev["rachel_units"].get(uid)
            if pu is None:
                continue
            if pu["pos"] != cu["pos"]:
                continue  # moved — heal not isolatable
            prop = curr["props"].get(cu["pos"])
            if prop is None or not _building_is_property(prop):
                continue  # not standing on any property tile
            # PHP only heals on player-OWNED property; if a heal happens we
            # know it's Rachel-owned. A 0 delta on neutral / enemy property
            # is also valid evidence (no over-heal). We log both.
            delta = cu["bars"] - pu["bars"]
            tile_country = _building_country(prop)
            deltas.append({
                "day": curr["day"],
                "uid": uid,
                "name": cu["name"],
                "pos": cu["pos"],
                "pre_bars": pu["bars"],
                "post_bars": cu["bars"],
                "delta": delta,
                "tile_terrain_country": tile_country,  # None = neutral / unknown
            })

    summary: dict[int, int] = defaultdict(int)
    summary_lt_max: dict[int, int] = defaultdict(int)  # only units with pre_bars < 10
    for d in deltas:
        summary[d["delta"]] += 1
        if d["pre_bars"] < 10:
            summary_lt_max[d["delta"]] += 1

    interesting = [
        d for d in deltas
        if d["pre_bars"] < 10 and d["delta"] > 0
    ]

    return {
        "zip": zip_path.name,
        "rachel_player_id": rachel_id,
        "turn_boundaries": sum(
            1 for k in range(1, len(per_frame))
            if per_frame[k]["active_pid"] == rachel_id
            and per_frame[k - 1]["active_pid"] != rachel_id
        ),
        "stationary_rachel_units_on_property": len(deltas),
        "delta_histogram_all": dict(sorted(summary.items())),
        "delta_histogram_pre_lt_max": dict(sorted(summary_lt_max.items())),
        "n_positive_heal_events_pre_lt_max": len(interesting),
        "samples": interesting[:8],
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: _phase11y_rachel_php_check.py [--co=N] <gid> [<gid> ...]")
        return 2
    co_id = 28
    rest: list[str] = []
    for a in argv:
        if a.startswith("--co="):
            co_id = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    for g in rest:
        path = REPO / "replays" / "amarriner_gl" / f"{g}.zip"
        if not path.exists():
            print(f"[{g}] missing zip: {path}")
            continue
        rep = inspect_zip(path, co_id=co_id)
        print(f"\n=== {g} ===")
        for k, v in rep.items():
            if k == "samples":
                print(f"  samples ({len(v)} shown):")
                for s in v:
                    print(f"    day={s['day']} unit={s['name']} uid={s['uid']} "
                          f"pos={s['pos']} bars {s['pre_bars']}->{s['post_bars']} "
                          f"delta={s['delta']:+d}")
            else:
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
