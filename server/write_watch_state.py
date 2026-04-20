"""
Utility to write watch_state.json from a live GameState object.
Import and call write_watch_state(state, map_name, last_action) after
each action during training or a watch-only game.

Also exposes small pure serializers (`units_list`, `properties_list`,
`board_dict`) reused by replay frame logging in `rl/env.py` so the
watch viewer and the replay viewer consume the same shape.
"""
import json
import time
from pathlib import Path
from typing import Optional

from rl.paths import WATCH_STATE_PATH, ensure_logs_dir


def units_list(state) -> list[dict]:
    """Serialize all alive units to the shape expected by board.js."""
    out: list[dict] = []
    for player in (0, 1):
        for unit in state.units.get(player, []):
            out.append({
                "player":  player,
                "type":    unit.unit_type.name,
                "type_id": int(unit.unit_type),
                "row":     unit.pos[0],
                "col":     unit.pos[1],
                "hp":      unit.hp,
                "moved":   unit.moved,
            })
    return out


def properties_list(state) -> list[dict]:
    """Serialize all properties to the shape expected by board.js."""
    out: list[dict] = []
    for prop in state.properties:
        out.append({
            "row":            prop.row,
            "col":            prop.col,
            "owner":          prop.owner,
            "is_hq":          prop.is_hq,
            "is_lab":         getattr(prop, "is_lab", False),
            "is_base":        getattr(prop, "is_base", False),
            "capture_points": getattr(prop, "capture_points", 20),
        })
    return out


def board_dict(state, *, include_terrain: bool = True) -> dict:
    """
    Build the `board` payload consumed by `server/static/board.js`.

    When `include_terrain=False` the static `height`, `width`, `terrain`
    fields are omitted so the caller can store them once and keep
    per-frame entries small. The client merges them back in.
    """
    out: dict = {
        "units":      units_list(state),
        "properties": properties_list(state),
    }
    if include_terrain:
        out["height"]  = state.map_data.height
        out["width"]   = state.map_data.width
        out["terrain"] = state.map_data.terrain
    return out


def write_watch_state(
    state,
    map_name: str = "",
    last_action: Optional[dict] = None,
) -> None:
    """
    Serialise current game state and write it atomically to
    logs/watch_state.json so the live viewer can pick it up.

    Args:
        state:       GameState object from engine (duck-typed, see below).
        map_name:    Human-readable map name (optional override).
        last_action: Dict describing the last action taken, e.g.
                     {"type": "move_attack", "from": [r,c], "to": [r,c], "target": [r,c]}
    """
    def co_dict(co):
        return {
            "id":          getattr(co, "co_id",      0),
            "name":        getattr(co, "name",        "?"),
            "cop_active":  getattr(co, "cop_active",  False),
            "scop_active": getattr(co, "scop_active", False),
            "power_bar":   getattr(co, "power_bar",   0),
            "cop_stars":   getattr(co, "cop_stars",   3),
            "scop_stars":  getattr(co, "scop_stars",  6),
        }

    watch_data = {
        "turn":          state.turn,
        "active_player": state.active_player,
        "funds":         list(state.funds),
        "co_p0":         co_dict(state.co_states[0]),
        "co_p1":         co_dict(state.co_states[1]),
        "map_name":      map_name or getattr(getattr(state, "map_data", None), "name", ""),
        "map_id":        getattr(getattr(state, "map_data", None), "map_id", None),
        "tier":          getattr(state, "tier_name", None),
        "done":          state.done,
        "winner":        state.winner,
        "board":         board_dict(state),
        "last_action":   last_action,
        "updated_at":    time.time(),
    }

    # Atomic write: write to a temp file then rename to avoid partial reads
    ensure_logs_dir()
    tmp_path = WATCH_STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(watch_data, f)
    tmp_path.replace(WATCH_STATE_PATH)
