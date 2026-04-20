"""
Compare **AWBW Replay Player reference state** (PHP ``awbwGame`` lines) to ``GameState``.

**Reference side (what we diff against):** each frame is one gzipped line of
PHP-serialized ``awbwGame`` from the site ``.zip`` — the same payload the
open-source **C# AWBW Replay Player** loads and renders
(`github.com/DeamonHunter/AWBW-Replay-Player`). We parse those bytes with
``tools.diff_replay_zips.load_replay``; we do not run the desktop app in this
harness, but we are **not** comparing to the in-repo Flask ``/replay`` JSONL
viewer (that path is engine-only).

Site zips use one of two shapes (both are valid AWBW exports — not gameplay bugs):

- **Trailing snapshot** — ``N+1`` gzip lines for ``N`` ``p:`` envelopes: ``frame[0]``
  is the opening state; after envelope ``i``, compare engine to ``frame[i+1]``
  (the last line is the final board).
- **Tight** — ``N`` gzip lines for ``N`` envelopes: same pairing for ``i = 0 .. N-2``;
  there is **no** extra line after the last envelope (common when the match ends on
  that half-turn), so we do not snapshot-compare after the final envelope.

**Under test:** our engine after ``tools.oracle_zip_replay.apply_oracle_action_json``.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

from engine.game import GameState
from engine.unit import UNIT_STATS

PairingMode = Literal["trailing", "tight"]


def replay_snapshot_pairing(n_frames: int, n_envelopes: int) -> Optional[PairingMode]:
    """
    Return how PHP lines pair with ``p:`` envelopes, or ``None`` if unsupported.

    Both supported modes use the same step loop: after applying envelope ``i``,
    compare the engine to ``frame[i+1]`` **when** ``i + 1 < n_frames`` (tight mode
    skips only the comparison after the last envelope).
    """
    if n_frames <= 0 or n_envelopes < 0:
        return None
    if n_frames == n_envelopes + 1:
        return "trailing"
    if n_frames == n_envelopes:
        return "tight"
    return None


def frames_envelopes_aligned(n_frames: int, n_envelopes: int) -> bool:
    """True if ``replay_snapshot_pairing`` accepts this pair (trailing or tight)."""
    return replay_snapshot_pairing(n_frames, n_envelopes) is not None


def _php_unit_bars(u: dict[str, Any]) -> int:
    """
    AWBW snapshot ``hit_points`` is the **internal HP / 10** as a float
    (e.g. ``6.3`` = 63 internal HP). The displayed bar is the **ceiling** of
    that value, matching ``engine.unit.Unit.display_hp`` (``(hp + 9) // 10``).

    Using ``round`` here produced spurious bar mismatches against the engine
    whenever PHP stored a non-integer ``hit_points`` whose rounded value
    differed from its ceiling (e.g. PHP ``6.3`` → ``round`` 6 vs engine ceil
    7). Both sides should use ceiling so only true internal-HP drift surfaces.
    """
    hp = u.get("hit_points")
    if hp is None:
        return 0
    v = int(math.ceil(float(hp)))
    return max(0, min(10, v))


def compare_funds(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    out: list[str] = []
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        pid = int(pl["id"])
        eng = awbw_to_engine[pid]
        php_f = int(pl.get("funds", 0) or 0)
        eng_f = int(state.funds[eng])
        if php_f != eng_f:
            out.append(f"P{eng} funds engine={eng_f} php_snapshot={php_f} (awbw_players_id={pid})")
    return out


def compare_units(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """
    Match alive units by **tile + owner**, not by numeric ``units_id``: AWBW stores
    database ids (e.g. ``191637002``) while the engine allocates ``unit_id`` from a
    local counter starting at 1 for ``make_initial_state``.
    """
    out: list[str] = []

    # PHP: (engine_seat, row, col) -> snapshot row.
    # AWBW exports loaded cargo with the SAME (x, y) as their carrier and
    # marks them ``carried: "Y"`` — the engine stores cargo inside
    # ``Unit.loaded_units`` (not on the tile). Filtering these here is what
    # makes "transport + loaded passenger at the same tile" stop registering
    # as a `php duplicate unit` / `unit tile set mismatch` (root cause of the
    # 1619695 APC-vs-Infantry false positive at (19, 2)).
    php_by_tile: dict[tuple[int, int, int], dict[str, Any]] = {}
    for _k, u in (php_frame.get("units") or {}).items():
        if not isinstance(u, dict):
            continue
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        col, row = int(u["x"]), int(u["y"])
        pid = int(u["players_id"])
        eng_seat = awbw_to_engine[pid]
        key = (eng_seat, row, col)
        if key in php_by_tile:
            out.append(f"php duplicate unit at P{eng_seat} (row={row},col={col})")
        php_by_tile[key] = u

    eng_by_tile: dict[tuple[int, int, int], Any] = {}
    for seat in (0, 1):
        for u in state.units[seat]:
            if u.is_alive:
                r, c = u.pos
                key = (seat, r, c)
                if key in eng_by_tile:
                    out.append(f"engine duplicate unit at P{seat} {u.pos}")
                eng_by_tile[key] = u

    if set(php_by_tile) != set(eng_by_tile):
        only_php = set(php_by_tile) - set(eng_by_tile)
        only_eng = set(eng_by_tile) - set(php_by_tile)
        if only_php or only_eng:
            out.append(
                f"unit tile set mismatch only_in_php={sorted(only_php)[:16]}"
                f"{'…' if len(only_php) > 16 else ''} only_in_engine={sorted(only_eng)[:16]}"
                f"{'…' if len(only_eng) > 16 else ''}"
            )

    for key in sorted(set(php_by_tile) & set(eng_by_tile)):
        pu, eu = php_by_tile[key], eng_by_tile[key]
        php_name = str(pu.get("name", "")).strip()
        eng_name = UNIT_STATS[eu.unit_type].name
        if php_name and eng_name != php_name:
            # AWBW sometimes omits spaces ("Md.Tank"); tolerate alias
            aliases = {"Md.Tank": "Md. Tank"}
            eng_cmp = aliases.get(eng_name, eng_name)
            php_cmp = aliases.get(php_name, php_name)
            if eng_cmp != php_cmp:
                out.append(f"at {key} type engine={eng_name!r} php={php_name!r}")
        php_bars = _php_unit_bars(pu)
        eng_bars = eu.display_hp
        if php_bars != eng_bars:
            out.append(
                f"at {key} hp_bars engine={eng_bars} (hp={eu.hp}) php_bars={php_bars} "
                f"php_id={pu.get('id')}"
            )
    return out


def compare_snapshot_to_engine(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
    *,
    check_funds: bool = True,
    check_units: bool = True,
) -> list[str]:
    """Return a list of human-readable mismatch lines (empty => match on checked axes)."""
    out: list[str] = []
    if check_funds:
        out.extend(compare_funds(php_frame, state, awbw_to_engine))
    if check_units:
        out.extend(compare_units(php_frame, state, awbw_to_engine))
    return out
