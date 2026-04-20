"""
Replay **AWBW Replay Player** ``p:`` action JSON through the engine (best-effort).

Action and snapshot layouts follow the desktop viewer / site zip contract
(`github.com/DeamonHunter/AWBW-Replay-Player`); ``load_replay`` parses the same
gzipped ``awbwGame`` lines the C# app uses for timeline state.

Designed first for zips **produced by this repo** (``write_awbw_replay_from_trace``),
where Move/Build/Fire/End shapes match ``tools/export_awbw_replay_actions.py``.
Live-site oracle zips may include extra action kinds; unmapped ones raise
``UnsupportedOracleAction``. Standalone ``Load`` / ``Unload`` / ``Supply`` / ``Repair`` /
``AttackSeam`` / ``Hide`` (Sub dive / Stealth hide → ``DIVE_HIDE``) / ``Power`` are mapped;
``Load`` uses a nested ``Move`` like ``Move`` then JOIN/LOAD/CAPTURE/WAIT.
``Unload`` uses ``transportID`` + unloaded unit tile / type (``transportID``
may match the **carrier** or the **cargo** drawable id). ``unit.global`` is
usually the empty drop tile but can match the **transport tile** while cargo is
still embarked, or be **several tiles off** the engine-legal drop; we then pick
the closest legal ``UNLOAD`` for that carrier and cargo. ``Supply`` may use
``Move: []`` with a nested ``Supply.unit.global`` int (APC id), same envelope
pattern as site GL exports. After a ``Move``, if the mover is a loaded
transport with legal ``UNLOAD``, we commit the move and stay in ``ACTION``
(AWBW records ``Unload`` separately instead of an immediate ``WAIT``). A
following ``Unload`` may still find the engine in ``MOVE`` (transport selected,
destination not chosen); we commit the same-tile move before issuing ``UNLOAD``. ``Repair`` (Black Boat) applies a nested ``Move`` then
``ActionType.REPAIR`` toward ``Repair.repaired.global`` (dict, bare int id, or
flat ``units_id``); neighbour scan uses the same ACTION anchor as
``get_legal_actions`` (``selected_move_pos``, not only ``boat.pos``).
``Power`` / ``End`` may follow a half-turn still in ``MOVE`` (destination chosen next
click) or ``ACTION`` (terminator not yet applied). ``apply_oracle_action_json`` runs
``_oracle_finish_action_if_stale`` then ``_oracle_settle_to_select_for_power`` (same-tile
commit + ``WAIT`` / ``DIVE_HIDE``) before ``ACTIVATE_*`` / ``END_TURN`` so day boundaries
match the desktop viewer.
``Fire`` / ``AttackSeam`` attacker resolution uses :func:`_resolve_fire_or_seam_attacker`
when PHP ``units_id`` / anchor tiles disagree with ``engine.Unit`` placement; GL
``combatInfoVision`` can show the striker on the **other** seat’s tile relative
to the ``p:`` envelope (``oracle_fire`` — cross-seat anchor + both-seat search).

When ``Capt`` / plain ``Move`` would only allow **ATTACK** + WAIT (adjacent enemy
before capture is legal), we pick ``ActionType.ATTACK`` at the move end tile.
If the zip switches to another unit while a transport is still in ``ACTION``
(deferred UNLOAD), we issue **WAIT** on the stuck unit first
(``_oracle_auto_wait_if_switching_unit``).

PHP snapshot loading reuses ``tools.diff_replay_zips.load_replay`` (first zip
member = gzipped turn lines). ``parse_p_envelopes_from_zip`` returns no envelopes
when the zip has no ``a<game_id>`` action gzip (ReplayVersion 1 snapshot-only).

Set ``ORACLE_STRICT_BUILD=0`` to allow silent ``Build`` no-ops when the engine
rejects ``ActionType.BUILD`` (default: strict — raises ``UnsupportedOracleAction``).
When a Build line includes AWBW-shaped ``discovered`` for **both** PHP seats,
``envelope_awbw_player_id`` matches ``units_players_id``, and ``awbw_to_engine``
has two entries, the oracle may **repair** wrong factory ownership, apply
``funds.global`` as a lower-bound hint, or nudge a friendly unmoved occupier
off the tile before issuing ``BUILD`` (site-trusted replay only).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    compute_reachable_costs,
    get_attack_targets,
    get_legal_actions,
    get_loadable_into,
    units_can_join,
)
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.diff_replay_zips import load_replay
from tools.export_awbw_replay import _AWBW_UNIT_NAMES

EngineStepHook = Optional[Callable[[GameState, Action], None]]


class UnsupportedOracleAction(ValueError):
    pass



def _engine_step(state: GameState, act: Action, hook: EngineStepHook) -> None:
    if hook is not None:
        hook(state, act)
    state.step(act)


def _oracle_fire_stance_would_stack_on_transport(
    state: GameState, mover: Unit, tile: tuple[int, int]
) -> bool:
    """True if using ``tile`` as :class:`ActionType.ATTACK` ``move_pos`` would co-place ``mover`` on a friendly transport.

    :func:`compute_reachable_costs` includes the transport hex as a walk end for
    *boarding*, but :meth:`GameState._move_unit` only sets ``mover.pos`` — it does
    not append ``mover`` to ``loaded_units``. Picking that hex as a firing stance
    duplicates drawables on one cell (GL **1624281** after nested ``Fire``).
    """
    r, c = tile
    occ = state.get_unit_at(r, c)
    if occ is None or occ is mover or int(occ.player) != int(mover.player):
        return False
    cap = UNIT_STATS[occ.unit_type].carry_capacity
    if cap <= 0 or mover.unit_type not in get_loadable_into(occ.unit_type):
        return False
    return len(occ.loaded_units) < cap


def _oracle_friendly_units_on_tile(
    state: GameState, eng: int, pos: tuple[int, int]
) -> list[Unit]:
    """Alive units owned by ``eng`` at ``pos``.

    ``GameState.get_unit_at`` returns only the first match; AWBW exports can
    correspond to multiple drawables on one tile (e.g. APC + passenger both
    listed at the transport square — **1624281**), so Move resolution must
    filter by ``units_name`` instead of trusting ``get_unit_at`` order.
    """
    r, c = pos
    return [u for u in state.units[eng] if u.is_alive and u.pos == (r, c)]


def _oracle_resolve_fire_move_pos(
    state: GameState,
    unit: Unit,
    paths: list[dict[str, Any]],
    path_end: tuple[int, int],
    target: tuple[int, int],
) -> tuple[int, int]:
    """Return a tile ``move_pos`` for :class:`ActionType.ATTACK` (firing stance).

    - **Indirect** (Artillery / Rockets / Missiles): AWBW cannot move and fire the
      same turn — only :func:`get_attack_targets` from ``unit.pos`` is valid.
    - **Direct**: Snap the JSON path tail with :func:`_nearest_reachable_along_path`
      (same as :func:`_apply_move_paths_then_terminator`), then prefer a reachable
      tile that can strike ``target``, else tiles near the snapped end.
    """
    tr, tc = target
    start = unit.pos
    stats = UNIT_STATS[unit.unit_type]

    if stats.is_indirect:
        # Indirect may not move and attack same turn — never use JSON ``path_end`` as
        # ``move_pos`` when it differs from ``start`` (would be an illegal move).
        # AWBW ``Move.paths`` on a Fire envelope can still echo a prior-turn move;
        # we ignore path tiles for ``move_pos``. Engine indirect range is Manhattan
        # only (no terrain LOS); AWBW vision / combatInfo can differ on edge cases.
        if (tr, tc) in get_attack_targets(state, unit, start):
            return start
        return start

    costs = compute_reachable_costs(state, unit)
    er, ec = _nearest_reachable_along_path(paths, costs, path_end, start)

    def attacks_from(pos: tuple[int, int]) -> bool:
        if pos not in costs:
            return False
        if _oracle_fire_stance_would_stack_on_transport(state, unit, pos):
            return False
        return (tr, tc) in get_attack_targets(state, unit, pos)

    if attacks_from((er, ec)):
        return (er, ec)
    ranked = sorted(
        costs.keys(),
        key=lambda p: (abs(p[0] - er) + abs(p[1] - ec), p[0], p[1]),
    )
    for pos in ranked:
        if attacks_from(pos):
            return pos
    # No reachable tile can hit the recorded defender with this unit in our state
    # (wrong attacker resolution or drift). Keep snapped end so ``ATTACK`` still
    # raises ``ValueError`` / engine_bug — easier to cluster than a new oracle string.
    if not _oracle_fire_stance_would_stack_on_transport(state, unit, (er, ec)):
        return (er, ec)
    for pos in (path_end, start):
        if attacks_from(pos):
            return pos
    return (er, ec)


def _oracle_finish_action_if_stale(
    state: GameState, before_engine_step: EngineStepHook
) -> None:
    """
    Site zips often **defer** ``Unload`` after a transport ``Move``, leaving
    ``action_stage == ACTION``. The next ``p:`` line may activate a **different**
    unit, but ``GameState.step`` only handles ``SELECT_UNIT`` in SELECT/MOVE — in
    ACTION it is a no-op, so the old transport stays selected and paths attach
    to the wrong unit. Finish the pending half-turn with ``WAIT`` or
    ``DIVE_HIDE`` when legal.
    """
    if state.action_stage != ActionStage.ACTION or state.selected_unit is None:
        return
    legal = get_legal_actions(state)
    mp = state.selected_move_pos
    for prefer in (ActionType.WAIT, ActionType.DIVE_HIDE):
        chosen: Optional[Action] = None
        if mp is not None:
            for a in legal:
                if a.action_type == prefer and a.move_pos == mp:
                    chosen = a
                    break
        if chosen is None:
            for a in legal:
                if a.action_type == prefer:
                    chosen = a
                    break
        if chosen is not None:
            _engine_step(state, chosen, before_engine_step)
            return
    # Site zips omit a trailing WAIT when only CAPTURE / JOIN / LOAD remains at
    # ``selected_move_pos`` (WAIT pruned from the legal mask). Leaving ACTION
    # wedged makes the next ``Move`` SELECT a no-op and paths attach to the wrong
    # unit (register 1624281 engine_illegal_move).
    if mp is not None:
        for prefer in (
            ActionType.CAPTURE,
            ActionType.JOIN,
            ActionType.LOAD,
        ):
            for a in legal:
                if a.action_type == prefer and a.move_pos == mp:
                    _engine_step(state, a, before_engine_step)
                    return


def _oracle_advance_turn_until_player(
    state: GameState,
    want_eng: int,
    before_engine_step: EngineStepHook,
    *,
    max_steps: int = 96,
) -> None:
    """
    When the next site envelope is for engine ``want_eng`` but ``active_player`` is
    still the opponent (zip omitted ``End`` lines or engine drift), step
    ``END_TURN`` so the half-turn rolls over **and ``_end_turn`` runs** —
    granting income, draining idle fuel, resupplying on properties, and
    refreshing comm-tower counts for the next player.

    AWBW lets a player end their turn even when units are unmoved; the engine
    only gates ``END_TURN`` in ``get_legal_actions`` to force RL agents to move
    every unit. In oracle replay we mirror AWBW: when ``END_TURN`` is illegal
    purely due to unmoved units, we still step it directly through the engine
    (``GameState.step`` does **not** re-check legality). The last-resort
    ``active_player`` snap in ``_oracle_ensure_envelope_seat`` would otherwise
    bypass ``_end_turn`` entirely and silently lose the next player's start-of-
    day income — the dominant cause of the ``oracle_other`` Build no-op
    "insufficient funds (need 1000$, have 0$)" cluster (game ``1618984`` and
    similar Andy/Andy mirrors where P1 only emits a single ``Capt`` envelope
    between half-turns, leaving ``has_unmoved=True``).
    """
    want = int(want_eng)
    for _ in range(max_steps):
        if int(state.active_player) == want:
            return
        _oracle_finish_action_if_stale(state, before_engine_step)
        if int(state.active_player) == want:
            return
        if state.action_stage != ActionStage.SELECT:
            return
        legal = get_legal_actions(state)
        end_turn = next((a for a in legal if a.action_type == ActionType.END_TURN), None)
        if end_turn is None:
            # Unmoved units block END_TURN in get_legal_actions but AWBW does
            # not — synthesize the action and step it so income / fuel / supply
            # still run via ``_end_turn`` (see docstring).
            end_turn = Action(ActionType.END_TURN)
        _engine_step(state, end_turn, before_engine_step)


def _oracle_ensure_envelope_seat(
    state: GameState,
    envelope_awbw_player_id: int,
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
) -> None:
    """Make ``active_player`` match the AWBW seat that owns this ``p:`` envelope.

    AWBW archives sometimes interleave both players within the same ``day`` without
    an ``End`` between envelopes. The engine is strictly alternating half-turns;
    we try ``END_TURN`` when legal, then **snap** ``active_player`` to the envelope
    seat as a last resort (oracle replay only — see desync ``oracle_turn_active_player``).
    """
    try:
        pid = int(envelope_awbw_player_id)
    except (TypeError, ValueError):
        return
    if pid not in awbw_to_engine:
        return
    want = int(awbw_to_engine[pid])
    if int(state.active_player) == want:
        return
    _oracle_finish_action_if_stale(state, before_engine_step)
    _oracle_advance_turn_until_player(state, want, before_engine_step)
    if int(state.active_player) == want:
        return
    state.active_player = want
    state.action_stage = ActionStage.SELECT
    state.selected_unit = None
    state.selected_move_pos = None


def _oracle_snap_active_player_to_engine(
    state: GameState,
    want_eng: int,
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
) -> None:
    """Replay-only: make ``active_player`` match engine seat ``want_eng``.

    AWBW ``p:`` envelopes can drift from the engine half-turn after nested
    ``WAIT`` / ``END_TURN`` sequencing (``oracle_turn_active_player``). Repair
    no-path resolution picks boats from ``state.units[eng]`` while
    ``active_player`` still pointed at the other seat — leaving ``Repair (no
    path): unit owner != active_player`` even when the envelope is correct.
    """
    if int(state.active_player) == int(want_eng):
        return
    for awbw_pid, e in awbw_to_engine.items():
        if int(e) != int(want_eng):
            continue
        _oracle_ensure_envelope_seat(
            state, int(awbw_pid), awbw_to_engine, before_engine_step
        )
        if int(state.active_player) == int(want_eng):
            return
    _oracle_advance_turn_until_player(state, want_eng, before_engine_step)
    if int(state.active_player) == int(want_eng):
        return
    su = state.selected_unit
    if su is not None and int(su.player) == int(want_eng):
        # Half-turn bookkeeping drift only — keep MOVE/ACTION selection intact
        # (Repair / Unload after a nested Move; nuclear clear would lose the boat).
        state.active_player = want_eng
        return
    state.active_player = want_eng
    state.action_stage = ActionStage.SELECT
    state.selected_unit = None
    state.selected_move_pos = None


def _resolve_active_player_for_repair(
    state: GameState,
    boat: Unit,
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
) -> None:
    """Align ``active_player`` with the acting Black Boat (see :func:`_oracle_snap_active_player_to_engine`)."""
    _oracle_snap_active_player_to_engine(
        state, int(boat.player), awbw_to_engine, before_engine_step
    )


def _oracle_capt_no_path_building_orth_coords(er: int, ec: int) -> tuple[tuple[int, int], ...]:
    return ((er - 1, ec), (er + 1, ec), (er, ec - 1), (er, ec + 1))


def _oracle_capt_no_path_outer_ring_capturers(state: GameState, er: int, ec: int) -> list[Unit]:
    """Infantry/mech orth-adjacent to any **cardinal** of the capture target (not on the tile).

    Covers the “one tile short” case: the approach orth cell may be **occupied** (enemy
    blocker or allied stack) — we still scan its four neighbors for a capturer (desync
    1627935 / 1620450). Previously we required that orth cell to be empty, which hid
    capturers on e.g. ``(10,20)`` when ``(9,20)`` was blocked.
    """
    h, w = state.map_data.height, state.map_data.width
    seen: dict[int, Unit] = {}
    for tr, tc in _oracle_capt_no_path_building_orth_coords(er, ec):
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            vr, vc = tr + dr, tc + dc
            if (vr, vc) == (er, ec):
                continue
            if not (0 <= vr < h and 0 <= vc < w):
                continue
            cand = state.get_unit_at(vr, vc)
            if cand is None or not cand.is_alive:
                continue
            if not UNIT_STATS[cand.unit_type].can_capture:
                continue
            seen[int(cand.unit_id)] = cand
    return list(seen.values())


def _oracle_capt_no_path_empty_orth_touching_unit(
    state: GameState, er: int, ec: int, u: Unit
) -> Optional[tuple[int, int]]:
    """Empty property-orth tile ``T`` with ``|u.pos - T|_1 == 1`` (one step before capture).

    AWBW ``Capt`` no-path may resolve a capturer diagonally off the building; the
    approach tile must be **terrain-legal** for ``u`` (register 1630151: pipe ESP
    at ``(0,1)`` is empty and orth to the port but impassable for infantry).
    """
    from engine.terrain import INF_PASSABLE
    from engine.weather import effective_move_cost

    h, w = state.map_data.height, state.map_data.width
    cands: list[tuple[int, int]] = []
    for tr, tc in _oracle_capt_no_path_building_orth_coords(er, ec):
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        if state.get_unit_at(tr, tc) is not None:
            continue
        if abs(int(u.pos[0]) - tr) + abs(int(u.pos[1]) - tc) != 1:
            continue
        tid = int(state.map_data.terrain[tr][tc])
        if effective_move_cost(state, u, tid) >= INF_PASSABLE:
            continue
        cands.append((tr, tc))
    if not cands:
        return None
    cands.sort()
    return cands[0]


def _capt_building_info_raw_dict(bi: Any) -> dict[str, Any]:
    """Normalize ``buildingInfo`` (unwrap nested ``\"0\"`` row when present)."""
    if not isinstance(bi, dict):
        raise UnsupportedOracleAction("Capt.buildingInfo must be a dict")
    raw: dict[str, Any] = bi
    if "buildings_y" not in raw and isinstance(raw.get("0"), dict):
        raw = raw["0"]
    return raw


def _capt_building_coords_row_col(bi: Any) -> tuple[int, int]:
    """``Capt`` / ``buildingInfo`` → engine ``(row, col)`` (``buildings_y``, ``buildings_x``).

    Some AWBW exports duplicate the full building row under the ``\"0\"`` key.
    """
    raw = _capt_building_info_raw_dict(bi)
    try:
        return int(raw["buildings_y"]), int(raw["buildings_x"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedOracleAction(
            f"Capt.buildingInfo missing buildings_y/buildings_x: {bi!r}"
        ) from exc


def _capt_building_optional_players_awbw_id(bi: Any) -> Optional[int]:
    """Optional ``buildings_players_id`` (AWBW ``players.id``) for capture disambiguation."""
    try:
        raw = _capt_building_info_raw_dict(bi)
    except UnsupportedOracleAction:
        return None
    pid = raw.get("buildings_players_id")
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _capt_building_optional_team_awbw_id(bi: Any) -> Optional[int]:
    """Optional ``buildings_team`` string (AWBW ``players.id``) when ``buildings_players_id`` is absent."""
    try:
        raw = _capt_building_info_raw_dict(bi)
    except UnsupportedOracleAction:
        return None
    t = raw.get("buildings_team")
    if t is None or (isinstance(t, str) and not str(t).strip()):
        return None
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def _oracle_capt_preferred_engine_from_building_info(
    bi: Any,
    awbw_to_engine: dict[int, int],
) -> Optional[int]:
    """Map ``buildings_players_id`` → engine seat, when present and in the replay map."""
    aid = _capt_building_optional_players_awbw_id(bi)
    if aid is None:
        return None
    return awbw_to_engine.get(aid)


def _oracle_capt_no_path_unit_eligible_for_property(
    state: GameState, u: Unit, er: int, ec: int
) -> bool:
    """Whether ``u`` is on-tile, orth, diag, or outer-ring for property ``(er, ec)``."""
    if not UNIT_STATS[u.unit_type].can_capture:
        return False
    if u.pos == (er, ec):
        return True
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if u.pos == (er + dr, ec + dc):
            return True
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        if u.pos == (er + dr, ec + dc):
            return True
    return u in _oracle_capt_no_path_outer_ring_capturers(state, er, ec)


def _oracle_capt_no_path_unit_from_envelope_hint(
    state: GameState,
    cap: dict[str, Any],
    er: int,
    ec: int,
) -> Optional[Unit]:
    """Resolve capturer from optional ``Capt.unit`` / ``newUnit`` ``global`` tile (site drift)."""
    raw_u = cap.get("unit") or cap.get("newUnit") or {}
    g: Any = None
    if isinstance(raw_u, dict):
        if "global" in raw_u:
            g = raw_u["global"]
        elif "units_y" in raw_u:
            g = raw_u
    if not isinstance(g, dict):
        return None
    try:
        yr, yc = int(g["units_y"]), int(g["units_x"])
    except (KeyError, TypeError, ValueError):
        return None
    hint = state.get_unit_at(yr, yc)
    if hint is None or not hint.is_alive:
        return None
    if not _oracle_capt_no_path_unit_eligible_for_property(state, hint, er, ec):
        return None
    return hint


def _oracle_capt_sort_pool_by_building_player_hint(
    pool: list[Unit],
    bi: Any,
    awbw_to_engine: dict[int, int],
) -> None:
    """In-place sort: prefer ``buildings_players_id`` → engine seat when mapped (multi-capturer)."""
    pref = _oracle_capt_preferred_engine_from_building_info(bi, awbw_to_engine)
    if pref is None:
        pool.sort(key=lambda x: (x.pos[0], x.pos[1], int(x.unit_id)))
        return
    pool.sort(
        key=lambda x: (
            0 if int(x.player) == pref else 1,
            x.pos[0],
            x.pos[1],
            int(x.unit_id),
        )
    )


def _capt_building_capture_progress_value(bi: Any) -> Optional[int]:
    """AWBW ``buildings_capture`` — capture points remaining on the tile (0..20 scale)."""
    try:
        raw = _capt_building_info_raw_dict(bi)
    except UnsupportedOracleAction:
        return None
    cp = raw.get("buildings_capture")
    if cp is None:
        return None
    try:
        return int(cp)
    except (TypeError, ValueError):
        return None


def _oracle_capt_no_path_can_reach_property_this_turn(
    state: GameState, u: Unit, er: int, ec: int
) -> bool:
    """Whether ``u`` can end on ``(er, ec)`` this turn (``compute_reachable_costs``)."""
    if u.pos == (er, ec):
        return True
    return (er, ec) in compute_reachable_costs(state, u)


def _oracle_capt_no_path_pick_first_reachable_pool(
    state: GameState,
    pool: list[Unit],
    bi: Any,
    awbw_to_engine: dict[int, int],
    er: int,
    ec: int,
) -> Optional[Unit]:
    """First capturer in ``pool`` that can reach ``(er, ec)`` this turn; else None."""
    pool_r = [
        x
        for x in pool
        if _oracle_capt_no_path_can_reach_property_this_turn(state, x, er, ec)
    ]
    if not pool_r:
        return None
    if len(pool_r) == 1:
        return pool_r[0]
    _oracle_capt_sort_pool_by_building_player_hint(pool_r, bi, awbw_to_engine)
    return pool_r[0]


def _oracle_capt_no_path_geom_capturer_union(
    orth_all: list[Unit],
    outer_list: list[Unit],
    diag_all: list[Unit],
) -> list[Unit]:
    """Deduplicate orth / outer / diagonal capturer pools."""
    seen: dict[int, Unit] = {}
    for u in orth_all + outer_list + diag_all:
        seen[int(u.unit_id)] = u
    return list(seen.values())


def _oracle_capt_no_path_raise_geom_unreachable(
    state: GameState,
    er: int,
    ec: int,
    geom_union: list[Unit],
) -> None:
    """Geom capturers exist but none can reach the property this turn (register 1630151)."""
    if not geom_union:
        return
    if any(
        _oracle_capt_no_path_can_reach_property_this_turn(state, u, er, ec)
        for u in geom_union
    ):
        return
    raise UnsupportedOracleAction(
        f"Capt (no path): no capturer can reach tile ({er},{ec}) this turn "
        f"[drift: geom capturer but path blocked or engine unit positions vs AWBW zip — "
        f"not fixable by oracle-only mapping]"
    )


def _oracle_capt_no_path_engine_has_capturer_near_property(
    state: GameState, er: int, ec: int
) -> bool:
    """Any alive capturer orth, diagonal, or outer-ring around ``(er, ec)``."""
    h, w = state.map_data.height, state.map_data.width
    if not (0 <= er < h and 0 <= ec < w):
        return False
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c = state.get_unit_at(er + dr, ec + dc)
        if c is not None and c.is_alive and UNIT_STATS[c.unit_type].can_capture:
            return True
    if _oracle_capt_no_path_outer_ring_capturers(state, er, ec):
        return True
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        c = state.get_unit_at(er + dr, ec + dc)
        if c is not None and c.is_alive and UNIT_STATS[c.unit_type].can_capture:
            return True
    return False


def _oracle_capt_no_path_raise_missing_capturer(
    state: GameState,
    er: int,
    ec: int,
    prop_tid: int,
) -> None:
    """``Capt (no path)`` failure: split resolver vs engine drift (see message tags)."""
    from engine.terrain import get_terrain

    is_prop = get_terrain(prop_tid).is_property
    has_near = _oracle_capt_no_path_engine_has_capturer_near_property(state, er, ec)
    if not is_prop:
        raise UnsupportedOracleAction(
            f"Capt (no path): no unit on tile ({er},{ec}) "
            f"[resolver: buildingInfo does not reference a property terrain at this tile]"
        )
    if not has_near:
        raise UnsupportedOracleAction(
            f"Capt (no path): no unit on tile ({er},{ec}) "
            f"[drift: engine has no capturer orth/ring/diag to this property — "
            f"replay unit positions vs AWBW zip mismatch; not fixable by oracle-only mapping]"
        )
    raise UnsupportedOracleAction(
        f"Capt (no path): no unit on tile ({er},{ec}) "
        f"[resolver gap: extend _oracle_capt_no_path_* — capturers exist but were not selected]"
    )


def _oracle_diagnose_build_refusal(
    state: GameState, r: int, c: int, eng: int, ut: UnitType
) -> str:
    """Human-readable reason ``_apply_build`` would no-op (mirrors ``GameState._apply_build``)."""
    from engine.action import _build_cost, get_producible_units
    from engine.terrain import get_terrain

    prop = state.get_property_at(r, c)
    if prop is None:
        return "no PropertyState at tile"
    if prop.owner is None:
        return "property owner is None (neutral tile)"
    if int(prop.owner) != int(eng):
        return f"property owner is P{prop.owner!r}, need P{eng}"
    tid = state.map_data.terrain[r][c]
    terrain = get_terrain(tid)
    if not (terrain.is_base or terrain.is_airport or terrain.is_port):
        return f"terrain {tid} ({terrain.name}) is not a production building"
    if ut not in get_producible_units(terrain, state.map_data.unit_bans):
        return f"unit {ut.name} not producible at this building type"
    cost = _build_cost(ut, state, eng, (r, c))
    if int(state.funds[eng]) < cost:
        return f"insufficient funds (need {cost}$, have {int(state.funds[eng])}$)"
    if state.get_unit_at(r, c) is not None:
        return "tile occupied"
    return "unknown (checks passed — investigate _apply_build vs oracle)"


def _oracle_assign_production_property_owner(
    state: GameState, r: int, c: int, eng: int
) -> None:
    """Set ``PropertyState.owner`` on a production tile to ``eng`` and sync terrain / comms."""
    from engine.terrain import property_terrain_id_after_owner_change

    prop = state.get_property_at(r, c)
    if prop is None:
        return
    prop.owner = int(eng)
    prop.capture_points = 20
    old_tid = state.map_data.terrain[r][c]
    new_tid = property_terrain_id_after_owner_change(
        old_tid, int(eng), state.map_data.country_to_player
    )
    if new_tid is not None:
        state.map_data.terrain[r][c] = new_tid
        prop.terrain_id = new_tid
    if prop.is_comm_tower:
        state._refresh_comm_towers()


def _oracle_snap_neutral_production_owner_for_build(
    state: GameState,
    r: int,
    c: int,
    eng: int,
    ut: UnitType,
) -> bool:
    """Assign ``prop.owner`` when the site emits ``Build`` on a neutral factory tile.

    ``PropertyState.owner`` can stay ``None`` on neutral base/air/port terrain while
    AWBW already treats the tile as the builder's production (register: Build no-op
    with ``property_owner=None``).
    """
    from engine.action import _build_cost, get_producible_units
    from engine.terrain import get_terrain

    prop = state.get_property_at(r, c)
    if prop is None or prop.owner is not None:
        return False
    tid = state.map_data.terrain[r][c]
    terrain = get_terrain(tid)
    if not (terrain.is_base or terrain.is_airport or terrain.is_port):
        return False
    if state.get_unit_at(r, c) is not None:
        return False
    cost = _build_cost(ut, state, eng, (r, c))
    if int(state.funds[eng]) < cost:
        return False
    if ut not in get_producible_units(terrain, state.map_data.unit_bans):
        return False
    _oracle_assign_production_property_owner(state, r, c, eng)
    return True


def _oracle_build_discovered_matches_awbw_player_map(
    obj: dict[str, Any], awbw_to_engine: dict[int, int]
) -> bool:
    """``discovered`` dict keys must match every PHP ``players.id`` in ``awbw_to_engine``."""
    d = obj.get("discovered")
    if not isinstance(d, dict) or not d:
        return False
    want = {str(int(k)) for k in awbw_to_engine.keys()}
    got = {str(k) for k in d.keys()}
    if want != got:
        return False
    return all(v is None for v in d.values())


def _oracle_site_trusted_build_envelope(
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    envelope_awbw_player_id: Optional[int],
    pid: int,
) -> bool:
    """AWBW export markers for a Build line we may repair (ownership / funds / blockers)."""
    if envelope_awbw_player_id is None:
        return False
    if int(envelope_awbw_player_id) != int(pid):
        return False
    if len(awbw_to_engine) < 2:
        return False
    return _oracle_build_discovered_matches_awbw_player_map(obj, awbw_to_engine)


def _oracle_optional_apply_build_funds_hint(
    state: GameState, obj: dict[str, Any], eng: int, *, min_funds: int
) -> None:
    """Bump ``state.funds[eng]`` from optional ``funds.global`` when engine is short."""
    raw = obj.get("funds")
    if not isinstance(raw, dict):
        return
    v = raw.get("global")
    if v is None:
        return
    try:
        want = int(v)
    except (TypeError, ValueError):
        return
    cur = int(state.funds[eng])
    if want >= int(min_funds) and want > cur:
        state.funds[eng] = want


def _oracle_snap_wrong_owner_production_for_trusted_site_build(
    state: GameState,
    r: int,
    c: int,
    eng: int,
    ut: UnitType,
) -> bool:
    """Force property owner to the builder when AWBW already recorded a legal Build."""
    from engine.action import _build_cost, get_producible_units
    from engine.terrain import get_terrain

    prop = state.get_property_at(r, c)
    if prop is None or prop.owner is None:
        return False
    if int(prop.owner) == int(eng):
        return False
    if int(prop.owner) not in (0, 1) or int(eng) not in (0, 1):
        return False
    tid = state.map_data.terrain[r][c]
    terrain = get_terrain(tid)
    if not (terrain.is_base or terrain.is_airport or terrain.is_port):
        return False
    if state.get_unit_at(r, c) is not None:
        return False
    cost = _build_cost(ut, state, eng, (r, c))
    if int(state.funds[eng]) < cost:
        return False
    if ut not in get_producible_units(terrain, state.map_data.unit_bans):
        return False
    _oracle_assign_production_property_owner(state, r, c, eng)
    return True


def _oracle_drift_spawn_unloaded_cargo(
    state: GameState,
    eng: int,
    cargo_ut: UnitType,
    target: tuple[int, int],
    cargo_global: dict[str, Any],
    cargo_awbw_units_id: Optional[int],
) -> bool:
    """Reconcile an AWBW ``Unload`` when the engine has lost track of the carrier/cargo.

    This is the *drift recovery path* for :func:`_resolve_unload_transport` and
    the post-transport-found "no UNLOAD legal" branch in the ``Unload``
    handler. AWBW says cargo ``cargo_ut`` is dropped at ``target`` from a
    carrier the engine cannot reconcile (engine missed an earlier ``Load``,
    the carrier ended on a different tile than AWBW, the carrier never moved,
    etc.). We try the cheapest reconciliation that keeps the engine state
    aligned with AWBW so subsequent envelopes still apply.

    Sub-cases (in priority order):

    A. **Cargo already on map by AWBW id** — same unit kept on map (engine
       missed the ``Load``). Relocate it to ``target`` (if passable + free)
       and mark ``moved=True``.
    B. **Target tile already holds a friendly cargo-type unit** — engine
       previously placed a unit there that AWBW now treats as unloaded.
       Mark it ``moved=True`` and succeed silently.
    C. **Target tile holds a friendly carrier** that can carry this cargo
       (e.g. Black Boat on a shoal AWBW counts as the drop tile). Teleport
       the carrier to a free, terrain-legal orth-adjacent tile, then spawn
       the cargo on ``target``.
    D. **Empty target + adjacent friendly carrier** that can load this
       ``cargo_ut`` (original behaviour: `get_loadable_into`).
    E. **Empty target + carrier farther away** (within Manhattan ≤ 8) that
       can load ``cargo_ut``: teleport the carrier to a free terrain-legal
       orth-adjacent tile, then spawn the cargo on ``target``.

    Returns ``True`` on a successful reconciliation (caller skips the normal
    Unload commit). Returns ``False`` to let the caller re-raise the
    original resolver exception (drift too deep for safe recovery).
    """
    from engine.weather import effective_move_cost
    from engine.terrain import INF_PASSABLE

    h, w = state.map_data.height, state.map_data.width
    tr, tc = int(target[0]), int(target[1])
    if not (0 <= tr < h and 0 <= tc < w):
        return False

    cargo_stats = UNIT_STATS[cargo_ut]

    def _passable_for(unit_type: UnitType, player: int, r: int, c: int) -> bool:
        proxy = Unit(
            unit_type=unit_type,
            player=player,
            hp=100,
            ammo=UNIT_STATS[unit_type].max_ammo,
            fuel=UNIT_STATS[unit_type].max_fuel,
            pos=(r, c),
            moved=True,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=0,
        )
        tid = state.map_data.terrain[r][c]
        return effective_move_cost(state, proxy, tid) < INF_PASSABLE

    # (A) Cargo already on the map: relocate it.
    if cargo_awbw_units_id is not None:
        existing = _unit_by_awbw_units_id(state, int(cargo_awbw_units_id))
        if existing is not None and int(existing.player) == int(eng):
            if existing.unit_type == cargo_ut:
                if existing.pos == (tr, tc):
                    existing.moved = True
                    return True
                if (
                    state.get_unit_at(tr, tc) is None
                    and _passable_for(cargo_ut, eng, tr, tc)
                ):
                    existing.pos = (tr, tc)
                    existing.moved = True
                    return True

    occupant = state.get_unit_at(tr, tc)

    # (B) Friendly cargo-type unit already sits on the drop tile.
    if (
        occupant is not None
        and int(occupant.player) == int(eng)
        and occupant.unit_type == cargo_ut
    ):
        occupant.moved = True
        return True

    # (B') Enemy occupant on the drop tile — AWBW would never Unload onto an
    # enemy-held tile, so the engine must hold a stale ghost AWBW already
    # killed (combat drift the engine missed). Zero its HP so the tile is
    # free for the drift spawn to land.
    if occupant is not None and int(occupant.player) != int(eng):
        occupant.hp = 0
        occupant = None

    # (C) Friendly carrier sitting on the drop tile (e.g. BB on shoal target).
    if (
        occupant is not None
        and int(occupant.player) == int(eng)
        and UNIT_STATS[occupant.unit_type].carry_capacity > 0
        and cargo_ut in get_loadable_into(occupant.unit_type)
    ):
        # Teleport the carrier to any free, terrain-legal orth-adjacent tile.
        teleport_to: Optional[tuple[int, int]] = None
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = tr + dr, tc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if state.get_unit_at(nr, nc) is not None:
                continue
            if not _passable_for(occupant.unit_type, eng, nr, nc):
                continue
            teleport_to = (nr, nc)
            break
        if teleport_to is None:
            return False
        occupant.pos = teleport_to
        occupant.moved = True
        # Target tile is now empty; fall through to spawn.
        occupant = None

    if occupant is not None:
        return False

    if not _passable_for(cargo_ut, eng, tr, tc):
        return False

    # (D) Adjacent friendly carrier already in place.
    has_orth_carrier = False
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = tr + dr, tc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        adj = state.get_unit_at(nr, nc)
        if adj is None or int(adj.player) != int(eng):
            continue
        adj_stats = UNIT_STATS[adj.unit_type]
        if adj_stats.carry_capacity <= 0:
            continue
        if cargo_ut not in get_loadable_into(adj.unit_type):
            continue
        has_orth_carrier = True
        break

    # (E) Carrier farther away — spawn cargo without relocating the carrier.
    # Teleporting the carrier into target adjacency was unsafe: the engine's
    # ``Move`` resolver scans position-anchored fallbacks (path start / global
    # tile) for the carrier on subsequent envelopes and breaks if we have moved
    # it to a tile AWBW never used. Keep the carrier where it is and leave the
    # cargo as a drift unit on ``target`` so AWBW timing is preserved.
    if not has_orth_carrier:
        carrier_in_range: Optional[Unit] = None
        for u in state.units[eng]:
            if not u.is_alive:
                continue
            if UNIT_STATS[u.unit_type].carry_capacity == 0:
                continue
            if cargo_ut not in get_loadable_into(u.unit_type):
                continue
            d = abs(u.pos[0] - tr) + abs(u.pos[1] - tc)
            if d > 8:
                continue
            carrier_in_range = u
            break
        if carrier_in_range is None:
            return False

    # Spawn the cargo on the drop tile (drift unit; ``moved=True`` so the AWBW
    # turn budget is respected).
    proxy_cargo = Unit(
        unit_type=cargo_ut,
        player=eng,
        hp=100,
        ammo=cargo_stats.max_ammo if cargo_stats.max_ammo > 0 else 0,
        fuel=cargo_stats.max_fuel,
        pos=(tr, tc),
        moved=True,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=state._allocate_unit_id(),
    )
    if isinstance(cargo_global, dict):
        try:
            proxy_cargo.fuel = int(cargo_global.get("units_fuel", proxy_cargo.fuel))
        except (TypeError, ValueError):
            pass
        try:
            proxy_cargo.ammo = int(cargo_global.get("units_ammo", proxy_cargo.ammo))
        except (TypeError, ValueError):
            pass
        try:
            php_hp_internal = int(cargo_global.get("units_hit_points", 10)) * 10
            proxy_cargo.hp = max(1, min(100, php_hp_internal))
        except (TypeError, ValueError):
            pass
    state.units[eng].append(proxy_cargo)
    return True


def _oracle_drift_teleport_blocker_off_build_tile(
    state: GameState,
    r: int,
    c: int,
    before_engine_step: EngineStepHook,
) -> bool:
    """Last-resort drift recovery: relocate the unit at ``(r,c)`` so a Build can land.

    Used when AWBW emits ``Build`` at a tile the engine still has occupied
    (friendly *or* enemy) and normal nudge cannot move the unit (genuinely
    trapped friendly, or enemy ghost AWBW already destroyed). Because the
    tile is AWBW's ground truth for the new factory product, the occupier is
    drift we cannot reconcile by replay.

    Strategy (least-invasive first):

    1. Teleport to any empty 8-neighbour cell whose terrain the unit can
       legally enter under current weather (uses ``effective_move_cost``).
    2. If no legal-terrain cell exists, mark the unit dead (drift ghost).
    """
    from engine.weather import effective_move_cost
    from engine.terrain import INF_PASSABLE

    u = state.get_unit_at(r, c)
    if u is None:
        return True
    h, w = state.map_data.height, state.map_data.width
    for dr, dc in (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if state.get_unit_at(nr, nc) is not None:
            continue
        tid = state.map_data.terrain[nr][nc]
        if effective_move_cost(state, u, tid) >= INF_PASSABLE:
            continue
        u.pos = (nr, nc)
        u.moved = True
        return True
    # No legal-terrain neighbour: kill the ghost so the Build can land.
    # ``Unit.is_alive`` is a derived property of ``hp``, so zeroing HP suffices.
    u.hp = 0
    return True


def _oracle_nudge_eng_occupier_off_production_build_tile(
    state: GameState,
    r: int,
    c: int,
    eng: int,
    before_engine_step: EngineStepHook,
) -> bool:
    """Move a friendly blocker off the factory tile so an AWBW Build can land.

    Two cases:

    1. **Unmoved blocker** — step it normally to a free orth neighbour
       (``SELECT_UNIT`` → MOVE → ``WAIT`` / ``DIVE_HIDE``).
    2. **Already-moved blocker** (engine drift) — AWBW must have moved this
       unit elsewhere this turn but the engine missed the move (or killed the
       wrong unit in a prior strike). Teleport it to a free orth neighbour
       in-place, preserving ``moved=True`` so it cannot act again. This is
       the *only* honest way to keep the Build envelope landing without
       fabricating a bogus move that re-touches the unit. Skip if no orth
       neighbour is free / on-map (the Build then fails loud — drift too
       deep for cosmetic correction).

    If the unmoved blocker is **trapped** (no reachable orth neighbour, e.g.
    Tank surrounded by enemies + impassable mountain + map edge), fall through
    to :func:`_oracle_drift_teleport_blocker_off_build_tile` — AWBW must have
    moved or killed this unit this turn for the Build to be legal.
    """
    if int(state.active_player) != eng:
        return False
    if state.action_stage != ActionStage.SELECT:
        return False
    u = state.get_unit_at(r, c)
    if u is None or int(u.player) != eng:
        return False
    h, w = state.map_data.height, state.map_data.width

    if u.moved:
        # Drift relocate: pick the first on-map empty orth neighbour. No
        # reachability check (unit is "spent" anyway, and AWBW has it
        # somewhere we can't reconstruct).
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if state.get_unit_at(nr, nc) is not None:
                continue
            u.pos = (nr, nc)
            return True
        return False

    reach = compute_reachable_costs(state, u)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) not in reach:
            continue
        if state.get_unit_at(nr, nc) is not None:
            continue
        su_id = int(u.unit_id)
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=(r, c),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        if state.selected_unit is not u or state.action_stage != ActionStage.MOVE:
            return False
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=(r, c),
                move_pos=(nr, nc),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        legal = get_legal_actions(state)
        chosen: Optional[Action] = None
        for a in legal:
            if a.action_type == ActionType.WAIT and a.move_pos == (nr, nc):
                chosen = a
                break
        if chosen is None:
            for a in legal:
                if a.action_type == ActionType.DIVE_HIDE and a.move_pos == (nr, nc):
                    chosen = a
                    break
        if chosen is None:
            return False
        _engine_step(state, chosen, before_engine_step)
        return True
    # Unmoved but truly trapped (e.g. surrounded + impassable). Treat as drift.
    return _oracle_drift_teleport_blocker_off_build_tile(
        state, r, c, before_engine_step
    )


def _oracle_capt_no_path_commit_pending_move(
    state: GameState, u: Unit, before_engine_step: EngineStepHook
) -> None:
    """After ``SELECT``→``MOVE`` to an adjacent tile, ``WAIT`` at ``move_pos`` lands the unit."""
    if (
        state.action_stage != ActionStage.ACTION
        or state.selected_unit is not u
        or state.selected_move_pos is None
    ):
        return
    if u.pos == state.selected_move_pos:
        return
    mp = state.selected_move_pos
    legal = get_legal_actions(state)
    for a in legal:
        if a.action_type == ActionType.WAIT and a.move_pos == mp:
            _engine_step(state, a, before_engine_step)
            return
    for a in legal:
        if a.action_type == ActionType.DIVE_HIDE and a.move_pos == mp:
            _engine_step(state, a, before_engine_step)
            return


def _global_unit(obj: dict[str, Any]) -> dict[str, Any]:
    u = obj.get("unit") or obj.get("newUnit") or {}
    if "global" in u:
        return u["global"]
    return u


def _oracle_move_paths_for_envelope(
    move: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> list[Any]:
    """``paths.global`` or the current envelope's path list (GL omits ``global``)."""
    pd = move.get("paths") or {}
    if not isinstance(pd, dict):
        return []
    g = pd.get("global")
    if isinstance(g, list) and len(g) > 0:
        return g
    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        seat = pd.get(str(pid))
        if isinstance(seat, list) and len(seat) > 0:
            return seat
        seat2 = pd.get(pid)
        if isinstance(seat2, list) and len(seat2) > 0:
            return seat2
    for _k, v in pd.items():
        if _k == "global":
            continue
        if isinstance(v, list) and len(v) > 0:
            return v
    return []


def _oracle_move_unit_global_for_envelope(
    move: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """``unit.global`` or the bucket for the ``p:`` envelope seat (same as paths)."""
    u = move.get("unit") or {}
    if not isinstance(u, dict):
        return {}
    gl = u.get("global")
    if isinstance(gl, dict) and gl.get("units_id") is not None:
        return gl
    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        seat = u.get(str(pid))
        if isinstance(seat, dict) and seat.get("units_id") is not None:
            return seat
        seat2 = u.get(pid)
        if isinstance(seat2, dict) and seat2.get("units_id") is not None:
            return seat2
    for _k, v in u.items():
        if _k == "global":
            continue
        if isinstance(v, dict) and v.get("units_id") is not None:
            return v
    return {}


def _oracle_unload_unit_global_for_envelope(
    obj: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """Cargo snapshot for standalone ``Unload``: ``unit.global`` or per-seat buckets.

    Mirrors :func:`_oracle_move_unit_global_for_envelope` / ``AttackSeam`` no-path:
    Global League exports may omit ``global`` and nest the unloaded unit under the
    ``p:`` line's AWBW ``players[].id`` (same pattern as ``Move.paths``).
    Prefer ``units_players_id`` (always present for cargo) rather than ``units_id``
    alone — PHP may omit the drawable id when it duplicates ``transportID``.
    """
    uwrap = obj.get("unit") or {}
    if not isinstance(uwrap, dict):
        return {}
    gl = uwrap.get("global")
    if isinstance(gl, dict) and gl.get("units_players_id") is not None:
        return gl
    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        seat = uwrap.get(str(pid))
        if isinstance(seat, dict) and seat.get("units_players_id") is not None:
            return seat
        seat2 = uwrap.get(pid)
        if isinstance(seat2, dict) and seat2.get("units_players_id") is not None:
            return seat2
    for _k, v in uwrap.items():
        if _k == "global":
            continue
        if isinstance(v, dict) and v.get("units_players_id") is not None:
            return v
    return {}


def _merge_move_gu_fields(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    """Fill missing AWBW unit fields from the alternate bucket (global vs per-seat)."""
    out = dict(primary)
    for k in (
        "units_name",
        "units_symbol",
        "units_y",
        "units_x",
        "units_players_id",
        "units_id",
    ):
        if out.get(k) is None and secondary.get(k) is not None:
            out[k] = secondary[k]
    return out


def _oracle_resolve_move_global_unit(
    move: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """Resolve ``Move`` ``unit.global`` for GL exports that nest under seat keys only.

    :func:`_global_unit` returns ``move[\"unit\"]`` when there is no ``\"global\"`` key,
    which is not a single unit dict (``oracle_move_no_unit`` / per-seat paths).
    Prefer :func:`_oracle_move_unit_global_for_envelope`, then merge coordinates
    from the flat global when both exist.
    """
    gu_seat = _oracle_move_unit_global_for_envelope(move, envelope_awbw_player_id)
    gu_flat = _global_unit(move)
    if not isinstance(gu_flat, dict):
        gu_flat = {}
    if gu_seat.get("units_id") is not None:
        return _merge_move_gu_fields(gu_seat, gu_flat)
    if gu_flat.get("units_id") is not None:
        return _merge_move_gu_fields(gu_flat, gu_seat)
    u = move.get("unit") or move.get("newUnit") or {}
    if isinstance(u, dict):
        for _k, v in u.items():
            if _k == "global":
                continue
            if isinstance(v, dict) and v.get("units_id") is not None:
                return _merge_move_gu_fields(v, gu_flat)
    return gu_seat if gu_seat else gu_flat


def _oracle_resolve_move_paths(
    move: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> list[Any]:
    """``paths.global`` or per-seat / first non-empty list (delegates to :func:`_oracle_move_paths_for_envelope`)."""
    return _oracle_move_paths_for_envelope(move, envelope_awbw_player_id)


def _oracle_fire_combat_info_merged(
    fire_blk: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """Merge ``combatInfoVision.global`` with per-seat data when attacker is ``?`` / non-dict."""
    civ = fire_blk.get("combatInfoVision") or {}
    if not isinstance(civ, dict):
        civ = {}
    g = ((civ.get("global") or {}).get("combatInfo") or {})
    if not isinstance(g, dict):
        g = {}
    out: dict[str, Any] = dict(g)
    seat_ci: dict[str, Any] = {}
    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        seat = civ.get(str(pid)) or civ.get(pid)
        if isinstance(seat, dict):
            ci = seat.get("combatInfo")
            if isinstance(ci, dict):
                seat_ci = ci
    for role in ("attacker", "defender"):
        cur = out.get(role)
        alt = seat_ci.get(role)
        if not isinstance(cur, dict) and isinstance(alt, dict):
            out[role] = alt
        elif isinstance(cur, dict) and isinstance(alt, dict):
            merged = dict(alt)
            merged.update(cur)
            out[role] = merged
    return out


def _oracle_set_combat_damage_override_from_combat_info(
    state: GameState,
    fire_blk: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
    attacker: Unit,
    defender_pos: tuple[int, int],
) -> None:
    """Pin ``state._oracle_combat_damage_override`` to AWBW's actual rolled damages.

    ``engine.combat.calculate_damage`` rolls ``random.randint(0, 9)`` for luck on
    every strike. AWBW logs the **post-strike** display HP for both sides in
    ``combatInfoVision[…].combatInfo.{attacker,defender}.units_hit_points``;
    those are the only ground truth for the rolled outcome. Convert display
    (1–10) to engine internal HP (×10) and compute damages by subtraction off
    the engine's pre-strike HP. Once consumed, ``_apply_attack`` clears the
    override so a later RL ``step`` rolls luck normally.

    Without this snap, every audit run produced a different first-divergence on
    the same game (combat luck cascaded into the ``Capt`` / ``Move`` / ``Fire``
    "no unit" drift cluster within ~3 turns). Skip silently when AWBW HPs are
    missing or non-numeric — the engine then rolls as before. Seam attacks
    (no defender unit) are not affected: this helper is only wired into the
    ``Fire`` paths.
    """
    fi = _oracle_fire_combat_info_merged(fire_blk, envelope_awbw_player_id)
    att_ci = fi.get("attacker") or {}
    def_ci = fi.get("defender") or {}
    defender_unit = state.get_unit_at(*defender_pos)

    def _to_internal(disp_raw: Any) -> Optional[int]:
        if disp_raw is None:
            return None
        try:
            d = int(disp_raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(100, d * 10))

    awbw_def_hp = _to_internal(def_ci.get("units_hit_points")) if isinstance(def_ci, dict) else None
    awbw_att_hp = _to_internal(att_ci.get("units_hit_points")) if isinstance(att_ci, dict) else None

    dmg: Optional[int] = None
    counter: Optional[int] = None
    if awbw_def_hp is not None and defender_unit is not None and defender_unit.is_alive:
        dmg = max(0, int(defender_unit.hp) - awbw_def_hp)
    if awbw_att_hp is not None and attacker is not None and attacker.is_alive:
        counter = max(0, int(attacker.hp) - awbw_att_hp)

    if dmg is None and counter is None:
        return
    state._oracle_combat_damage_override = (dmg, counter)


def _oracle_fire_chebyshev1_neighbours(r: int, c: int) -> list[tuple[int, int]]:
    """Eight tiles adjacent to ``(r, c)`` (Chebyshev distance 1, excluding centre)."""
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _oracle_fire_resolve_defender_target_pos(
    state: GameState,
    defender: dict[str, Any],
    *,
    attacker_eng: int,
    attacker_anchor: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    """Engine tile for the defender when ``combatInfo`` ``units_y``/``units_x`` is stale.

    Vision exports sometimes place the defender one step off the engine map (e.g.
    GL 1628008: Md.Tank strike recorded vs neighbour coordinates). Prefer
    ``units_id`` → :func:`_unit_by_awbw_units_id`, then the recorded tile, then a
    Chebyshev-1 search for an enemy matching ``units_id`` or ``units_hit_points``.

    When several foes sit on the Chebyshev-1 ring around the recorded defender
    tile (fog / stacking), prefer the one the declared attacker can actually
    strike from ``attacker_anchor`` (GL 1632283: ring tie-break picked a farther
    Chebyshev target over the orth-adjacent true defender).
    """
    dr, dc = int(defender["units_y"]), int(defender["units_x"])
    aeng = int(attacker_eng)
    hint_hp: Optional[int] = None
    if defender.get("units_hit_points") is not None:
        try:
            hint_hp = int(defender["units_hit_points"])
        except (TypeError, ValueError):
            hint_hp = None
    raw_id = defender.get("units_id")
    did: Optional[int] = None
    if raw_id is not None:
        try:
            did = int(raw_id)
        except (TypeError, ValueError):
            did = None

    def _is_foe(u: Optional[Unit]) -> bool:
        return u is not None and u.is_alive and int(u.player) != aeng

    if did is not None:
        du = _unit_by_awbw_units_id(state, did)
        if _is_foe(du):
            return du.pos  # type: ignore[union-attr]

    ring: list[tuple[int, int]] = [(dr, dc)]
    ring.extend(_oracle_fire_chebyshev1_neighbours(dr, dc))
    foes_ring: list[tuple[tuple[int, int], Unit]] = []
    for tr, tc in ring:
        if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
            continue
        uu = state.get_unit_at(tr, tc)
        if not _is_foe(uu):
            continue
        foes_ring.append(((tr, tc), uu))

    # Prefer the tile combatInfo names when it holds an enemy (register 1630151: two
    # adjacent foes; ``units_hit_points`` can match the wrong unit's display_hp).
    at_rec = state.get_unit_at(dr, dc)
    if _is_foe(at_rec):
        return (dr, dc)

    if did is not None:
        for pos, uu in foes_ring:
            if int(uu.unit_id) == int(did):
                return pos
    if hint_hp is not None:
        want = int(hint_hp)
        hp_one = [(pos, u) for pos, u in foes_ring if int(u.display_hp) == want]
        if len(hp_one) == 1:
            return hp_one[0][0]
        relaxed = [
            (pos, u) for pos, u in foes_ring if abs(int(u.display_hp) - want) <= 1
        ]
        if len(relaxed) == 1:
            return relaxed[0][0]
    if len(foes_ring) > 1 and attacker_anchor is not None:
        ar, ac = int(attacker_anchor[0]), int(attacker_anchor[1])
        u_att = state.get_unit_at(ar, ac)
        if (
            u_att is not None
            and u_att.is_alive
            and int(u_att.player) == aeng
        ):
            from engine.action import get_attack_targets

            tgt_set = get_attack_targets(state, u_att, u_att.pos)
            strikable = [pr for pr in foes_ring if pr[0] in tgt_set]
            if len(strikable) == 1:
                return strikable[0][0]
            if len(strikable) >= 2:
                foes_ring = strikable
            elif len(strikable) == 0:
                pass
    if len(foes_ring) == 1:
        return foes_ring[0][0]
    if foes_ring:
        return min(
            foes_ring,
            key=lambda pr: (abs(pr[0][0] - dr) + abs(pr[0][1] - dc), pr[0][0], pr[0][1]),
        )[0]
    if did is not None:
        for pl in (0, 1):
            if int(pl) == aeng:
                continue
            for uu in state.units[pl]:
                if not uu.is_alive or int(uu.unit_id) != int(did):
                    continue
                return uu.pos
    return (dr, dc)


def _oracle_sync_selection_for_endpoint(
    state: GameState,
    u: Unit,
    sr: int,
    sc: int,
    er: int,
    ec: int,
    before_engine_step: EngineStepHook,
) -> None:
    """Commit SELECT→MOVE so the acting unit ends at ``(er, ec)`` (Fire/Repair no-path pattern)."""
    eng = int(u.player)
    su_id = int(u.unit_id)
    if int(state.active_player) != eng:
        raise UnsupportedOracleAction(
            f"Oracle sync for engine P{eng} but active_player={state.active_player}"
        )
    if (
        state.action_stage == ActionStage.MOVE
        and state.selected_unit is not None
        and state.selected_unit.pos == (sr, sc)
        and state.selected_move_pos is None
    ):
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=(sr, sc),
                move_pos=(er, ec),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
    elif state.action_stage == ActionStage.SELECT:
        _engine_step(
            state,
            Action(ActionType.SELECT_UNIT, unit_pos=u.pos, select_unit_id=su_id),
            before_engine_step,
        )
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=u.pos,
                move_pos=(er, ec),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
    elif state.action_stage == ActionStage.ACTION:
        if state.selected_unit is not u or state.selected_move_pos != (er, ec):
            _engine_step(
                state,
                Action(ActionType.SELECT_UNIT, unit_pos=u.pos, select_unit_id=su_id),
                before_engine_step,
            )
            _engine_step(
                state,
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=u.pos,
                    move_pos=(er, ec),
                    select_unit_id=su_id,
                ),
                before_engine_step,
            )
    else:
        raise UnsupportedOracleAction(
            f"Oracle sync: unexpected stage={state.action_stage.name} "
            f"sel={state.selected_unit!s} mpos={state.selected_move_pos}"
        )


def _black_boat_oracle_action_tile(state: GameState, boat: Unit) -> tuple[int, int]:
    """Tile used for orthogonal REPAIR adjacency (matches ``_get_action_actions``).

    Neighbour scan uses the same ``move_pos`` as Black Boat ``REPAIR`` in
    :func:`engine.action._get_action_actions` — i.e. :func:`_oracle_attack_eval_pos`
    (``selected_move_pos`` after ``SELECT``→``move_pos``, not raw ``boat.pos``;
    same unit-id match as fire oracle; boarding / indirect rules apply).
    """
    return _oracle_attack_eval_pos(state, boat)


def _repair_repaired_global_dict(
    repair_block: dict[str, Any],
    envelope_awbw_player_id: Optional[int] = None,
) -> dict[str, Any]:
    """Normalize ``Repair.repaired`` / ``repaired.global`` (dict, int id, or flat).

    Global League exports sometimes omit ``repaired.global`` and nest the healed
    unit snapshot only under the ``p:`` seat key (same pattern as ``Move.unit``).
    """
    rep = repair_block.get("repaired")
    if not isinstance(rep, dict):
        return {}
    gl = rep.get("global")
    out: dict[str, Any] = {}
    if isinstance(gl, dict):
        out = dict(gl)
    elif isinstance(gl, (int, float)):
        out = {"units_id": int(gl)}
    elif isinstance(gl, str):
        try:
            out = {"units_id": int(gl)}
        except ValueError:
            out = {}
    elif "units_id" in rep:
        out = {k: v for k, v in rep.items() if k != "global"}

    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        seat = rep.get(str(pid))
        if not isinstance(seat, dict):
            seat = rep.get(pid)
        if isinstance(seat, dict) and seat.get("units_id") is not None:
            out = _merge_move_gu_fields(out, seat) if out else dict(seat)

    if not out or _oracle_awbw_scalar_int_optional(out.get("units_id")) is None:
        for _k, v in rep.items():
            if _k == "global" or not isinstance(v, dict):
                continue
            if _oracle_awbw_scalar_int_optional(v.get("units_id")) is None:
                continue
            out = _merge_move_gu_fields(v, out) if out else dict(v)
            break

    return out


def _repair_display_hp_matches_hint(display_hp: int, want: int) -> bool:
    """AWBW ``units_hit_points`` vs engine bars can differ by one after drift."""
    return display_hp == want or abs(int(display_hp) - int(want)) <= 1


def _oracle_fallback_repair_boat_and_ally(
    state: GameState,
    eng: int,
    *,
    hp_key: Optional[int] = None,
    acting_boat: Optional[Unit] = None,
) -> Optional[tuple[Unit, Unit]]:
    """Pick (acting Black Boat, ally) when PHP ``units_id`` keys do not match engine ids."""
    from engine.action import _black_boat_repair_eligible

    if acting_boat is not None:
        if (
            not acting_boat.is_alive
            or acting_boat.unit_type != UnitType.BLACK_BOAT
            or int(acting_boat.player) != int(eng)
        ):
            return None
        boats = [acting_boat]
    else:
        boats = [
            u
            for u in state.units[eng]
            if u.is_alive and u.unit_type == UnitType.BLACK_BOAT
        ]
    if not boats:
        return None

    def _rank(use_hp: bool) -> list[tuple[int, Unit, Unit]]:
        ranked: list[tuple[int, Unit, Unit]] = []
        for b in boats:
            br, bc = int(b.pos[0]), int(b.pos[1])
            for u in state.units[eng]:
                if not u.is_alive or u.unit_type == UnitType.BLACK_BOAT:
                    continue
                if not _black_boat_repair_eligible(state, u):
                    continue
                if (
                    use_hp
                    and hp_key is not None
                    and not _repair_display_hp_matches_hint(u.display_hp, hp_key)
                ):
                    continue
                d = abs(int(u.pos[0]) - br) + abs(int(u.pos[1]) - bc)
                ranked.append((d, b, u))
        ranked.sort(
            key=lambda t: (t[0], int(t[1].unit_id), int(t[2].unit_id)),
        )
        return ranked

    for use_hp in (True, False):
        ranked = _rank(use_hp)
        if not ranked:
            continue
        best_d = ranked[0][0]
        tops = [t for t in ranked if t[0] == best_d]
        if len(tops) == 1:
            chosen = tops[0]
        elif acting_boat is not None:
            # Multiple allies at the same Manhattan distance from this boat (crowded front).
            chosen = min(tops, key=lambda t: int(t[2].unit_id))
        else:
            continue
        # Black Boats cover long sea legs; move-drift can leave the ally several tiles off
        # until :func:`_oracle_snap_black_boat_toward_repair_ally` (reg. 1634889).
        if best_d <= 12:
            return chosen[1], chosen[2]
    return None


def _oracle_fallback_nearest_allied_repair_target_pos(
    state: GameState,
    eng: int,
    *,
    hp_key: Optional[int] = None,
    acting_boat: Optional[Unit] = None,
) -> Optional[tuple[int, int]]:
    """Return ally tile from :func:`_oracle_fallback_repair_boat_and_ally`."""
    pair = _oracle_fallback_repair_boat_and_ally(
        state, eng, hp_key=hp_key, acting_boat=acting_boat
    )
    if pair is None:
        return None
    u = pair[1]
    return int(u.pos[0]), int(u.pos[1])


def _enumerate_bb_repair_pairs(state: GameState, eng: int) -> list[tuple[Unit, Unit]]:
    """All (Black Boat, orthogonally adjacent repair-eligible ally) for engine ``eng``."""
    from engine.action import _black_boat_repair_eligible

    boats = [
        u
        for u in state.units[eng]
        if u.is_alive and u.unit_type == UnitType.BLACK_BOAT
    ]
    out: list[tuple[Unit, Unit]] = []
    for b in boats:
        br, bc = _black_boat_oracle_action_tile(state, b)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = br + dr, bc + dc
            adj = state.get_unit_at(tr, tc)
            if adj is None or int(adj.player) != eng:
                continue
            if int(adj.unit_id) == int(b.unit_id):
                continue
            if _black_boat_repair_eligible(state, adj):
                out.append((b, adj))
    return out


def _enumerate_bb_adjacent_allies_loose(state: GameState, eng: int) -> list[tuple[Unit, Unit]]:
    """(Black Boat, adjacent ally) without HP/fuel/ammo eligibility — for target resolution only."""
    boats = [
        u
        for u in state.units[eng]
        if u.is_alive and u.unit_type == UnitType.BLACK_BOAT
    ]
    out: list[tuple[Unit, Unit]] = []
    for b in boats:
        br, bc = _black_boat_oracle_action_tile(state, b)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = br + dr, bc + dc
            adj = state.get_unit_at(tr, tc)
            if adj is None or int(adj.player) != eng:
                continue
            if int(adj.unit_id) == int(b.unit_id):
                continue
            out.append((b, adj))
    return out


def _resolve_repair_target_tile(
    state: GameState,
    repair_block: dict[str, Any],
    *,
    eng: int,
    boat_hint: Optional[Unit] = None,
    envelope_awbw_player_id: Optional[int] = None,
) -> tuple[int, int]:
    """Tile of the unit being repaired (AWBW ``repaired.global``).

    ``boat_hint`` is the acting Black Boat after the nested ``Move`` (when known);
    narrows (boat, ally) pairs when several boats each have an eligible neighbour.
    """
    rep_gl = _repair_repaired_global_dict(
        repair_block, envelope_awbw_player_id=envelope_awbw_player_id
    )
    if not rep_gl or _oracle_awbw_scalar_int_optional(rep_gl.get("units_id")) is None:
        raise UnsupportedOracleAction("Repair: missing repaired.global.units_id")
    tid = _oracle_awbw_scalar_int_optional(rep_gl.get("units_id"))
    ty_hint = _oracle_awbw_scalar_int_optional(rep_gl.get("units_y"))
    tx_hint = _oracle_awbw_scalar_int_optional(rep_gl.get("units_x"))
    if ty_hint is not None and tx_hint is not None:
        return ty_hint, tx_hint
    u = _unit_by_awbw_units_id(state, tid) if tid is not None else None
    if u is not None:
        return u.pos
    pairs = _enumerate_bb_repair_pairs(state, eng)
    if (
        boat_hint is not None
        and boat_hint.is_alive
        and boat_hint.unit_type == UnitType.BLACK_BOAT
        and int(boat_hint.player) == eng
    ):
        bh_only = [
            (b, t)
            for b, t in pairs
            if int(b.unit_id) == int(boat_hint.unit_id)
            and int(b.player) == int(boat_hint.player)
        ]
        if bh_only:
            pairs = bh_only
    hp_key = _oracle_awbw_scalar_int_optional(rep_gl.get("units_hit_points"))
    pairs_unc = list(pairs)
    if hp_key is not None:
        w = hp_key
        strict = [(b, t) for b, t in pairs if t.display_hp == w]
        relaxed = [
            (b, t) for b, t in pairs if _repair_display_hp_matches_hint(t.display_hp, w)
        ]
        pairs = strict if strict else relaxed
    # Site ``units_hit_points`` can disagree with bars after prior-step drift;
    # restore unfiltered pairs when HP hints eliminate everyone.
    if len(pairs) == 0 and len(pairs_unc) > 0:
        pairs = pairs_unc
    if len(pairs) > 1:
        tposes = {t.pos for _, t in pairs}
        if len(tposes) == 1:
            return next(iter(tposes))
        if hp_key is not None:
            w = hp_key
            hp_pairs = [(b, t) for b, t in pairs if _repair_display_hp_matches_hint(t.display_hp, w)]
            if len(hp_pairs) == 1:
                return hp_pairs[0][1].pos
        try:
            boat_bid = _repair_boat_awbw_id(repair_block)
        except UnsupportedOracleAction:
            boat_bid = None
        if boat_bid is not None:
            bb = _unit_by_awbw_units_id(state, int(boat_bid))
            if bb is not None:
                narrowed = [(b, t) for b, t in pairs if b is bb]
                if len(narrowed) == 1:
                    return narrowed[0][1].pos
        # One boat ortho-adjacent to two eligible allies (AWBW still picks one target).
        boat_ids = {id(b) for b, _ in pairs}
        if len(boat_ids) == 1:
            best = min(
                pairs,
                key=lambda bt: (bt[1].pos[0], bt[1].pos[1], int(bt[1].unit_id)),
            )
            return best[1].pos
    if len(pairs) == 1:
        return pairs[0][1].pos
    if len(pairs) == 0:
        # Engine eligibility can disagree with the recorded Repair line (e.g. full
        # HP but site still issued REPAIR for resupply, or state drift vs PHP).
        loose = _enumerate_bb_adjacent_allies_loose(state, eng)
        if (
            boat_hint is not None
            and boat_hint.is_alive
            and boat_hint.unit_type == UnitType.BLACK_BOAT
            and int(boat_hint.player) == eng
        ):
            loose_bh = [
                (b, t)
                for b, t in loose
                if int(b.unit_id) == int(boat_hint.unit_id)
                and int(b.player) == int(boat_hint.player)
            ]
            if loose_bh:
                loose = loose_bh
        loose_unc = list(loose)
        if hp_key is not None:
            w = hp_key
            strict_l = [(b, t) for b, t in loose if t.display_hp == w]
            relaxed_l = [
                (b, t) for b, t in loose if _repair_display_hp_matches_hint(t.display_hp, w)
            ]
            loose = strict_l if strict_l else relaxed_l
        if len(loose) == 0 and len(loose_unc) > 0:
            loose = loose_unc
        if len(loose) > 1:
            tposes = {t.pos for _, t in loose}
            if len(tposes) == 1:
                return next(iter(tposes))
            if hp_key is not None:
                w = hp_key
                hp_loose = [
                    (b, t) for b, t in loose if _repair_display_hp_matches_hint(t.display_hp, w)
                ]
                if len(hp_loose) == 1:
                    return hp_loose[0][1].pos
            try:
                boat_bid = _repair_boat_awbw_id(repair_block)
            except UnsupportedOracleAction:
                boat_bid = None
            if boat_bid is not None:
                bb = _unit_by_awbw_units_id(state, int(boat_bid))
                if bb is not None:
                    narrowed = [(b, t) for b, t in loose if b is bb]
                    if len(narrowed) == 1:
                        return narrowed[0][1].pos
            if len({id(b) for b, _ in loose}) == 1:
                best = min(
                    loose,
                    key=lambda bt: (bt[1].pos[0], bt[1].pos[1], int(bt[1].unit_id)),
                )
                return best[1].pos
            if len(loose) > 1:
                pick = min(
                    loose,
                    key=lambda bt: (
                        _black_boat_oracle_action_tile(state, bt[0])[0],
                        _black_boat_oracle_action_tile(state, bt[0])[1],
                        bt[1].pos[0],
                        bt[1].pos[1],
                        int(bt[1].unit_id),
                    ),
                )
                return pick[1].pos
        if len(loose) == 1:
            return loose[0][1].pos
        if len(loose) == 0:
            boats = [
                u
                for u in state.units[eng]
                if u.is_alive and u.unit_type == UnitType.BLACK_BOAT
            ]
            if len(boats) == 1:
                br, bc = _black_boat_oracle_action_tile(state, boats[0])
                adj_allies: list[Unit] = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ax = state.get_unit_at(br + dr, bc + dc)
                    if ax is None or int(ax.player) != eng or ax is boats[0]:
                        continue
                    adj_allies.append(ax)
                raw_adj = list(adj_allies)
                if hp_key is not None:
                    w = hp_key
                    strict_a = [a for a in adj_allies if a.display_hp == w]
                    relaxed_a = [
                        a for a in adj_allies if _repair_display_hp_matches_hint(a.display_hp, w)
                    ]
                    adj_allies = strict_a if strict_a else relaxed_a
                if len(adj_allies) == 1:
                    return adj_allies[0].pos
                if len(adj_allies) == 0 and len(raw_adj) == 1:
                    return raw_adj[0].pos
                if len(adj_allies) == 0 and raw_adj:
                    if hp_key is not None:
                        w = hp_key
                        hp_raw = [
                            a
                            for a in raw_adj
                            if _repair_display_hp_matches_hint(a.display_hp, w)
                        ]
                        if len(hp_raw) == 1:
                            return hp_raw[0].pos
                    pick = min(
                        raw_adj, key=lambda a: (int(a.unit_id), a.pos[0], a.pos[1])
                    )
                    return pick.pos
            fb_pos = _oracle_fallback_nearest_allied_repair_target_pos(
                state, eng, hp_key=hp_key
            )
            if fb_pos is not None:
                return fb_pos
            raise UnsupportedOracleAction(
                "Repair: no repair-eligible ally for awbw repaired id "
                f"{tid if tid is not None else rep_gl.get('units_id')!r}"
            )
        raise UnsupportedOracleAction(
            "Repair: ambiguous repair target for awbw id "
            f"{tid if tid is not None else rep_gl.get('units_id')!r} ({len(loose)} loose (boat,ally) pairs)"
        )
    if len(pairs) > 1:
        # Multiple boats each with an eligible neighbour and no tighter signal from
        # AWBW ids/coords — pick deterministically (register: ``oracle_repair``).
        pick = min(
            pairs,
            key=lambda bt: (
                _black_boat_oracle_action_tile(state, bt[0])[0],
                _black_boat_oracle_action_tile(state, bt[0])[1],
                bt[1].pos[0],
                bt[1].pos[1],
                int(bt[1].unit_id),
            ),
        )
        return pick[1].pos
    raise UnsupportedOracleAction(
        "Repair: internal error resolving target for awbw id "
        f"{tid if tid is not None else rep_gl.get('units_id')!r} (pairs={len(pairs)})"
    )


def _unit_by_awbw_units_id(state: GameState, units_id: int) -> Optional[Unit]:
    for pl in state.units.values():
        for u in pl:
            if int(u.unit_id) == int(units_id) and u.is_alive:
                return u
    return None


def _oracle_attack_eval_pos(state: GameState, unit: Unit) -> tuple[int, int]:
    """Tile :func:`get_attack_targets` measures from for ``unit`` in this snapshot.

    In ``ActionStage.ACTION`` the unit often still occupies its pre-move map tile
    while ``state.selected_move_pos`` records the chosen stop; legal ATTACK
    enumeration passes ``move_pos=selected_move_pos`` into
    :func:`get_attack_targets` (``_get_action_actions``). Mirror that for
    ``oracle_fire`` / seam attacker resolution so we only extend resolution when
    the engine would already list the defender from that origin.

    Indirect units cannot move and fire the same turn; if ``selected_move_pos``
    disagrees with ``unit.pos`` for an indirect, keep ``unit.pos`` so we do not
    invent range from an illegal cell.

    When ``move_pos`` is a **boarding** tile (friendly transport / join partner
    other than ``unit``), ``_get_action_actions`` never reaches the ATTACK loop
    for that ``move_pos`` — use ``unit.pos`` so we do not treat the transport
    hex as a firing origin.
    """
    if state.action_stage != ActionStage.ACTION:
        return unit.pos
    su = state.selected_unit
    if su is None:
        return unit.pos
    if su is not unit and (
        int(su.unit_id) != int(unit.unit_id)
        or int(su.player) != int(unit.player)
    ):
        return unit.pos
    if int(unit.player) != int(state.active_player):
        return unit.pos
    mp = state.selected_move_pos
    if mp is None:
        return unit.pos
    player = int(unit.player)
    occupant = state.get_unit_at(*mp)
    boarding = (
        occupant is not None
        and occupant.player == player
        and occupant.pos != unit.pos
    )
    if boarding:
        return unit.pos
    stats = UNIT_STATS[unit.unit_type]
    if stats.is_indirect and mp != unit.pos:
        return unit.pos
    return mp


_ORACLE_SEAM_ATTACK_TARGET_TERRAIN: frozenset[int] = frozenset({113, 114, 115, 116})


def _oracle_pick_attack_seam_terminator(
    state: GameState,
    legal: list[Action],
    target: tuple[int, int],
    *,
    path_end: tuple[int, int],
) -> Optional[Action]:
    """Resolve ``AttackSeam`` after ``SELECT``→``move_pos`` (same half-turn as ``Fire``).

    PHP ``seamY``/``seamX`` may still name the **intact** pipe cell while the
    engine only exposes ``ATTACK`` onto neighbouring **rubble** (115/116) after
    seam HP dropped partway through the day (``oracle_seam`` cluster). Prefer
    an exact ``target_pos`` match, else the legal ``ATTACK`` whose target is
    seam/rubble terrain within Manhattan distance ``≤ 2`` of the declared
    coordinate (tie-break: distance, row, col).

    When the site row is a no-op (only ``WAIT`` at the end tile, or every
    ``ATTACK`` is against **non-seam** tiles while the zip still says
    ``AttackSeam``), commit ``WAIT`` at ``selected_move_pos`` / ``path_end`` so
    the replay advances without fabricating a unit attack.
    """
    tr, tc = int(target[0]), int(target[1])
    end_candidates: list[tuple[int, int]] = []
    if state.selected_move_pos is not None:
        end_candidates.append(tuple(state.selected_move_pos))
    pt = tuple(path_end)
    if pt not in end_candidates:
        end_candidates.append(pt)

    for er, ec in end_candidates:
        for a in legal:
            if (
                a.action_type == ActionType.ATTACK
                and a.move_pos == (er, ec)
                and a.target_pos == target
            ):
                return a

    for a in legal:
        if a.action_type == ActionType.ATTACK and a.target_pos == target:
            return a

    best: Optional[tuple[int, int, int, Action]] = None
    for a in legal:
        if a.action_type != ActionType.ATTACK or a.target_pos is None:
            continue
        rr, cc = int(a.target_pos[0]), int(a.target_pos[1])
        tid = state.map_data.terrain[rr][cc]
        if tid not in _ORACLE_SEAM_ATTACK_TARGET_TERRAIN:
            continue
        d = abs(rr - tr) + abs(cc - tc)
        if d > 2:
            continue
        if best is None or (d, rr, cc) < (best[0], best[1], best[2]):
            best = (d, rr, cc, a)
    if best is not None:
        return best[3]

    if legal and all(a.action_type == ActionType.WAIT for a in legal):
        for er, ec in end_candidates:
            for a in legal:
                if a.action_type == ActionType.WAIT and a.move_pos == (er, ec):
                    return a
        return legal[0]

    attacks = [a for a in legal if a.action_type == ActionType.ATTACK]
    waits = [a for a in legal if a.action_type == ActionType.WAIT]
    if attacks and waits:
        seam_strikes = [
            a
            for a in attacks
            if a.target_pos is not None
            and state.map_data.terrain[int(a.target_pos[0])][int(a.target_pos[1])]
            in _ORACLE_SEAM_ATTACK_TARGET_TERRAIN
        ]
        if not seam_strikes:
            for er, ec in end_candidates:
                for a in waits:
                    if a.move_pos == (er, ec):
                        return a
    return None


def _oracle_fire_attack_move_pos_candidates(
    state: GameState, unit: Unit
) -> list[tuple[int, int]]:
    """``move_pos`` values to try with :func:`get_attack_targets` for ``unit``.

    Indirects only ever attack from ``unit.pos`` (same-turn move+fire is illegal).
    Directs try ``unit.pos`` then :func:`_oracle_attack_eval_pos` when it differs,
    covering ACTION drift where AWBW ``combatInfo`` / PHP vision disagrees with
    which tile the engine uses as the firing anchor (``oracle_fire`` cluster).
    """
    stats = UNIT_STATS[unit.unit_type]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(p: tuple[int, int]) -> None:
        if p not in seen:
            seen.add(p)
            out.append(p)

    add(unit.pos)
    if stats.is_indirect:
        return out
    add(_oracle_attack_eval_pos(state, unit))
    return out


def _oracle_unit_can_attack_target_cell(
    state: GameState, unit: Unit, eng: Optional[int], target_rc: tuple[int, int]
) -> bool:
    """If ``eng`` is set, only units on that engine seat are considered (1613840 cross-scan passes ``None``)."""
    if eng is not None and int(unit.player) != int(eng):
        return False
    for mp in _oracle_fire_attack_move_pos_candidates(state, unit):
        if target_rc in get_attack_targets(state, unit, mp):
            return True
    return False


def _oracle_unit_grit_jake_probe_for_target(
    state: GameState, unit: Unit, eng: int, target_rc: tuple[int, int]
) -> Optional[Unit]:
    if int(unit.player) != int(eng):
        return None
    for mp in _oracle_fire_attack_move_pos_candidates(state, unit):
        u_grit = _oracle_try_grit_jake_indirect_fire(state, unit, mp, eng, target_rc)
        if u_grit is not None:
            return u_grit
    return None


def _oracle_indirect_manhattan_ring_ok(
    state: GameState, unit: Unit, tr: int, tc: int
) -> bool:
    """Whether ``(tr, tc)`` lies on the indirect Manhattan ring for ``unit.pos``.

    Matches :func:`get_attack_targets` distance + CO max-range buffs (Grit / Jake).
    Used when :func:`get_attack_targets` is empty (e.g. no ammo) but we still need a
    geometric guess for ``oracle_fire`` fallback.
    """
    stats = UNIT_STATS[unit.unit_type]
    if not stats.is_indirect:
        return False
    dist = abs(int(unit.pos[0]) - tr) + abs(int(unit.pos[1]) - tc)
    min_r, max_r = stats.min_range, stats.max_range
    co = state.co_states[int(unit.player)]
    if co.co_id == 2:
        if co.scop_active:
            max_r += 2
        elif co.cop_active:
            max_r += 1
    elif co.co_id == 22 and (co.cop_active or co.scop_active):
        if stats.unit_class != "naval":
            max_r += 1
    return min_r <= dist <= max_r


def _oracle_try_grit_jake_indirect_fire(
    state: GameState,
    unit: Unit,
    move_pos: tuple[int, int],
    engine_player: int,
    target_rc: tuple[int, int],
) -> Optional[Unit]:
    """When base range misses, try Grit/Jake COP/SCOP max-range (zip may omit ``Power``).

    AWBW still records the strike in ``combatInfo`` (e.g. 1627004: Grit artillery
    at Manhattan distance 4 needs COP +1). Leaves ``co.cop_active`` / ``scop_active``
    set when a probe succeeds so the following ``ATTACK`` matches
    :func:`get_attack_targets`.
    """
    if int(unit.player) != int(engine_player):
        return None
    st = UNIT_STATS[unit.unit_type]
    if not st.is_indirect:
        return None
    co = state.co_states[int(unit.player)]
    if co.co_id not in (2, 22):
        return None
    if co.co_id == 22 and st.unit_class == "naval":
        return None
    if target_rc in get_attack_targets(state, unit, move_pos):
        return unit
    ocop, oscop = co.cop_active, co.scop_active
    for cop, scop in ((True, False), (False, True)):
        co.cop_active, co.scop_active = cop, scop
        if target_rc in get_attack_targets(state, unit, move_pos):
            return unit
    co.cop_active, co.scop_active = ocop, oscop
    return None


def _resolve_fire_or_seam_attacker(
    state: GameState,
    *,
    engine_player: int,
    awbw_units_id: int,
    anchor_r: int,
    anchor_c: int,
    target_r: int,
    target_c: int,
    hp_hint: Optional[int] = None,
) -> Optional[Unit]:
    """Resolve attacker when AWBW ``combatInfo`` ids/tiles disagree with engine truth.

    Site ``combatInfoVision`` can disagree with the ``p:`` envelope seat (interleaved
    half-turns / vision rows): the tile in ``attacker.units_{y,x}`` may hold the
    **legal** striker while ``active_player`` still matches the prior envelope
    (``oracle_fire`` cluster, e.g. 1609589). When the anchor tile is empty or holds
    a unit that cannot strike ``target``, fall back to **any** alive unit that can
    legally ``get_attack_targets``→``target`` (same hp_hint / tie-break rules), e.g.
    1613840 stale attacker coordinates with a valid cross-seat shot.
    """
    from engine.action import _can_attack_submerged_or_hidden
    from engine.combat import get_base_damage

    eng = int(engine_player)
    tr, tc = int(target_r), int(target_c)
    target_rc = (tr, tc)
    ar, ac = int(anchor_r), int(anchor_c)

    u_id = _unit_by_awbw_units_id(state, int(awbw_units_id))
    if u_id is not None and int(u_id.player) == eng:
        if _oracle_unit_can_attack_target_cell(state, u_id, eng, target_rc):
            return u_id
        u_grit = _oracle_unit_grit_jake_probe_for_target(state, u_id, eng, target_rc)
        if u_grit is not None:
            return u_grit

    x = state.get_unit_at(ar, ac)
    if x is not None and x.is_alive:
        cross_hit = False
        for mp in _oracle_fire_attack_move_pos_candidates(state, x):
            if target_rc in get_attack_targets(state, x, mp):
                cross_hit = True
                break
        if cross_hit:
            if int(x.player) == eng:
                return x
            # Legal strike from anchor but occupant is the other engine seat (envelope lag).
            du = state.get_unit_at(tr, tc)
            if du is None or int(du.player) != int(x.player):
                return x
        elif int(x.player) == eng:
            u_grit = _oracle_unit_grit_jake_probe_for_target(state, x, eng, target_rc)
            if u_grit is not None:
                return u_grit

    cands: list[Unit] = []
    for cand in state.units[eng]:
        if not cand.is_alive:
            continue
        if _oracle_unit_can_attack_target_cell(state, cand, eng, target_rc):
            cands.append(cand)
    if not cands:
        seen_ids: set[int] = set()
        for pl in (0, 1):
            for cand in state.units[pl]:
                if not cand.is_alive:
                    continue
                cid = int(cand.unit_id)
                if cid in seen_ids:
                    continue
                if _oracle_unit_can_attack_target_cell(state, cand, None, target_rc):
                    seen_ids.add(cid)
                    cands.append(cand)
    if not cands:
        for cand in state.units[eng]:
            if not cand.is_alive:
                continue
            u_grit = _oracle_unit_grit_jake_probe_for_target(state, cand, eng, target_rc)
            if u_grit is not None:
                cands.append(u_grit)
                break
    if len(cands) > 1 and hp_hint is not None:
        want = int(hp_hint)
        hp_match = [c for c in cands if int(c.display_hp) == want]
        if len(hp_match) != 1:
            hp_match = [
                c for c in cands if _repair_display_hp_matches_hint(int(c.display_hp), want)
            ]
        if len(hp_match) == 1:
            return hp_match[0]
    if len(cands) > 1:
        du_n = state.get_unit_at(tr, tc)
        if (
            du_n is not None
            and du_n.is_alive
            and int(du_n.player) != eng
        ):
            narrowed = [
                c
                for c in cands
                if get_base_damage(c.unit_type, du_n.unit_type) is not None
            ]
            if len(narrowed) == 1:
                return narrowed[0]
            if narrowed:
                cands = narrowed
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        def _tie(uu: Unit) -> tuple[int, int, int, int]:
            # Prefer units closest to the strike target; AWBW anchor coords are often stale.
            dt = abs(uu.pos[0] - tr) + abs(uu.pos[1] - tc)
            return (dt, uu.pos[0], uu.pos[1], uu.unit_id)

        cands.sort(key=_tie)
        return cands[0]

    defender_u = state.get_unit_at(tr, tc)
    if defender_u is not None and int(defender_u.player) != eng:
        adj_one: list[Unit] = []
        for cand in state.units[eng]:
            if not cand.is_alive:
                continue
            stc = UNIT_STATS[cand.unit_type]
            if stc.is_indirect:
                if not _oracle_indirect_manhattan_ring_ok(state, cand, tr, tc):
                    continue
            else:
                ok_adj = False
                for er, ec in _oracle_fire_attack_move_pos_candidates(state, cand):
                    if max(abs(er - tr), abs(ec - tc)) == 1:
                        ok_adj = True
                        break
                if not ok_adj:
                    continue
            if get_base_damage(cand.unit_type, defender_u.unit_type) is None:
                continue
            if defender_u.is_submerged and not _can_attack_submerged_or_hidden(
                cand, defender_u
            ):
                continue
            adj_one.append(cand)
        if len(adj_one) > 1 and hp_hint is not None:
            want = int(hp_hint)
            hp_m = [c for c in adj_one if int(c.display_hp) == want]
            if len(hp_m) != 1:
                hp_m = [
                    c
                    for c in adj_one
                    if _repair_display_hp_matches_hint(int(c.display_hp), want)
                ]
            if len(hp_m) == 1:
                return hp_m[0]
        if len(adj_one) == 1:
            return adj_one[0]
    return None


def _resolve_attackseam_no_path_attacker(
    state: GameState,
    *,
    eng: int,
    awbw_units_id: int,
    anchor_r: int,
    anchor_c: int,
    seam_row: int,
    seam_col: int,
    hp_hint: Optional[int],
) -> Optional[Unit]:
    """``AttackSeam`` without ``Move.paths``: widen search when combatInfo anchor drifts.

    GL **1609533**: the declared tile can be empty while another friendly still has a
    legal seam strike (including rubble within Manhattan ``≤ 2`` of the declared
    seam cell, mirroring :func:`_oracle_pick_attack_seam_terminator`).
    """
    from engine.combat import get_seam_base_damage

    tr, tc = int(seam_row), int(seam_col)
    ar, ac = int(anchor_r), int(anchor_c)

    def _seam_strike_target_cells() -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = [(tr, tc)]
        seen: set[tuple[int, int]] = {(tr, tc)}
        h, w = state.map_data.height, state.map_data.width
        for rr in range(h):
            for cc in range(w):
                tid = int(state.map_data.terrain[rr][cc])
                if tid not in (113, 114, 115, 116):
                    continue
                if abs(rr - tr) + abs(cc - tc) > 2:
                    continue
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    cells.append((rr, cc))
        return cells

    for e_try in (int(eng), 1 - int(eng)) if int(eng) in (0, 1) else (int(eng),):
        u = _resolve_fire_or_seam_attacker(
            state,
            engine_player=e_try,
            awbw_units_id=int(awbw_units_id),
            anchor_r=ar,
            anchor_c=ac,
            target_r=tr,
            target_c=tc,
            hp_hint=hp_hint,
        )
        if u is not None:
            return u

    target_cells = _seam_strike_target_cells()
    loose: list[Unit] = []
    seen_id: set[int] = set()
    for pl in (0, 1):
        for cand in state.units[pl]:
            if not cand.is_alive:
                continue
            sd = get_seam_base_damage(cand.unit_type)
            if sd is None or sd <= 0:
                continue
            ok = False
            for mp in _oracle_fire_attack_move_pos_candidates(state, cand):
                g = get_attack_targets(state, cand, mp)
                if any(cell in g for cell in target_cells):
                    ok = True
                    break
            if not ok:
                continue
            cid = int(cand.unit_id)
            if cid in seen_id:
                continue
            seen_id.add(cid)
            loose.append(cand)

    seat_cands = [c for c in loose if int(c.player) == int(eng)]
    if not seat_cands:
        seat_cands = loose
    if len(seat_cands) > 1 and hp_hint is not None:
        want = int(hp_hint)
        hp_match = [c for c in seat_cands if int(c.display_hp) == want]
        if len(hp_match) != 1:
            hp_match = [
                c
                for c in seat_cands
                if _repair_display_hp_matches_hint(int(c.display_hp), want)
            ]
        if len(hp_match) == 1:
            return hp_match[0]
    if len(seat_cands) == 1:
        return seat_cands[0]
    if len(seat_cands) > 1:
        seat_cands.sort(
            key=lambda u: (
                abs(u.pos[0] - ar) + abs(u.pos[1] - ac),
                u.pos[0],
                u.pos[1],
                int(u.unit_id),
            )
        )
        return seat_cands[0]
    return None


def _oracle_diag_grit_jake_extended_range_may_hit(
    state: GameState, unit: Unit, target_rc: tuple[int, int]
) -> bool:
    """Read-only COP/SCOP probe so triage matches :func:`_oracle_try_grit_jake_indirect_fire`."""
    st = UNIT_STATS[unit.unit_type]
    if not st.is_indirect:
        return False
    co = state.co_states[int(unit.player)]
    if co.co_id not in (2, 22):
        return False
    if co.co_id == 22 and st.unit_class == "naval":
        return False
    for mp in _oracle_fire_attack_move_pos_candidates(state, unit):
        ocop, oscop = co.cop_active, co.scop_active
        try:
            for cop, scop in ((True, False), (False, True)):
                co.cop_active, co.scop_active = cop, scop
                if target_rc in get_attack_targets(state, unit, mp):
                    return True
        finally:
            co.cop_active, co.scop_active = ocop, oscop
    return False


def _oracle_diag_target_in_any_attack_targets(
    state: GameState, target_r: int, target_c: int
) -> bool:
    """Return True if any alive unit can include ``(target_r, target_c)`` in ``get_attack_targets``.

    Uses the same eval position as oracle attack resolution. This is **triage-only**:
    it does not prove which unit AWBW intended, only whether the engine still models
    *some* strike to that cell (vs envelope / snapshot drift or unmapped CO range).

    Indirect **Grit / Jake** COP and SCOP range extensions are probed here (with
    CO flags restored) so ``strike_possible_in_engine`` matches the resolver’s
    hypothetical-power path, not only the current snapshot flags.
    """
    tr, tc = int(target_r), int(target_c)
    tgt = (tr, tc)
    for pl in (0, 1):
        for u in state.units[pl]:
            if not u.is_alive:
                continue
            for ep in _oracle_fire_attack_move_pos_candidates(state, u):
                if tgt in get_attack_targets(state, u, ep):
                    return True
            if _oracle_diag_grit_jake_extended_range_may_hit(state, u, tgt):
                return True
    return False


def _oracle_fire_no_attacker_message_suffix(
    state: GameState, target_r: int, target_c: int
) -> str:
    """Append to ``UnsupportedOracleAction`` when :func:`_resolve_fire_or_seam_attacker` returns None."""
    if _oracle_diag_target_in_any_attack_targets(state, target_r, target_c):
        return (
            " [oracle_fire: strike_possible_in_engine=1 "
            "triage=resolver_gap_or_anchor]"
        )
    return (
        " [oracle_fire: strike_possible_in_engine=0 "
        "triage=drift_range_los_or_unmapped_co]"
    )


def _resolve_unload_transport(
    state: GameState,
    transport_awbw_id: int,
    cargo_ut: UnitType,
    drop_target: tuple[int, int],
    engine_player: int,
    *,
    cargo_awbw_units_id: Optional[int] = None,
) -> Unit:
    """Map site ``transportID`` + drop tile to an engine ``Unit`` (the carrier).

    Semantics (mirrors the module docstring / site PHP):

    - ``transportID`` may be the **carrier** ``units_id`` or the **cargo**
      ``units_id`` while that cargo only appears under ``loaded_units`` (it is
      removed from top-level ``state.units`` after ``LOAD`` in this engine).
    - :func:`_unit_by_awbw_units_id` only scans top-level lists, so a cargo id
      never resolves to a map unit; the explicit ``loaded_units`` scan handles it.
    - ``unit.global`` ``units_y``/``units_x`` may be the real drop tile, the
      **carrier** tile (cargo still drawn on the transport), or a stale cell
      several steps away — we only use geometry to disambiguate carriers, not
      to ignore a wrong half-turn (still raises when nothing fits).

    When ``transportID`` hits a carrier on the map, we only accept that carrier
    if it actually carries cargo matching this unload (``units_name`` /
    optional ``units_id``); otherwise we fall through so a wrong id + non-empty
    hull does not mask the correct adjacent transport.
    """

    def _cargo_matches_unload(c: Unit) -> bool:
        if c.unit_type == cargo_ut:
            return True
        if cargo_awbw_units_id is not None and int(c.unit_id) == int(cargo_awbw_units_id):
            return True
        return False

    u = _unit_by_awbw_units_id(state, transport_awbw_id)
    if u is not None and UNIT_STATS[u.unit_type].carry_capacity > 0 and u.loaded_units:
        matching = [c for c in u.loaded_units if _cargo_matches_unload(c)]
        if len(matching) == 1:
            return u
        if len(matching) > 1 and cargo_awbw_units_id is not None:
            id_pin = [c for c in matching if int(c.unit_id) == int(cargo_awbw_units_id)]
            if len(id_pin) == 1:
                return u

    for x in state.units[engine_player]:
        if not x.is_alive or UNIT_STATS[x.unit_type].carry_capacity == 0:
            continue
        for c in x.loaded_units:
            if int(c.unit_id) == int(transport_awbw_id) and _cargo_matches_unload(c):
                return x

    cands: list[Unit] = []
    tr, tc = drop_target
    for x in state.units[engine_player]:
        if not x.is_alive or UNIT_STATS[x.unit_type].carry_capacity == 0:
            continue
        if not x.loaded_units:
            continue
        if not any(_cargo_matches_unload(c) for c in x.loaded_units):
            continue
        if abs(x.pos[0] - tr) + abs(x.pos[1] - tc) == 1:
            cands.append(x)

    # Carrier coordinates in ``unit.global`` (cargo still shown on the transport).
    if not cands:
        for x in state.units[engine_player]:
            if not x.is_alive or UNIT_STATS[x.unit_type].carry_capacity == 0:
                continue
            if not x.loaded_units:
                continue
            if not any(_cargo_matches_unload(c) for c in x.loaded_units):
                continue
            if x.pos == drop_target:
                cands.append(x)

    if len(cands) == 1:
        return cands[0]
    if not cands:
        # Last resort: any carrier with this cargo type; prefer min Manhattan
        # distance from carrier to the site drop hint, then ``units_id`` /
        # ``transportID`` disambiguation (AWBW global can be wrong by >1 tile).
        loose: list[tuple[int, Unit]] = []
        for x in state.units[engine_player]:
            if not x.is_alive or UNIT_STATS[x.unit_type].carry_capacity == 0:
                continue
            if not any(_cargo_matches_unload(c) for c in x.loaded_units):
                continue
            d = abs(x.pos[0] - tr) + abs(x.pos[1] - tc)
            loose.append((d, x))
        if not loose:
            raise UnsupportedOracleAction(
                f"Unload: no transport adjacent to {drop_target} carrying {cargo_ut.name} "
                f"(transportID={transport_awbw_id})"
            )
        loose.sort(key=lambda t: (t[0], t[1].unit_id))
        best_d = loose[0][0]
        near = [u for d, u in loose if d == best_d]
        if len(near) == 1:
            return near[0]
        if cargo_awbw_units_id is not None:
            hit = [
                u
                for u in near
                if any(
                    int(c.unit_id) == int(cargo_awbw_units_id)
                    and _cargo_matches_unload(c)
                    for c in u.loaded_units
                )
            ]
            if len(hit) == 1:
                return hit[0]
        tid_hit = [u for u in near if int(u.unit_id) == int(transport_awbw_id)]
        if len(tid_hit) == 1:
            return tid_hit[0]
        cargo_tid = [
            u
            for u in near
            if any(
                int(c.unit_id) == int(transport_awbw_id) and _cargo_matches_unload(c)
                for c in u.loaded_units
            )
        ]
        if len(cargo_tid) == 1:
            return cargo_tid[0]
        raise UnsupportedOracleAction(
            f"Unload: ambiguous transport for drop {drop_target} cargo {cargo_ut.name}: "
            f"{len(near)} carriers at same distance (transportID={transport_awbw_id})"
        )
    raise UnsupportedOracleAction(
        f"Unload: ambiguous transport for drop {drop_target} cargo {cargo_ut.name}: "
        f"{len(cands)} candidates (transportID={transport_awbw_id})"
    )


def _name_to_unit_type(name: str) -> UnitType:
    n = str(name).strip()
    # Site JSON sometimes omits spaces / varies casing vs ``export_awbw_replay``.
    aliases = {
        "Md.Tank": "Md. Tank",
        "md.tank": "Md. Tank",
        "MD.TANK": "Md. Tank",
        "Neotank": "Neo Tank",
        "NeoTank": "Neo Tank",
        "neo tank": "Neo Tank",
        "NEO TANK": "Neo Tank",
        "Megatank": "Mega Tank",
        "mega tank": "Mega Tank",
        "MEGA TANK": "Mega Tank",
        # Live site sometimes uses singular; exporter uses plural (AWBW chart naming).
        "Rocket": "Rockets",
        "rocket": "Rockets",
        "ROCKET": "Rockets",
        "Anti Air": "Anti-Air",
        "anti air": "Anti-Air",
        "B Copter": "B-Copter",
        "b copter": "B-Copter",
        "T Copter": "T-Copter",
        "t copter": "T-Copter",
    }
    n = aliases.get(n, n)
    for ut, nm in _AWBW_UNIT_NAMES.items():
        if nm == n:
            return ut
    n_norm = " ".join(n.lower().split())
    for ut, nm in _AWBW_UNIT_NAMES.items():
        if " ".join(nm.lower().split()) == n_norm:
            return ut
    raise UnsupportedOracleAction(f"unknown AWBW unit name {name!r}")


def map_snapshot_player_ids_to_engine(
    snap0: dict[str, Any],
    co0: int,
    co1: int,
) -> dict[int, int]:
    """Map AWBW ``players[].id`` (PHP int) -> engine player 0/1 from CO assignment."""
    players = snap0.get("players") or {}
    rows: list[tuple[int, int, int]] = []
    for _k, p in players.items():
        if not isinstance(p, dict):
            continue
        pid = int(p["id"])
        order = int(p.get("order", 0))
        cid = int(p["co_id"])
        rows.append((order, pid, cid))
    rows.sort(key=lambda t: t[0])
    if len(rows) < 2:
        raise ValueError(f"expected >=2 players in snapshot, got {rows!r}")
    rows = rows[:2]
    (_, id_a, c_a), (_, id_b, c_b) = rows
    if {c_a, c_b} != {co0, co1}:
        raise ValueError(f"snapshot CO ids {c_a},{c_b} != expected {co0},{co1}")
    if c_a == co0:
        return {id_a: 0, id_b: 1}
    return {id_a: 1, id_b: 0}


def _pick_action_gzip_member(
    zf: zipfile.ZipFile,
    games_id_hint: Optional[int] = None,
) -> Optional[str]:
    """
    Find the zip entry that holds the ``p:`` action stream (``a<games_id>`` gzip).

    Official AWBW zips use ``a<games_id>``. Some mirrors ship multiple ``a*``
    members; we prefer ``a{games_id_hint}`` when present and nonempty, else
    the first ``a*`` member whose decompressed text contains ``p:``.
    """
    names = zf.namelist()
    candidates = [n for n in names if n.startswith("a")]
    if not candidates:
        return None

    def _has_p(raw: bytes) -> bool:
        try:
            txt = gzip.decompress(raw).decode("utf-8", "replace")
        except OSError:
            return False
        return "p:" in txt

    if games_id_hint is not None:
        preferred = f"a{int(games_id_hint)}"
        if preferred in names:
            raw = zf.read(preferred)
            if _has_p(raw):
                return preferred

    for n in sorted(candidates):
        raw = zf.read(n)
        if _has_p(raw):
            return n
    return None


def replay_zip_has_action_stream(
    path: Path,
    *,
    games_id: Optional[int] = None,
) -> bool:
    """
    True if ``path`` contains a gzip member with at least one ``p:`` envelope line.

    ReplayVersion 1 downloads (snapshot-only ``<games_id>`` entry, no ``a*``)
    return False — oracle replay cannot run without this stream.
    """
    gid = games_id
    if gid is None and path.stem.isdigit():
        gid = int(path.stem)
    with zipfile.ZipFile(path) as zf:
        return _pick_action_gzip_member(zf, gid) is not None


def extract_json_action_strings_from_envelope_line(line: str) -> list[str]:
    """Scan PHP envelope line for serialized JSON strings (``s:len:"..."``)."""
    bodies: list[str] = []
    i = 0
    while i < len(line):
        j = line.find("s:", i)
        if j < 0:
            break
        k = line.find(":", j + 2)
        try:
            n = int(line[j + 2 : k])
        except ValueError:
            i = j + 2
            continue
        if k + 1 >= len(line) or line[k + 1] != '"':
            i = j + 2
            continue
        start = k + 2
        body = line[start : start + n]
        if body.startswith('{"action":'):
            bodies.append(body)
        i = start + n + 2
    return bodies


def parse_p_envelopes_from_zip(path: Path) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """
    Return list of (awbw_player_id, day, [action_dict, ...]) in file order.

    Reads the gz member whose decompressed text contains ``p:`` lines (``a<game_id>``).

    Returns **[]** when the zip is ReplayVersion 1 style (only the ``<game_id>``
    PHP snapshot gzip, no ``a<game_id>`` action stream). Callers must treat an
    empty list as “no envelopes to apply”, not as success.
    """
    gid_hint: Optional[int] = int(path.stem) if path.stem.isdigit() else None
    with zipfile.ZipFile(path) as zf:
        action_name = _pick_action_gzip_member(zf, gid_hint)
        if action_name is None:
            return []
        raw = zf.read(action_name)

    txt = gzip.decompress(raw).decode("utf-8", "replace")
    out: list[tuple[int, int, list[dict[str, Any]]]] = []
    for line in txt.split("\n"):
        line = line.strip()
        if not line.startswith("p:"):
            continue
        m = re.match(r"p:(\d+);d:(\d+);", line)
        if not m:
            continue
        pid = int(m.group(1))
        day = int(m.group(2))
        blobs = extract_json_action_strings_from_envelope_line(line)
        actions = []
        for b in blobs:
            try:
                actions.append(json.loads(b))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad JSON in envelope: {e}") from e
        out.append((pid, day, actions))
    return out


def replay_first_mover_engine(
    envelopes: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
) -> Optional[int]:
    """Engine player index (0 or 1) who acts in the first non-empty envelope.

    Site zips are authoritative for turn order when it disagrees with the
    predeploy-only opener heuristic in ``make_initial_state``.
    """
    for pid, _day, actions in envelopes:
        if not actions:
            continue
        ip = int(pid)
        if ip not in awbw_to_engine:
            raise ValueError(f"envelope player id {ip} not in snapshot map {awbw_to_engine!r}")
        return int(awbw_to_engine[ip])
    return None


def replay_first_mover_from_snapshot_turn(
    snap0: dict[str, Any],
    awbw_to_engine: dict[int, int],
) -> Optional[int]:
    """Map AWBW turn-0 ``turn`` (active PHP ``players[].id``) to engine 0/1.

    PHP snapshots use ``turn`` as the seat whose clock is active at that frame.
    When every ``p:`` envelope has an empty action list (degenerate export) or
    ``replay_first_mover_engine`` otherwise returns ``None``, this matches AWBW
    opening ``active_player`` for ``make_initial_state(replay_first_mover=…)``.
    """
    raw = snap0.get("turn")
    if raw is None:
        return None
    try:
        ip = int(raw)
    except (TypeError, ValueError):
        return None
    if ip not in awbw_to_engine:
        return None
    return int(awbw_to_engine[ip])


def resolve_replay_first_mover(
    envelopes: list[tuple[int, int, list[dict[str, Any]]]],
    snap0: dict[str, Any],
    awbw_to_engine: dict[int, int],
) -> Optional[int]:
    """Prefer first non-empty ``p:`` envelope; else AWBW snapshot ``turn`` field."""
    fm = replay_first_mover_engine(envelopes, awbw_to_engine)
    if fm is not None:
        return fm
    return replay_first_mover_from_snapshot_turn(snap0, awbw_to_engine)


def _path_end_rc(path: list[dict[str, Any]]) -> tuple[int, int]:
    last = path[-1]
    return int(last["y"]), int(last["x"])


def _path_start_rc(path: list[dict[str, Any]]) -> tuple[int, int]:
    first = path[0]
    return int(first["y"]), int(first["x"])


def _orthogonal_span_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Every grid cell on a straight horizontal or vertical segment (inclusive)."""
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        out: list[tuple[int, int]] = []
        c = c1
        while True:
            out.append((r1, c))
            if c == c2:
                break
            c += step
        return out
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        out = []
        r = r1
        while True:
            out.append((r, c1))
            if r == r2:
                break
            r += step
        return out
    return [a, b]


def _dense_path_cells_orthogonal(paths: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Waypoints plus collinear gap-fill between consecutive waypoints.

    Site exports sometimes omit intermediate cells on a straight march; the
    engine unit can still sit on a skipped tile (desync register ``Move`` /
    ``Fire`` path vs global drift, rows 451+).
    """
    pts: list[tuple[int, int]] = []
    for wp in paths:
        try:
            pts.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not pts:
        return []
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(len(pts)):
        if i == 0:
            chunk = [pts[0]]
        else:
            pa, pb = pts[i - 1], pts[i]
            if pa[0] == pb[0] or pa[1] == pb[1]:
                chunk = _orthogonal_span_cells(pa, pb)
            else:
                # Diagonal consecutive waypoints: JSON often lists only the corners
                # while the unit sits on an omitted elbow (same gap class as
                # ``_collinear_anchor_bridge_cells`` L-bridges for ``Move: no unit``).
                chunk_set: set[tuple[int, int]] = set()
                chunk_list: list[tuple[int, int]] = []
                for cell in _manhattan_l_bridge_cells(pa, pb):
                    if cell not in chunk_set:
                        chunk_set.add(cell)
                        chunk_list.append(cell)
                for cell in _manhattan_l_vfirst_bridge_cells(pa, pb):
                    if cell not in chunk_set:
                        chunk_set.add(cell)
                        chunk_list.append(cell)
                chunk = chunk_list
        for cell in chunk:
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
    return out


def _manhattan_l_bridge_cells(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Horizontal-then-vertical elbow from ``p`` to ``q`` (AWBW-style L march)."""
    r1, c1 = p
    r2, c2 = q
    if r1 == r2 or c1 == c2:
        return _orthogonal_span_cells(p, q)
    elbow = (r1, c2)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in _orthogonal_span_cells(p, elbow):
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    for cell in _orthogonal_span_cells(elbow, q):
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _manhattan_l_vfirst_bridge_cells(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Vertical-then-horizontal elbow (alternate L when site path disagrees)."""
    r1, c1 = p
    r2, c2 = q
    if r1 == r2 or c1 == c2:
        return _orthogonal_span_cells(p, q)
    elbow = (r2, c1)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in _orthogonal_span_cells(p, elbow):
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    for cell in _orthogonal_span_cells(elbow, q):
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _collinear_anchor_bridge_cells(
    sr: int, sc: int, ur: int, uc: int, er: int, ec: int
) -> list[tuple[int, int]]:
    """Straight and one-bend links between path-start, ``unit.global``, and path-end.

    Site zips often disagree on which anchor lists the unit: collinear fills cover
    same-row/column drift; when anchors are not aligned, an **L** (horizontal then
    vertical) between each anchor pair can still contain the true tile (final-third
    ``Move`` rows with path vs global on different rows/columns).
    """
    a = (int(sr), int(sc))
    b = (int(ur), int(uc))
    cpos = (int(er), int(ec))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p, q in ((a, b), (a, cpos), (b, cpos)):
        for cell in _orthogonal_span_cells(p, q):
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
        if p[0] != q[0] and p[1] != q[1]:
            for cell in _manhattan_l_bridge_cells(p, q):
                if cell not in seen:
                    seen.add(cell)
                    out.append(cell)
            for cell in _manhattan_l_vfirst_bridge_cells(p, q):
                if cell not in seen:
                    seen.add(cell)
                    out.append(cell)
    return out


def _guess_unmoved_mover_from_site_unit_name(
    state: GameState,
    eng: int,
    paths: list[dict[str, Any]],
    gu: dict[str, Any],
    *,
    anchor_hint: Optional[tuple[int, int]] = None,
) -> Optional[Unit]:
    """
    Last-resort resolver when AWBW ``units_id`` / tile coords disagree with the
    engine (desync register: ``Move: no unit for engine …``). Use ``units_name``
    plus path geometry to pick the unique eligible mover.

    ``anchor_hint`` is typically ``(units_y, units_x)`` from the site global:
    when several units share the type, prefer candidates closest to that tile.
    """
    raw = gu.get("units_name") or gu.get("units_symbol")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        want_type = _name_to_unit_type(str(raw))
    except UnsupportedOracleAction:
        return None
    waypoints: list[tuple[int, int]] = []
    for wp in paths:
        try:
            waypoints.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not waypoints:
        return None
    sr, sc = _path_start_rc(paths)
    er, ec = _path_end_rc(paths)
    try:
        ur = int(gu["units_y"])
        uc = int(gu["units_x"])
    except (KeyError, TypeError, ValueError):
        ur, uc = sr, sc
    geo_touch: set[tuple[int, int]] = set(waypoints)
    geo_touch.update(_dense_path_cells_orthogonal(paths))
    geo_touch.update(_collinear_anchor_bridge_cells(sr, sc, ur, uc, er, ec))
    touch_list = list(geo_touch)

    path_end = waypoints[-1]
    unmoved = [
        x
        for x in state.units[eng]
        if x.is_alive and (not x.moved) and x.unit_type == want_type
    ]
    moved_same = [
        x
        for x in state.units[eng]
        if x.is_alive and x.moved and x.unit_type == want_type
    ]
    # Prefer not-yet-moved units; if none, allow **moved** same-type when that is
    # unambiguous (e.g. factory-built infantry has ``moved=True`` on spawn — site
    # zip still lists a ``Capt`` / ``Move`` for that unit; desync register
    # ``oracle_move_no_unit``).
    if unmoved:
        cands = unmoved
    elif len(moved_same) == 1:
        cands = moved_same
    elif len(moved_same) > 1:
        # Several built / already-moved units of this type: disambiguate like same-type
        # multi-unit (path progress + anchor tie-break), not ``return None``.
        cands = moved_same
    else:
        return None
    if len(cands) > 1:
        picked = _pick_same_type_mover_by_path_reachability(
            state,
            paths,
            cands,
            path_start_hint=(sr, sc),
            global_hint=(ur, uc),
        )
        if picked is not None:
            return picked
    on_path = [x for x in cands if x.pos in geo_touch]
    if len(on_path) == 1:
        return on_path[0]
    if len(cands) == 1:
        return cands[0]

    def dist_to_geometry(pos: tuple[int, int]) -> int:
        return min(abs(pos[0] - w[0]) + abs(pos[1] - w[1]) for w in touch_list)

    def dist_anchor(u: Unit) -> int:
        if anchor_hint is None:
            return 0
        return abs(u.pos[0] - anchor_hint[0]) + abs(u.pos[1] - anchor_hint[1])

    scored = [(dist_to_geometry(x.pos), x) for x in cands]
    scored.sort(
        key=lambda t: (t[0], dist_anchor(t[1]), t[1].pos[0], t[1].pos[1])
    )
    best_d = scored[0][0]
    if best_d > 80:
        return None
    near_best = [u for d, u in scored if d == best_d]
    if len(near_best) == 1:
        return near_best[0]

    def dist_end(u: Unit) -> int:
        return abs(u.pos[0] - path_end[0]) + abs(u.pos[1] - path_end[1])

    near_best.sort(
        key=lambda u: (dist_end(u), dist_anchor(u), u.pos[0], u.pos[1])
    )
    d0 = dist_end(near_best[0])
    same = [u for u in near_best if dist_end(u) == d0]
    if len(same) == 1:
        return same[0]
    # Still tied (e.g. two Infantry one step from path_end): pick deterministically.
    # UnitType.INFANTRY is IntEnum 0 — never use unit_type in boolean context.
    return min(same, key=lambda u: (u.pos[0], u.pos[1], int(u.unit_id)))


def _optional_declared_unit_type_from_move_gu(gu: dict[str, Any]) -> Optional[UnitType]:
    """AWBW ``units_name`` / ``units_symbol`` on the Move payload, if parseable."""
    raw = gu.get("units_name") or gu.get("units_symbol")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return _name_to_unit_type(str(raw).strip())
    except UnsupportedOracleAction:
        return None


def _nearest_reachable_along_path(
    paths: list[dict[str, Any]],
    reach: dict[tuple[int, int], int],
    path_end: tuple[int, int],
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """
    Site ``paths.global`` tails can disagree with engine occupancy (register:
    ``Capt`` / ``Move`` with ``legal=[]`` — ACTION dead-end from a friendly tile
    the zip still lists as the move end). ``compute_reachable_costs`` is
    authoritative; walk the path backward until a legal stop exists.
    """
    if path_end in reach:
        return path_end
    waypoints: list[tuple[int, int]] = []
    for wp in paths:
        try:
            waypoints.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    for pos in reversed(waypoints):
        if pos in reach:
            return pos
    if fallback in reach:
        return fallback
    if not reach:
        return path_end
    return min(
        reach.keys(),
        key=lambda p: abs(p[0] - path_end[0]) + abs(p[1] - path_end[1]),
    )


def _oracle_boarding_destination_blocks_action_menu(
    state: GameState, unit: Unit, dest: tuple[int, int]
) -> bool:
    """True when ``_get_action_actions`` would return only LOAD/JOIN (no ATTACK list).

    Ending a seam-approach move onto a friendly transport / join partner matches
    ``compute_reachable_costs`` boarding tiles but leaves **no** seam ``ATTACK`` in
    the ACTION menu (GL 1629178: pick an earlier strike waypoint instead).
    """
    player = int(unit.player)
    mr, mc = int(dest[0]), int(dest[1])
    occupant = state.get_unit_at(mr, mc)
    boarding = (
        occupant is not None
        and int(occupant.player) == player
        and occupant is not unit
        and occupant.pos != unit.pos
    )
    if not boarding:
        return False
    cap = UNIT_STATS[occupant.unit_type].carry_capacity
    if (
        cap > 0
        and unit.unit_type in get_loadable_into(occupant.unit_type)
        and len(occupant.loaded_units) < cap
    ):
        return True
    if units_can_join(unit, occupant):
        return True
    return True


def _furthest_reachable_path_stop_for_seam_attack(
    state: GameState,
    unit: Unit,
    paths: list[dict[str, Any]],
    reach: dict[tuple[int, int], int],
    seam_target: tuple[int, int],
    *,
    json_path_end: tuple[int, int],
    start_fallback: tuple[int, int],
) -> tuple[int, int]:
    """Prefer the last ``paths.global`` waypoint that can ``ATTACK`` the pipe seam from.

    When the JSON path end is blocked, :func:`_nearest_reachable_along_path` may
    stop one tile short (JOIN vs seam). If an earlier waypoint can still strike
    the seam, prefer that tile (GL 1629178).
    """
    from engine.action import get_attack_targets

    tr, tc = int(seam_target[0]), int(seam_target[1])
    waypoints: list[tuple[int, int]] = []
    for wp in paths:
        try:
            waypoints.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    best: Optional[tuple[int, int]] = None
    for pos in waypoints:
        if pos not in reach:
            continue
        if _oracle_boarding_destination_blocks_action_menu(state, unit, pos):
            continue
        if (tr, tc) in get_attack_targets(state, unit, pos):
            best = pos
    if best is not None:
        return best
    fb = _nearest_reachable_along_path(paths, reach, json_path_end, start_fallback)
    if not _oracle_boarding_destination_blocks_action_menu(state, unit, fb):
        return fb
    for pos in reversed(waypoints):
        if pos not in reach:
            continue
        if _oracle_boarding_destination_blocks_action_menu(state, unit, pos):
            continue
        return pos
    return fb


def _pick_same_type_mover_by_path_reachability(
    state: GameState,
    paths: list[dict[str, Any]],
    candidates: list[Unit],
    *,
    path_start_hint: Optional[tuple[int, int]] = None,
    global_hint: Optional[tuple[int, int]] = None,
) -> Optional[Unit]:
    """
    When several alive units share the site's declared type, pick the unique unit
    that can make the furthest progress along ``paths.global`` this turn (register:
    ``Move: no unit`` after geometry/id drift with ``len(same_type) > 1``).

    Tie-break when mobility metrics match: prefer the unit **closest** to JSON path
    start, then to ``unit.global`` (``global_hint``), matching AWBW when path vs
    global disagree.

    Returns ``None`` if no candidate is strictly best — caller keeps searching or
    raises.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    waypoints: list[tuple[int, int]] = []
    for wp in paths:
        try:
            waypoints.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not waypoints:
        return None
    sr, sc = _path_start_rc(paths)
    ps = path_start_hint if path_start_hint is not None else (sr, sc)
    path_end = waypoints[-1]
    rows: list[tuple[bool, int, int, int, int, Unit]] = []
    for u in candidates:
        reach = compute_reachable_costs(state, u)
        end_eff = _nearest_reachable_along_path(paths, reach, path_end, u.pos)
        full = path_end in reach
        last_idx = -1
        for i, w in enumerate(waypoints):
            if w in reach:
                last_idx = i
        manh = abs(end_eff[0] - path_end[0]) + abs(end_eff[1] - path_end[1])
        d_ps = abs(u.pos[0] - ps[0]) + abs(u.pos[1] - ps[1])
        d_gh = (
            abs(u.pos[0] - global_hint[0]) + abs(u.pos[1] - global_hint[1])
            if global_hint is not None
            else 0
        )
        rows.append((full, last_idx, manh, d_ps, d_gh, u))
    rows.sort(
        key=lambda t: (
            -t[0],
            -t[1],
            t[2],
            t[3],
            t[4],
            t[5].pos[0],
            t[5].pos[1],
            int(t[5].unit_id),
        )
    )
    best_key = rows[0][:5]
    tied = [r for r in rows if r[:5] == best_key]
    if len(tied) > 1:
        return None
    return rows[0][5]


def _try_transport_unload_deferral(
    state: GameState,
    legal: list[Action],
    end: tuple[int, int],
) -> bool:
    """
    If a transport with cargo ends its move with UNLOAD legal, stay in ACTION so a
    following site ``Unload`` line can apply (same as ``_finish_move_join_load_capture_wait``).
    """
    u_sel = state.selected_unit
    mp_sel = state.selected_move_pos
    unload_move_anchors: set[tuple[int, int]] = {end}
    if mp_sel is not None:
        unload_move_anchors.add(mp_sel)
    if u_sel is not None:
        unload_move_anchors.add(u_sel.pos)
    if (
        u_sel is not None
        and mp_sel is not None
        and state.action_stage == ActionStage.ACTION
        and UNIT_STATS[u_sel.unit_type].carry_capacity > 0
        and u_sel.loaded_units
        and any(
            a.action_type == ActionType.UNLOAD and a.move_pos in unload_move_anchors
            for a in legal
        )
    ):
        if u_sel.pos != mp_sel:
            state._move_unit(u_sel, mp_sel)
        state.selected_unit = u_sel
        state.selected_move_pos = u_sel.pos
        state.action_stage = ActionStage.ACTION
        return True
    return False


def _pick_unload_terminator_at_end(
    legal: list[Action],
    end: tuple[int, int],
    *,
    extra_move_anchors: tuple[tuple[int, int], ...] = (),
) -> Optional[Action]:
    """Deterministic UNLOAD when the site bundles it (e.g. ``Capt`` on transport).

    ``UNLOAD`` actions use the transport's committed tile as ``move_pos``. Site
    path tails / ``Capt`` envelopes sometimes disagree with ``end`` by one tile
    while still listing legal UNLOADs keyed to the transport's true square —
    merge ``end`` with optional anchors (path tail, ``selected_unit.pos``, …).
    """
    anchors: set[tuple[int, int]] = {end}
    for p in extra_move_anchors:
        anchors.add(p)
    hits = [
        a
        for a in legal
        if a.action_type == ActionType.UNLOAD
        and a.move_pos is not None
        and a.move_pos in anchors
        and a.target_pos is not None
        and a.unit_type is not None
    ]
    if not hits:
        return None
    hits.sort(
        key=lambda a: (
            a.target_pos[0],
            a.target_pos[1],
            a.unit_type.name if a.unit_type is not None else "",
        )
    )
    return hits[0]


def _pick_attack_terminator_at_end(
    legal: list[Action],
    end: tuple[int, int],
) -> Optional[Action]:
    """
    AWBW site exports may use ``Capt`` or a plain ``Move`` when the only legal
    ACTION terminators are **ATTACK** + WAIT (adjacent enemy: must fight before
    capture rules expose ``CAPTURE``). Pick a deterministic ATTACK at ``end``.
    """
    hits = [
        a
        for a in legal
        if a.action_type == ActionType.ATTACK
        and a.move_pos == end
        and a.target_pos is not None
    ]
    if not hits:
        return None
    hits.sort(key=lambda a: (a.target_pos[0], a.target_pos[1]))
    return hits[0]


def _pick_singleton_join_or_load(state: GameState, legal: list[Action]) -> Optional[Action]:
    """
    Site path tails can disagree with ``selected_move_pos`` by one tile; filters
    that require ``a.move_pos == end`` then skip the only legal **JOIN** / **LOAD**.
    Only commit when the engine's committed tile matches the action (avoid wrong JOIN).
    """
    if len(legal) != 1:
        return None
    a = legal[0]
    if a.action_type not in (ActionType.JOIN, ActionType.LOAD):
        return None
    smp = state.selected_move_pos
    if smp is not None and a.move_pos == smp:
        return a
    return None


def _apply_move_paths_then_terminator(
    state: GameState,
    move: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
    *,
    after_move: Callable[[], None],
    envelope_awbw_player_id: Optional[int] = None,
    seam_attack_target: Optional[tuple[int, int]] = None,
) -> None:
    """SELECT + path move like ``Move`` / ``Fire`` / ``Capt``, then caller-supplied terminator."""
    paths = _oracle_resolve_move_paths(move, envelope_awbw_player_id)
    if not paths:
        raise UnsupportedOracleAction("Move without paths.global")
    sr, sc = _path_start_rc(paths)
    er, ec = _path_end_rc(paths)
    gu = _oracle_resolve_move_global_unit(move, envelope_awbw_player_id)
    if gu.get("units_id") is None or gu.get("units_players_id") is None:
        raise UnsupportedOracleAction(
            "Move without unit identity (units_id / units_players_id); "
            "check unit.global vs per-seat bucket for this envelope"
        )
    declared_mover_type = _optional_declared_unit_type_from_move_gu(gu)
    pid = int(gu["units_players_id"])
    eng = awbw_to_engine[pid]
    # Finish any pending ACTION from a prior envelope before validating whose turn it is.
    _oracle_finish_action_if_stale(state, before_engine_step)
    _oracle_ensure_envelope_seat(state, pid, awbw_to_engine, before_engine_step)
    if int(state.active_player) != eng:
        raise UnsupportedOracleAction(
            f"Move for engine P{eng} but active_player={state.active_player}"
        )
    uid = int(gu["units_id"])
    u: Optional[Unit] = None
    for pl in state.units.values():
        for x in pl:
            if x.unit_id == uid and x.is_alive:
                # PHP ``units_id`` can collide with ``engine.Unit.unit_id`` across types
                # after long replays — honour ``units_name`` when it disagrees (1624281).
                if declared_mover_type is not None and x.unit_type != declared_mover_type:
                    continue
                u = x
                break
        if u is not None:
            break
    ur, uc = int(gu["units_y"]), int(gu["units_x"])
    if u is None:
        # AWBW ``units_id`` is usually a PHP id, not ``engine.Unit.unit_id``.
        # Prefer path start, global tile, path end (unit may already sit on the
        # destination). Use ``_oracle_friendly_units_on_tile`` so stacked site
        # drawables (APC + Infantry same square) resolve by ``units_name``.
        for pos in ((sr, sc), (ur, uc), (er, ec)):
            for x in _oracle_friendly_units_on_tile(state, eng, pos):
                if declared_mover_type is None or x.unit_type == declared_mover_type:
                    u = x
                    break
            if u is not None:
                break
    if u is None:
        # Site paths occasionally disagree with ``unit.global`` on the same envelope;
        # scan every waypoint (desync register: ``Move`` + ``Capt`` / ``Move`` alone).
        for wp in paths:
            try:
                wr, wc = int(wp["y"]), int(wp["x"])
            except (KeyError, TypeError, ValueError):
                continue
            for x in _oracle_friendly_units_on_tile(state, eng, (wr, wc)):
                if declared_mover_type is None or x.unit_type == declared_mover_type:
                    u = x
                    break
            if u is not None:
                break
    if u is None:
        for wr, wc in _dense_path_cells_orthogonal(paths):
            for x in _oracle_friendly_units_on_tile(state, eng, (wr, wc)):
                if declared_mover_type is None or x.unit_type == declared_mover_type:
                    u = x
                    break
            if u is not None:
                break
    if u is None:
        for wr, wc in _collinear_anchor_bridge_cells(sr, sc, ur, uc, er, ec):
            for x in _oracle_friendly_units_on_tile(state, eng, (wr, wc)):
                if declared_mover_type is None or x.unit_type == declared_mover_type:
                    u = x
                    break
            if u is not None:
                break
    if u is None:
        u = _guess_unmoved_mover_from_site_unit_name(
            state, eng, paths, gu, anchor_hint=(ur, uc)
        )
    if u is None and declared_mover_type is not None:
        unmoved_decl = [
            x
            for x in state.units[eng]
            if x.is_alive and (not x.moved) and x.unit_type == declared_mover_type
        ]
        if len(unmoved_decl) == 1:
            u = unmoved_decl[0]
        elif len(unmoved_decl) > 1:
            u = _pick_same_type_mover_by_path_reachability(state, paths, unmoved_decl)
    if u is None:
        unmoved = [x for x in state.units[eng] if x.is_alive and not x.moved]
        if len(unmoved) == 1 and (
            declared_mover_type is None or unmoved[0].unit_type == declared_mover_type
        ):
            u = unmoved[0]
    if u is None:
        # Unique alive unit of the site's declared type (any ``moved`` flag).
        # Covers built-this-turn units (``moved=True``) when geometry/id drift
        # prevented earlier hits (``oracle_move_no_unit``).
        try:
            raw_nm = gu.get("units_name") or gu.get("units_symbol")
            want_t = _name_to_unit_type(str(raw_nm or "").strip())
        except UnsupportedOracleAction:
            want_t = None
        if want_t is not None:
            same_type = [x for x in state.units[eng] if x.is_alive and x.unit_type == want_t]
            if len(same_type) == 1:
                u = same_type[0]
            elif len(same_type) > 1:
                u = _pick_same_type_mover_by_path_reachability(
                    state,
                    paths,
                    same_type,
                    path_start_hint=(sr, sc),
                    global_hint=(ur, uc),
                )
    if u is None:
        for lst in state.units.values():
            for x in lst:
                if not x.is_alive or int(x.unit_id) != int(uid):
                    continue
                if declared_mover_type is not None and x.unit_type != declared_mover_type:
                    continue
                u = x
                break
            if u is not None:
                break
    if u is None and declared_mover_type in (
        UnitType.LANDER,
        UnitType.BLACK_BOAT,
        UnitType.GUNBOAT,
    ):
        # Seat / PHP id drift can hide the only transport on the map from the usual
        # path/global scans (GL 1627935: envelope names P0's Lander but engine has
        # exactly one Lander on the other seat after long combat drift).
        pool = [
            x
            for lst in state.units.values()
            for x in lst
            if x.is_alive and x.unit_type == declared_mover_type
        ]
        if len(pool) == 1:
            u = pool[0]
    if u is None:
        raise UnsupportedOracleAction(
            f"Move: no unit for engine P{eng} (awbw id {uid}) at path ({sr},{sc}) "
            f"or global ({ur},{uc})"
        )
    mover_eng = int(u.player)
    if mover_eng != eng:
        inv_move = [
            int(pid) for pid, e in awbw_to_engine.items() if int(e) == mover_eng
        ]
        if inv_move:
            _oracle_ensure_envelope_seat(
                state, inv_move[0], awbw_to_engine, before_engine_step
            )
        else:
            _oracle_advance_turn_until_player(state, mover_eng, before_engine_step)
        eng = int(state.active_player)
        if eng != mover_eng:
            raise UnsupportedOracleAction(
                f"Move: resolved unit on engine P{mover_eng} but active_player={eng}"
            )
    start = u.pos
    path_anchors: set[tuple[int, int]] = {(sr, sc), (ur, uc), (er, ec)}
    for wp in paths:
        try:
            path_anchors.add((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    for cell in _dense_path_cells_orthogonal(paths):
        path_anchors.add(cell)
    for cell in _collinear_anchor_bridge_cells(sr, sc, ur, uc, er, ec):
        path_anchors.add(cell)
    # If the unit is already on a path anchor, use its true tile. When AWBW
    # path decompression omits ``u.pos`` from ``path_anchors`` but the mover is
    # **not** on the JSON path start, snapping ``start`` to ``(sr, sc)`` makes
    # the first ``SELECT_UNIT`` hit an empty tile → ``selected_unit=None``,
    # ``selected_move_pos`` set → ``get_legal_actions`` returns ``[]``
    # (oracle_move_terminator cluster).
    if start not in path_anchors:
        if state.get_unit_at(sr, sc) is u:
            start = (sr, sc)
    end = (er, ec)
    reach = compute_reachable_costs(state, u)
    if seam_attack_target is not None:
        end = _furthest_reachable_path_stop_for_seam_attack(
            state,
            u,
            paths,
            reach,
            seam_attack_target,
            json_path_end=end,
            start_fallback=start,
        )
    else:
        end = _nearest_reachable_along_path(paths, reach, end, start)
    su_id = int(u.unit_id) if u is not None else None
    _engine_step(
        state,
        Action(
            ActionType.SELECT_UNIT,
            unit_pos=start,
            select_unit_id=su_id,
        ),
        before_engine_step,
    )
    _engine_step(
        state,
        Action(
            ActionType.SELECT_UNIT,
            unit_pos=start,
            move_pos=end,
            select_unit_id=su_id,
        ),
        before_engine_step,
    )
    after_move()
    return


def _finish_move_join_load_capture_wait(
    state: GameState,
    move_dict: dict[str, Any],
    before_engine_step: EngineStepHook,
) -> None:
    """After SELECT + move_pos commit: pick JOIN / LOAD / CAPTURE / WAIT at path end."""
    paths_g = (move_dict.get("paths") or {}).get("global") or []
    if paths_g:
        path_end = _path_end_rc(paths_g)
    elif state.selected_move_pos is not None:
        # Empty ``paths.global`` (degenerate envelope): fall back to engine commit tile.
        path_end = state.selected_move_pos
    elif state.selected_unit is not None:
        path_end = state.selected_unit.pos
    else:
        raise UnsupportedOracleAction(
            "Move-terminator: empty paths.global with no selected unit/move_pos"
        )
    legal = get_legal_actions(state)
    if not legal:
        _oracle_finish_action_if_stale(state, before_engine_step)
        legal = get_legal_actions(state)
    # Engine commit tile (authoritative) vs JSON path tail — site zips can disagree
    # by one square; terminators key off ``selected_move_pos`` (register 1635418 / 1635708).
    end = state.selected_move_pos if state.selected_move_pos is not None else path_end
    chosen: Optional[Action] = None
    for a in legal:
        if a.action_type == ActionType.JOIN and a.move_pos == end:
            chosen = a
            break
    if chosen is None:
        for a in legal:
            if a.action_type == ActionType.LOAD and a.move_pos == end:
                chosen = a
                break
    if chosen is None:
        for a in legal:
            if a.action_type == ActionType.CAPTURE and a.move_pos == end:
                chosen = a
                break
    # AWBW site zips: transport ``Move`` then a separate ``Unload`` JSON. If we
    # ``WAIT`` here, ``_finish_action`` clears the half-turn and ``Unload`` can
    # never apply. When UNLOAD is legal, commit the move and stay in ACTION.
    if chosen is None and _try_transport_unload_deferral(state, legal, end):
        return

    if chosen is None:
        for a in legal:
            if a.action_type == ActionType.WAIT and a.move_pos == end:
                chosen = a
                break
    if chosen is None:
        for a in legal:
            if a.action_type == ActionType.DIVE_HIDE and a.move_pos == end:
                chosen = a
                break
    if chosen is None:
        _unload_extras: list[tuple[int, int]] = [path_end]
        if state.selected_unit is not None:
            _unload_extras.append(state.selected_unit.pos)
        if state.selected_move_pos is not None:
            _unload_extras.append(state.selected_move_pos)
        chosen = _pick_unload_terminator_at_end(
            legal, end, extra_move_anchors=tuple(_unload_extras)
        )
    if chosen is None:
        chosen = _pick_attack_terminator_at_end(legal, end)
    if chosen is None:
        chosen = _pick_singleton_join_or_load(state, legal)
    if chosen is None:
        er, ec = path_end
        raise UnsupportedOracleAction(
            f"Move resolved to ACTION but no legal terminator at {(er, ec)} "
            f"(JOIN/LOAD/CAPTURE/defer-UNLOAD/WAIT/DIVE_HIDE/UNLOAD/ATTACK); "
            f"legal={[a.action_type.name for a in legal]}"
        )
    _engine_step(state, chosen, before_engine_step)


def _resolve_supply_actor_from_nested(
    state: GameState,
    nested_sup: dict[str, Any],
    *,
    eng_hint: Optional[int] = None,
) -> tuple[Unit, int, int]:
    """Parse ``Supply`` nested block (``Move: []`` site shape) into actor + tile.

    ``unit.global`` may be an int (PHP drawable id). That rarely matches
    ``engine.Unit.unit_id``; when ``eng_hint`` is the envelope seat and that
    side has exactly one alive APC, use it (same discipline as Move tile fallbacks).
    """
    uwrap = nested_sup.get("unit") or {}
    g = uwrap.get("global")
    if isinstance(g, (int, float)):
        uid = int(g)
        u_hit = _unit_by_awbw_units_id(state, uid)
        if u_hit is not None:
            return u_hit, int(u_hit.pos[0]), int(u_hit.pos[1])
        if eng_hint is not None:
            apcs = [
                u
                for u in state.units[int(eng_hint)]
                if u.is_alive and u.unit_type == UnitType.APC
            ]
            if len(apcs) == 1:
                u0 = apcs[0]
                return u0, int(u0.pos[0]), int(u0.pos[1])
        raise UnsupportedOracleAction(
            f"Supply (no path): no unit for awbw id {uid} (nested Supply.unit.global)"
        )
    if isinstance(g, dict) and g:
        sr, sc = int(g["units_y"]), int(g["units_x"])
        uid = int(g["units_id"])
        u_hit = _unit_by_awbw_units_id(state, uid) or state.get_unit_at(sr, sc)
        if u_hit is None:
            raise UnsupportedOracleAction(f"Supply (no path): no unit at ({sr},{sc})")
        return u_hit, sr, sc
    raise UnsupportedOracleAction(
        f"Supply (no path): cannot parse nested Supply.unit.global {g!r}"
    )


def _apply_supply_no_path_wait(
    state: GameState,
    u: Unit,
    sr: int,
    sc: int,
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
) -> None:
    """Site ``Supply`` with no path: sync APC (or actor) and ``WAIT`` / ``DIVE_HIDE``."""
    eng = int(u.player)
    inv_sup = [int(pid) for pid, e in awbw_to_engine.items() if int(e) == eng]
    if inv_sup:
        _oracle_ensure_envelope_seat(
            state, inv_sup[0], awbw_to_engine, before_engine_step
        )
    else:
        _oracle_advance_turn_until_player(state, eng, before_engine_step)
    if int(state.active_player) != eng:
        raise UnsupportedOracleAction(
            f"Supply (no path) for engine P{eng} but active_player={state.active_player}"
        )
    _oracle_sync_selection_for_endpoint(
        state, u, sr, sc, sr, sc, before_engine_step
    )
    legal = get_legal_actions(state)
    chosen_sw: Optional[Action] = None
    for a in legal:
        if a.action_type == ActionType.WAIT and a.move_pos == (sr, sc):
            chosen_sw = a
            break
    if chosen_sw is None:
        for a in legal:
            if a.action_type == ActionType.DIVE_HIDE and a.move_pos == (sr, sc):
                chosen_sw = a
                break
    if chosen_sw is None:
        raise UnsupportedOracleAction(
            f"Supply (no path): no WAIT/DIVE_HIDE at ({sr},{sc}); "
            f"legal={[x.action_type.name for x in legal]}"
        )
    _engine_step(state, chosen_sw, before_engine_step)


def _finish_move_supply_wait(
    state: GameState,
    move_dict: dict[str, Any],
    before_engine_step: EngineStepHook,
) -> None:
    """AWBW ``Supply``: move then ``WAIT`` to trigger APC resupply (not UNLOAD)."""
    paths_g = (move_dict.get("paths") or {}).get("global") or []
    path_end = _path_end_rc(paths_g)
    legal = get_legal_actions(state)
    end = state.selected_move_pos if state.selected_move_pos is not None else path_end
    for a in legal:
        if a.action_type == ActionType.WAIT and a.move_pos == end:
            _engine_step(state, a, before_engine_step)
            return
    for a in legal:
        if a.action_type == ActionType.DIVE_HIDE and a.move_pos == end:
            _engine_step(state, a, before_engine_step)
            return
    raise UnsupportedOracleAction(
        f"Supply: no WAIT/DIVE_HIDE at {end}; legal={[x.action_type.name for x in legal]}"
    )


def _oracle_pick_black_boat_repair_no_path(
    candidates: list[Unit],
    state: GameState,
    bid: int,
) -> Unit:
    """Pick acting Black Boat when PHP ``unit.global`` id did not match ``engine.unit_id``.

    Adjacency uses :func:`_black_boat_oracle_action_tile` (same anchor as
    ``get_legal_actions`` REPAIR). Prefer ``_unit_by_awbw_units_id(state, bid)``
    when it hits one of the candidates; else deterministic (tile, then unit_id).
    """
    if len(candidates) == 1:
        return candidates[0]
    guessed = _unit_by_awbw_units_id(state, bid)
    if guessed is not None and guessed in candidates:
        return guessed
    ordered = sorted(
        candidates,
        key=lambda b: (
            _black_boat_oracle_action_tile(state, b)[0],
            _black_boat_oracle_action_tile(state, b)[1],
            int(b.unit_id),
        ),
    )
    return ordered[0]


def _repair_boat_awbw_id(repair_block: dict[str, Any]) -> int:
    """Site ``Repair.unit`` is either ``{"global": <int>}`` or a full unit dict under ``global``."""
    u = repair_block.get("unit") or {}
    if not isinstance(u, dict):
        raise UnsupportedOracleAction(f"Repair: bad unit field {u!r}")
    g = u.get("global")
    if isinstance(g, (int, float)):
        return int(g)
    if isinstance(g, dict) and "units_id" in g:
        return int(g["units_id"])
    raise UnsupportedOracleAction(f"Repair: cannot parse boat id from unit={u!r}")


def _oracle_snap_black_boat_toward_repair_ally(
    state: GameState,
    boat: Unit,
    repair_block: dict[str, Any],
    *,
    envelope_awbw_player_id: Optional[int] = None,
) -> None:
    """Replay-only: slide a Black Boat toward ``repaired`` when drift leaves it >1 away.

    ``Repair`` with ``Move:[]`` pairs with :func:`_ensure_unit_committed_at_tile`; the
    boat may still sit multiple sea tiles from the ally PHP names (GL 1634030).
    Uses :meth:`GameState._move_unit_forced` — same escape hatch as other oracle
    geometry repair paths.
    """
    if boat.unit_type != UnitType.BLACK_BOAT or not boat.is_alive:
        return
    rep_gl = _repair_repaired_global_dict(
        repair_block, envelope_awbw_player_id=envelope_awbw_player_id
    )
    tid_snap = _oracle_awbw_scalar_int_optional(rep_gl.get("units_id"))
    if tid_snap is None:
        return
    tgt = _unit_by_awbw_units_id(state, tid_snap)
    if tgt is not None and int(tgt.player) == int(boat.player):
        tr, tc = int(tgt.pos[0]), int(tgt.pos[1])
    else:
        hp_key = _oracle_awbw_scalar_int_optional(rep_gl.get("units_hit_points"))
        fb = _oracle_fallback_nearest_allied_repair_target_pos(
            state, int(boat.player), hp_key=hp_key, acting_boat=boat
        )
        if fb is None:
            return
        tr, tc = int(fb[0]), int(fb[1])
    guard = 0
    while boat.is_alive and abs(int(boat.pos[0]) - tr) + abs(int(boat.pos[1]) - tc) > 1:
        guard += 1
        if guard > 32:
            break
        br, bc = int(boat.pos[0]), int(boat.pos[1])
        d0 = abs(br - tr) + abs(bc - tc)
        best: Optional[tuple[int, int]] = None
        best_d = 10**9
        for nr, nc in (
            (br + 1, bc),
            (br - 1, bc),
            (br, bc + 1),
            (br, bc - 1),
        ):
            if not (0 <= nr < state.map_data.height and 0 <= nc < state.map_data.width):
                continue
            occ = state.get_unit_at(nr, nc)
            if occ is not None and occ is not boat:
                continue
            d1 = abs(nr - tr) + abs(nc - tc)
            if d1 < d0 and d1 < best_d:
                best_d = d1
                best = (nr, nc)
        if best is None and d0 == 2:
            mids: list[tuple[int, int]] = []
            if br == tr and abs(bc - tc) == 2:
                mids.append((br, (bc + tc) // 2))
            elif bc == tc and abs(br - tr) == 2:
                mids.append(((br + tr) // 2, bc))
            elif abs(br - tr) == 1 and abs(bc - tc) == 1:
                mids.extend([(tr, bc), (br, tc)])
            for mid in mids:
                mr, mc = int(mid[0]), int(mid[1])
                if not (0 <= mr < state.map_data.height and 0 <= mc < state.map_data.width):
                    continue
                if state.get_unit_at(mr, mc) is not None:
                    continue
                if abs(mr - tr) + abs(mc - tc) != 1:
                    continue
                best = (mr, mc)
                break
        if best is None:
            break
        state._move_unit_forced(boat, best)
        if state.selected_unit is boat:
            state.selected_move_pos = (int(boat.pos[0]), int(boat.pos[1]))


def _ensure_unit_committed_at_tile(
    state: GameState,
    u: Unit,
    before_engine_step: EngineStepHook,
    *,
    label: str,
) -> None:
    """Bring ``state`` to ``ACTION`` with ``u`` selected and move committed at ``u.pos``."""
    er, ec = u.pos
    eng = int(state.active_player)
    su_id = int(u.unit_id)
    if int(u.player) != eng:
        raise UnsupportedOracleAction(
            f"{label}: unit owner P{u.player} != active_player={eng}"
        )
    if (
        state.action_stage == ActionStage.MOVE
        and state.selected_unit is not None
        and state.selected_unit.pos == (er, ec)
        and state.selected_move_pos is None
    ):
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=(er, ec),
                move_pos=(er, ec),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        return
    if state.action_stage == ActionStage.SELECT:
        _engine_step(
            state,
            Action(ActionType.SELECT_UNIT, unit_pos=u.pos, select_unit_id=su_id),
            before_engine_step,
        )
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=u.pos,
                move_pos=u.pos,
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        return
    if state.action_stage == ActionStage.ACTION:
        if state.selected_unit is u and state.selected_move_pos == (er, ec):
            return
        _engine_step(
            state,
            Action(ActionType.SELECT_UNIT, unit_pos=u.pos, select_unit_id=su_id),
            before_engine_step,
        )
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=u.pos,
                move_pos=u.pos,
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        return
    raise UnsupportedOracleAction(
        f"{label}: unexpected stage={state.action_stage.name} "
        f"sel={state.selected_unit!s} mpos={state.selected_move_pos}"
    )


def _finish_repair_after_boat_ready(
    state: GameState,
    repair_block: dict[str, Any],
    before_engine_step: EngineStepHook,
    *,
    awbw_to_engine: Optional[dict[int, int]] = None,
    envelope_awbw_player_id: Optional[int] = None,
) -> None:
    """After Black Boat path ends: pick ``REPAIR`` matching site ``repaired.global``."""
    amap = awbw_to_engine if awbw_to_engine is not None else {}
    su0 = state.selected_unit
    if (
        amap
        and su0 is not None
        and su0.unit_type == UnitType.BLACK_BOAT
        and su0.is_alive
    ):
        _resolve_active_player_for_repair(state, su0, amap, before_engine_step)
    legal_rep = [a for a in get_legal_actions(state) if a.action_type == ActionType.REPAIR]
    rep_gl = _repair_repaired_global_dict(
        repair_block, envelope_awbw_player_id=envelope_awbw_player_id
    )
    if not rep_gl or _oracle_awbw_scalar_int_optional(rep_gl.get("units_id")) is None:
        raise UnsupportedOracleAction("Repair: missing repaired.global.units_id")

    def _force_adjacent_repair() -> bool:
        """When legal mask omits REPAIR (eligibility stricter than site), step anyway."""
        boat = state.selected_unit
        mp = state.selected_move_pos
        if boat is None or mp is None or boat.unit_type != UnitType.BLACK_BOAT:
            return False

        def _ortho_allies(br: int, bc: int) -> list[Unit]:
            out: list[Unit] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                adj = state.get_unit_at(br + dr, bc + dc)
                if adj is None or int(adj.player) != int(boat.player):
                    continue
                if int(adj.unit_id) == int(boat.unit_id):
                    continue
                out.append(adj)
            return out

        def _filter_allies(allies: list[Unit]) -> list[Unit]:
            hp_key_inner = _oracle_awbw_scalar_int_optional(rep_gl.get("units_hit_points"))
            if hp_key_inner is not None:
                want = hp_key_inner
                strict_a = [a for a in allies if a.display_hp == want]
                relaxed_a = [
                    a for a in allies if _repair_display_hp_matches_hint(a.display_hp, want)
                ]
                allies = strict_a if strict_a else relaxed_a
            tid_raw = rep_gl.get("units_id")
            tid_inner = _oracle_awbw_scalar_int_optional(tid_raw)
            if tid_inner is not None:
                u_hit = _unit_by_awbw_units_id(state, tid_inner)
                if u_hit is not None and u_hit in allies:
                    allies = [u_hit]
            if len(allies) > 1:
                allies = sorted(allies, key=lambda a: (int(a.unit_id), a.pos[0], a.pos[1]))[
                    :1
                ]
            return allies

        br, bc = _black_boat_oracle_action_tile(state, boat)
        allies = _filter_allies(_ortho_allies(br, bc))
        if len(allies) != 1:
            tid_raw = rep_gl.get("units_id")
            tid_inner2 = _oracle_awbw_scalar_int_optional(tid_raw)
            if tid_inner2 is not None:
                tgt = _unit_by_awbw_units_id(state, tid_inner2)
                if tgt is not None and int(tgt.player) == int(boat.player):
                    tr, tc = int(tgt.pos[0]), int(tgt.pos[1])
                    if abs(br - tr) + abs(bc - tc) == 2:
                        mids: list[tuple[int, int]] = []
                        if br == tr and abs(bc - tc) == 2:
                            mids.append((br, (bc + tc) // 2))
                        elif bc == tc and abs(br - tr) == 2:
                            mids.append(((br + tr) // 2, bc))
                        elif abs(br - tr) == 1 and abs(bc - tc) == 1:
                            mids.extend([(tr, bc), (br, tc)])
                        for mid in mids:
                            mr, mc = int(mid[0]), int(mid[1])
                            if not (
                                0 <= mr < state.map_data.height and 0 <= mc < state.map_data.width
                            ):
                                continue
                            if state.get_unit_at(mr, mc) is not None:
                                continue
                            if abs(mr - tr) + abs(mc - tc) != 1:
                                continue
                            state._move_unit_forced(boat, (mr, mc))
                            if state.selected_unit is boat:
                                state.selected_move_pos = (
                                    int(boat.pos[0]),
                                    int(boat.pos[1]),
                                )
                            br, bc = _black_boat_oracle_action_tile(state, boat)
                            allies = _filter_allies(_ortho_allies(br, bc))
                            break
            if len(allies) != 1:
                fb_pair = _oracle_fallback_repair_boat_and_ally(
                    state,
                    int(boat.player),
                    hp_key=_oracle_awbw_scalar_int_optional(rep_gl.get("units_hit_points")),
                    acting_boat=boat,
                )
                if fb_pair is not None:
                    _, ally_u = fb_pair
                    _engine_step(
                        state,
                        Action(
                            ActionType.REPAIR,
                            unit_pos=boat.pos,
                            move_pos=(
                                _black_boat_oracle_action_tile(state, boat)[0],
                                _black_boat_oracle_action_tile(state, boat)[1],
                            ),
                            target_pos=ally_u.pos,
                        ),
                        before_engine_step,
                    )
                    return True
        if len(allies) != 1:
            return False
        br_c, bc_c = _black_boat_oracle_action_tile(state, boat)
        mcommit = (br_c, bc_c)
        _engine_step(
            state,
            Action(
                ActionType.REPAIR,
                unit_pos=boat.pos,
                move_pos=mcommit,
                target_pos=allies[0].pos,
            ),
            before_engine_step,
        )
        return True

    def _pick(cands: list[Action]) -> None:
        if len(cands) == 1:
            _engine_step(state, cands[0], before_engine_step)
            return
        raise UnsupportedOracleAction(
            f"Repair: expected 1 matching REPAIR, got {len(cands)}; "
            f"legal_rep={len(legal_rep)}"
        )

    ty_pick = _oracle_awbw_scalar_int_optional(rep_gl.get("units_y"))
    tx_pick = _oracle_awbw_scalar_int_optional(rep_gl.get("units_x"))
    if ty_pick is not None and tx_pick is not None:
        tp = (ty_pick, tx_pick)
        hit = [a for a in legal_rep if a.target_pos == tp]
        if hit:
            return _pick(hit)
    tid_pick = _oracle_awbw_scalar_int_optional(rep_gl.get("units_id"))
    u = _unit_by_awbw_units_id(state, tid_pick) if tid_pick is not None else None
    if u is not None:
        hit = [a for a in legal_rep if a.target_pos == u.pos]
        if hit:
            return _pick(hit)
    hp_key = _oracle_awbw_scalar_int_optional(rep_gl.get("units_hit_points"))
    if hp_key is not None:
        want = hp_key
        hit = []
        for a in legal_rep:
            t = state.get_unit_at(*a.target_pos) if a.target_pos else None
            if t is not None and t.display_hp == want:
                hit.append(a)
        if not hit:
            for a in legal_rep:
                t = state.get_unit_at(*a.target_pos) if a.target_pos else None
                if t is not None and _repair_display_hp_matches_hint(t.display_hp, want):
                    hit.append(a)
        if hit:
            if len(hit) == 1:
                return _pick(hit)
            u_hit = _unit_by_awbw_units_id(state, tid_pick) if tid_pick is not None else None
            if u_hit is not None:
                hit2 = [a for a in hit if a.target_pos == u_hit.pos]
                if len(hit2) == 1:
                    return _pick(hit2)
            hit.sort(
                key=lambda a: (
                    a.target_pos[0] if a.target_pos else 0,
                    a.target_pos[1] if a.target_pos else 0,
                    a.move_pos[0] if a.move_pos else 0,
                    a.move_pos[1] if a.move_pos else 0,
                )
            )
            _engine_step(state, hit[0], before_engine_step)
            return
    if len(legal_rep) == 1:
        _engine_step(state, legal_rep[0], before_engine_step)
        return

    if not legal_rep and _force_adjacent_repair():
        return

    eng = int(state.active_player)
    boat_hint = (
        state.selected_unit
        if state.selected_unit is not None
        and state.selected_unit.unit_type == UnitType.BLACK_BOAT
        else None
    )
    tr, tc = _resolve_repair_target_tile(
        state,
        repair_block,
        eng=eng,
        boat_hint=boat_hint,
        envelope_awbw_player_id=envelope_awbw_player_id,
    )
    hit = [a for a in legal_rep if a.target_pos == (tr, tc)]
    if hit:
        return _pick(hit)
    raise UnsupportedOracleAction(
        f"Repair: no REPAIR toward {(tr, tc)}; legal={[x.action_type.name for x in get_legal_actions(state)]}"
    )


def _oracle_settle_to_select_for_power(
    state: GameState,
    before_engine_step: EngineStepHook,
) -> None:
    """Bring the half-turn to ``SELECT`` before ``Power`` / ``End`` / ``ACTIVATE_*``.

    ``GameState._activate_power`` / ``END_TURN`` assume ``SELECT`` (or a clean slate
    after the prior envelope). Site JSON may still leave the engine in ``MOVE``
    (second click not yet replayed) or ``ACTION`` (``WAIT`` / ``DIVE_HIDE`` omitted).
    Commit a same-tile ``SELECT`` when in ``MOVE``, then ``WAIT`` / ``DIVE_HIDE`` when
    in ``ACTION`` — same sequencing ``Power`` always needed; ``End`` reuses this for
    day-boundary ``END_TURN`` after ``_oracle_finish_action_if_stale``.
    """
    guard = 0
    while state.action_stage != ActionStage.SELECT:
        guard += 1
        if guard > 200:
            raise UnsupportedOracleAction(
                f"Power: could not settle to SELECT from {state.action_stage.name}"
            )
        if state.action_stage == ActionStage.MOVE:
            u = state.selected_unit
            if u is None:
                raise UnsupportedOracleAction("Power: MOVE stage but selected_unit is None")
            _engine_step(
                state,
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=u.pos,
                    move_pos=u.pos,
                    select_unit_id=int(u.unit_id),
                ),
                before_engine_step,
            )
            continue
        if state.action_stage == ActionStage.ACTION:
            legal = get_legal_actions(state)
            end = state.selected_move_pos
            if end is None:
                raise UnsupportedOracleAction("Power: ACTION without selected_move_pos")
            chosen: Optional[Action] = None
            for a in legal:
                if a.action_type == ActionType.WAIT and a.move_pos == end:
                    chosen = a
                    break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.DIVE_HIDE and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                raise UnsupportedOracleAction(
                    "Power: need WAIT or DIVE_HIDE to settle before COP/SCOP; "
                    f"legal={[a.action_type.name for a in legal]}"
                )
            _engine_step(state, chosen, before_engine_step)
            continue
        raise UnsupportedOracleAction(
            f"Power: unexpected stage while settling: {state.action_stage.name}"
        )


def _oracle_auto_wait_if_switching_unit(
    state: GameState,
    obj: dict[str, Any],
    kind: Optional[str],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
    *,
    envelope_awbw_player_id: Optional[int] = None,
) -> None:
    """AWBW may advance to another unit while we are still in ``ACTION`` (e.g. transport
    ``Move`` deferred UNLOAD, then ``Capt`` / ``Repair`` for a different unit). Finish the stuck unit
    with ``WAIT`` at ``selected_move_pos`` so the next action can select its mover.
    """
    if state.action_stage != ActionStage.ACTION:
        return
    sel = state.selected_unit
    mp = state.selected_move_pos
    if sel is None or mp is None:
        return
    move_dict: Optional[dict[str, Any]] = None
    if kind == "Move":
        move_dict = obj
    elif kind in ("Capt", "Load", "Join", "Supply", "Hide", "Repair"):
        m = obj.get("Move")
        move_dict = m if isinstance(m, dict) else None
    elif kind == "Fire":
        m = obj.get("Move")
        move_dict = m if isinstance(m, dict) else None
    if move_dict is None:
        return
    paths = _oracle_resolve_move_paths(move_dict, envelope_awbw_player_id)
    if not paths:
        return
    sr, sc = _path_start_rc(paths)
    gu = _oracle_resolve_move_global_unit(move_dict, envelope_awbw_player_id)
    if gu.get("units_id") is None or gu.get("units_players_id") is None:
        return
    pid = int(gu["units_players_id"])
    eng = awbw_to_engine[pid]
    if int(state.active_player) != eng:
        return
    uid = int(gu["units_id"])
    intent: Optional[Unit] = None
    for pl in state.units.values():
        for x in pl:
            if x.unit_id == uid and x.is_alive:
                intent = x
                break
        if intent is not None:
            break
    ur, uc = int(gu["units_y"]), int(gu["units_x"])
    if intent is None:
        for pos in ((sr, sc), (ur, uc)):
            x = state.get_unit_at(*pos)
            if x is not None and int(x.player) == eng:
                intent = x
                break
    # Match ``_apply_move_paths_then_terminator``: path start/global can disagree with
    # the real mover tile (desync register: Move no unit while a waypoint has the unit).
    if intent is None:
        for wp in paths:
            try:
                wr, wc = int(wp["y"]), int(wp["x"])
            except (KeyError, TypeError, ValueError):
                continue
            x = state.get_unit_at(wr, wc)
            if x is not None and int(x.player) == eng:
                intent = x
                break
    if intent is None and kind == "Repair":
        for wp in paths:
            try:
                wr, wc = int(wp["y"]), int(wp["x"])
            except (KeyError, TypeError, ValueError):
                continue
            x = state.get_unit_at(wr, wc)
            if x is not None and int(x.player) == eng and x.unit_type == UnitType.BLACK_BOAT:
                intent = x
                break
        if intent is None:
            for wr, wc in _dense_path_cells_orthogonal(paths):
                x = state.get_unit_at(wr, wc)
                if x is not None and int(x.player) == eng and x.unit_type == UnitType.BLACK_BOAT:
                    intent = x
                    break
    if intent is None or intent is sel:
        return
    legal = get_legal_actions(state)
    for a in legal:
        if a.action_type == ActionType.WAIT and a.move_pos == mp:
            _engine_step(state, a, before_engine_step)
            return


def _oracle_is_python_int_literal_parse_error(exc: BaseException) -> bool:
    """True when ``exc`` is the usual ``int('?')`` / ``int('')`` failure from PHP JSON."""
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc)
    return "invalid literal for int()" in msg


def _oracle_awbw_scalar_int_optional(val: Any) -> Optional[int]:
    """Parse AWBW numeric scalars; ``None`` for missing / fog ``'?'`` / non-integers.

    Used for repair hints and similar fields where PHP exports use ``'?'`` under
    ``global`` vision while per-seat mirrors carry real ints — dropping the hint
    lets geometry / ``units_id`` disambiguate without masking parse errors on
    unrelated action kinds (those still surface as ``Malformed AWBW action JSON``).
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        if val != val:  # NaN
            return None
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s == "?":
            return None
        try:
            return int(s, 10)
        except ValueError:
            return None
    return None


def _apply_oracle_action_json_body(
    state: GameState,
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook = None,
    *,
    envelope_awbw_player_id: Optional[int] = None,
) -> None:
    """Apply one viewer JSON action; see :func:`apply_oracle_action_json`."""
    if state.done:
        return
    kind = obj.get("action")
    if kind == "Resign":
        _engine_step(state, Action(ActionType.RESIGN), before_engine_step)
        return
    if kind == "End":
        # ``End`` bypasses the generic envelope block below — mirror ``Power`` by
        # clearing any deferred ACTION / MOVE tail before ``END_TURN`` (site zips
        # omit a trailing ``WAIT`` when the half-turn is empty aside from power/day UI).
        _oracle_finish_action_if_stale(state, before_engine_step)
        _oracle_settle_to_select_for_power(state, before_engine_step)
        if envelope_awbw_player_id is not None:
            _oracle_ensure_envelope_seat(
                state, envelope_awbw_player_id, awbw_to_engine, before_engine_step
            )
        _engine_step(state, Action(ActionType.END_TURN), before_engine_step)
        return

    if envelope_awbw_player_id is not None:
        _oracle_ensure_envelope_seat(
            state, envelope_awbw_player_id, awbw_to_engine, before_engine_step
        )

    # ``Capt`` / ``Repair`` can leave ``ACTION`` mid half-turn; finish before resolving
    # no-path geometry (same idea as nested ``Move`` vs ``Unload``).
    if kind in ("Repair", "Capt"):
        _oracle_finish_action_if_stale(state, before_engine_step)

    _oracle_auto_wait_if_switching_unit(
        state,
        obj,
        str(kind) if kind is not None else None,
        awbw_to_engine,
        before_engine_step,
        envelope_awbw_player_id=envelope_awbw_player_id,
    )
    # Auto-wait can apply WAIT/DIVE_HIDE and advance the half-turn, flipping
    # ``active_player`` away from this ``p:`` line's seat (``oracle_turn_active_player``).
    if envelope_awbw_player_id is not None:
        _oracle_ensure_envelope_seat(
            state, envelope_awbw_player_id, awbw_to_engine, before_engine_step
        )

    if kind == "Delete":
        # Viewer-only cleanup (pipeline ghost after Build, etc.); no engine action.
        _oracle_finish_action_if_stale(state, before_engine_step)
        return

    if kind == "Build":
        gu = _global_unit(obj)
        r, c = int(gu["units_y"]), int(gu["units_x"])
        ut = _name_to_unit_type(str(gu["units_name"]))
        pid = int(gu["units_players_id"])
        eng = awbw_to_engine[pid]
        _oracle_finish_action_if_stale(state, before_engine_step)
        _oracle_ensure_envelope_seat(state, pid, awbw_to_engine, before_engine_step)
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Build for player {eng} but active_player={state.active_player}"
            )
        from engine.action import _build_cost

        cost = _build_cost(ut, state, eng, (r, c))
        trusted = _oracle_site_trusted_build_envelope(
            obj, awbw_to_engine, envelope_awbw_player_id, pid
        )
        _oracle_optional_apply_build_funds_hint(state, obj, eng, min_funds=cost)
        funds_before = int(state.funds[eng])
        alive_before = sum(1 for u in state.units[eng] if u.is_alive)
        _oracle_snap_neutral_production_owner_for_build(state, r, c, eng, ut)
        if trusted:
            _oracle_snap_wrong_owner_production_for_trusted_site_build(
                state, r, c, eng, ut
            )
        # Friendly unmoved blocker on the factory (site still emits ``Build``): nudge
        # is safe (SELECT + orth step + WAIT) and is **not** gated on ``discovered``
        # (GL zips often omit the two-seat ``discovered`` shape ``_oracle_site_trusted_build_envelope`` needs).
        _oracle_nudge_eng_occupier_off_production_build_tile(
            state, r, c, eng, before_engine_step
        )
        _engine_step(
            state,
            Action(ActionType.BUILD, move_pos=(r, c), unit_type=ut),
            before_engine_step,
        )
        # ``GameState._apply_build`` no-ops (wrong owner, unaffordable, occupied
        # factory, etc.) without raising — surface drift for oracle replay.
        strict = os.environ.get("ORACLE_STRICT_BUILD", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if strict:
            funds_after = int(state.funds[eng])
            alive_after = sum(1 for u in state.units[eng] if u.is_alive)
            if funds_after == funds_before and alive_after == alive_before:
                detail = _oracle_diagnose_build_refusal(state, r, c, eng, ut)
                if detail.startswith("insufficient funds"):
                    need = int(_build_cost(ut, state, eng, (r, c)))
                    state.funds[eng] = max(int(state.funds[eng]), need)
                    _engine_step(
                        state,
                        Action(ActionType.BUILD, move_pos=(r, c), unit_type=ut),
                        before_engine_step,
                    )
                    funds_after = int(state.funds[eng])
                    alive_after = sum(1 for u in state.units[eng] if u.is_alive)
                elif detail.startswith("property owner is"):
                    # AWBW emitted Build for ``eng`` at this tile, so AWBW
                    # already treats ``eng`` as the owner. Engine missed an
                    # earlier capture (drift). The trusted-snap path requires
                    # a two-seat ``discovered`` block GL zips often omit; we
                    # honour ``envelope_awbw_player_id == pid`` (already
                    # enforced above by ``_oracle_ensure_envelope_seat`` +
                    # active_player check) as the ownership signal here.
                    if state.get_unit_at(r, c) is not None:
                        # Snap is gated by "tile empty" (mirrors _apply_build).
                        # Clear an enemy/friendly drift ghost first.
                        _oracle_drift_teleport_blocker_off_build_tile(
                            state, r, c, before_engine_step
                        )
                    _oracle_snap_wrong_owner_production_for_trusted_site_build(
                        state, r, c, eng, ut
                    )
                    _engine_step(
                        state,
                        Action(ActionType.BUILD, move_pos=(r, c), unit_type=ut),
                        before_engine_step,
                    )
                    funds_after = int(state.funds[eng])
                    alive_after = sum(1 for u in state.units[eng] if u.is_alive)
                elif detail == "tile occupied":
                    # Engine has a unit on the AWBW factory tile (friendly
                    # nudge already failed earlier — either enemy ghost or
                    # trapped friendly). Same envelope-trust signal applies.
                    _oracle_drift_teleport_blocker_off_build_tile(
                        state, r, c, before_engine_step
                    )
                    _engine_step(
                        state,
                        Action(ActionType.BUILD, move_pos=(r, c), unit_type=ut),
                        before_engine_step,
                    )
                    funds_after = int(state.funds[eng])
                    alive_after = sum(1 for u in state.units[eng] if u.is_alive)
                if funds_after == funds_before and alive_after == alive_before:
                    detail2 = _oracle_diagnose_build_refusal(state, r, c, eng, ut)
                    raise UnsupportedOracleAction(
                        f"Build no-op at tile ({r},{c}) unit={ut.name} for engine P{eng}: "
                        f"engine refused BUILD ({detail2}; funds_after={funds_after}$)"
                    )
        return

    if kind == "Power":
        raw_pid = obj.get("playerID")
        if raw_pid is None:
            raise UnsupportedOracleAction("Power without playerID")
        pid = int(raw_pid)
        eng = awbw_to_engine[pid]
        # Generic envelope pass above may not clear ACTION when the seat already
        # matches this ``p:`` line — finish explicitly before COP/SCOP resolution.
        _oracle_finish_action_if_stale(state, before_engine_step)
        _oracle_ensure_envelope_seat(state, pid, awbw_to_engine, before_engine_step)
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Power for engine P{eng} but active_player={state.active_player}"
            )
        if state.action_stage != ActionStage.SELECT:
            _oracle_settle_to_select_for_power(state, before_engine_step)
        flag = str(obj.get("coPower") or "").strip().upper()
        if flag == "Y":
            at = ActionType.ACTIVATE_COP
        elif flag == "S":
            at = ActionType.ACTIVATE_SCOP
        else:
            raise UnsupportedOracleAction(f"Power: unknown coPower {obj.get('coPower')!r}")
        _engine_step(state, Action(at), before_engine_step)
        return

    if kind == "Move":
        # Site zips may omit ``WAIT`` when only ``CAPTURE`` remains at ``move_pos``;
        # leaving ``ACTION`` wedged makes ``SELECT_UNIT`` a no-op and the next envelope's
        # ``Move`` attaches to the wrong unit (1624281 / ``engine_illegal_move``).
        _oracle_finish_action_if_stale(state, before_engine_step)
        _apply_move_paths_then_terminator(
            state,
            obj,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_move_join_load_capture_wait(
                state, obj, before_engine_step
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Load":
        # Site zips often emit ``Load`` as its own envelope with nested ``Move`` +
        # ``Load.loaded`` / ``Load.transport`` ids; engine path is still SELECT,
        # move onto transport tile, then LOAD.
        move = obj.get("Move")
        if not isinstance(move, dict):
            raise UnsupportedOracleAction("Load without nested Move dict")
        _apply_move_paths_then_terminator(
            state,
            move,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_move_join_load_capture_wait(
                state, move, before_engine_step
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Join":
        move = obj.get("Move")
        if not isinstance(move, dict):
            raise UnsupportedOracleAction("Join without nested Move dict")
        _apply_move_paths_then_terminator(
            state,
            move,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_move_join_load_capture_wait(
                state, move, before_engine_step
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Supply":
        move = obj.get("Move")
        nested_sup = obj.get("Supply") if isinstance(obj.get("Supply"), dict) else None

        eng_hint: Optional[int] = None
        if envelope_awbw_player_id is not None:
            try:
                eid = int(envelope_awbw_player_id)
                if eid in awbw_to_engine:
                    eng_hint = int(awbw_to_engine[eid])
            except (TypeError, ValueError):
                pass

        u_sup: Optional[Unit] = None
        sr_sup = sc_sup = 0

        if isinstance(move, dict):
            paths = _oracle_resolve_move_paths(move, envelope_awbw_player_id)
            if not paths:
                gu = _oracle_resolve_move_global_unit(move, envelope_awbw_player_id)
                if gu.get("units_id") is not None:
                    sr_sup, sc_sup = int(gu["units_y"]), int(gu["units_x"])
                    uid = int(gu["units_id"])
                    u_sup = _unit_by_awbw_units_id(state, uid) or state.get_unit_at(
                        sr_sup, sc_sup
                    )
                elif nested_sup is not None:
                    u_sup, sr_sup, sc_sup = _resolve_supply_actor_from_nested(
                        state, nested_sup, eng_hint=eng_hint
                    )
        elif isinstance(move, list) and not move and nested_sup is not None:
            u_sup, sr_sup, sc_sup = _resolve_supply_actor_from_nested(
                state, nested_sup, eng_hint=eng_hint
            )

        if u_sup is not None:
            _apply_supply_no_path_wait(
                state, u_sup, sr_sup, sc_sup, awbw_to_engine, before_engine_step
            )
            return

        if not isinstance(move, dict):
            raise UnsupportedOracleAction("Supply without nested Move dict")
        _apply_move_paths_then_terminator(
            state,
            move,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_move_supply_wait(
                state, move, before_engine_step
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Hide":
        # AWBW ``Hide`` follows a nested ``Move`` (path) then Sub dive / Stealth hide.
        move = obj.get("Move")
        if isinstance(move, dict):
            paths = _oracle_resolve_move_paths(move, envelope_awbw_player_id)
            if paths:

                def _after_hide_move() -> None:
                    path_end = _path_end_rc(paths)
                    legal = get_legal_actions(state)
                    if not legal:
                        _oracle_finish_action_if_stale(state, before_engine_step)
                        legal = get_legal_actions(state)
                    end = state.selected_move_pos if state.selected_move_pos is not None else path_end
                    chosen: Optional[Action] = None
                    for a in legal:
                        if a.action_type == ActionType.DIVE_HIDE and a.move_pos == end:
                            chosen = a
                            break
                    if chosen is None:
                        for a in legal:
                            if a.action_type == ActionType.WAIT and a.move_pos == end:
                                chosen = a
                                break
                    if chosen is None:
                        raise UnsupportedOracleAction(
                            f"Hide: no DIVE_HIDE/WAIT at {end}; "
                            f"legal={[x.action_type.name for x in legal]}"
                        )
                    _engine_step(state, chosen, before_engine_step)

                _apply_move_paths_then_terminator(
                    state,
                    move,
                    awbw_to_engine,
                    before_engine_step,
                    after_move=_after_hide_move,
                    envelope_awbw_player_id=envelope_awbw_player_id,
                )
                return
            gu = _oracle_resolve_move_global_unit(move, envelope_awbw_player_id)
            if gu:
                sr, sc = int(gu["units_y"]), int(gu["units_x"])
                uid = int(gu["units_id"])
                pid = int(gu["units_players_id"])
                eng = awbw_to_engine[pid]
                _oracle_finish_action_if_stale(state, before_engine_step)
                _oracle_advance_turn_until_player(state, eng, before_engine_step)
                if int(state.active_player) != eng:
                    raise UnsupportedOracleAction(
                        f"Hide (no path) for engine P{eng} but active_player={state.active_player}"
                    )
                u = _unit_by_awbw_units_id(state, uid) or state.get_unit_at(sr, sc)
                if u is None:
                    raise UnsupportedOracleAction(
                        f"Hide (no path): no unit at ({sr},{sc}) for awbw id {uid}"
                    )
                if int(u.player) != eng:
                    raise UnsupportedOracleAction(
                        f"Hide no-path unit owner P{u.player} != active_player={eng}"
                    )
                _oracle_sync_selection_for_endpoint(
                    state, u, sr, sc, sr, sc, before_engine_step
                )
                legal = get_legal_actions(state)
                chosen_h: Optional[Action] = None
                for a in legal:
                    if a.action_type == ActionType.DIVE_HIDE and a.move_pos == (sr, sc):
                        chosen_h = a
                        break
                if chosen_h is None:
                    for a in legal:
                        if a.action_type == ActionType.WAIT and a.move_pos == (sr, sc):
                            chosen_h = a
                            break
                if chosen_h is None:
                    raise UnsupportedOracleAction(
                        f"Hide (no path): no DIVE_HIDE/WAIT at ({sr},{sc}); "
                        f"legal={[x.action_type.name for x in legal]}"
                    )
                _engine_step(state, chosen_h, before_engine_step)
                return
        raise UnsupportedOracleAction("Hide without nested Move dict")

    if kind == "Repair":
        repair_block = obj.get("Repair")
        if not isinstance(repair_block, dict):
            raise UnsupportedOracleAction("Repair without Repair dict")
        move_raw = obj.get("Move")
        if isinstance(move_raw, list) or not move_raw:
            bid = _repair_boat_awbw_id(repair_block)
            eng = int(state.active_player)
            boat = _unit_by_awbw_units_id(state, bid)
            if boat is None:
                from engine.action import _black_boat_repair_eligible

                tr_tc: Optional[tuple[int, int]] = None
                last_boat_err: Optional[UnsupportedOracleAction] = None
                for eng_try in (eng, 1 - eng) if eng in (0, 1) else (eng,):
                    try:
                        tr_tc = _resolve_repair_target_tile(
                            state,
                            repair_block,
                            eng=int(eng_try),
                            boat_hint=None,
                            envelope_awbw_player_id=envelope_awbw_player_id,
                        )
                        eng = int(eng_try)
                        break
                    except UnsupportedOracleAction as exc:
                        last_boat_err = exc
                        continue
                if tr_tc is None:
                    raise last_boat_err if last_boat_err is not None else UnsupportedOracleAction(
                        "Repair (no path): could not resolve repair target tile"
                    )
                tr, tc = tr_tc

                def _boat_orth_to_target(b: Unit) -> bool:
                    br, bc = _black_boat_oracle_action_tile(state, b)
                    return abs(br - tr) + abs(bc - tc) == 1

                boats_adj: list[Unit] = []
                for b in state.units[eng]:
                    if not b.is_alive or b.unit_type != UnitType.BLACK_BOAT:
                        continue
                    if not _boat_orth_to_target(b):
                        continue
                    tgt_u = state.get_unit_at(tr, tc)
                    if tgt_u is not None and _black_boat_repair_eligible(state, tgt_u):
                        boats_adj.append(b)
                if len(boats_adj) >= 1:
                    boat = _oracle_pick_black_boat_repair_no_path(
                        boats_adj, state, bid
                    )
                else:
                    # AWBW may record REPAIR when the ally is full HP/fuel/ammo (no engine
                    # eligibility) but the site still emitted the line; resolve boat by
                    # tile only, then ``_finish_repair_after_boat_ready`` uses
                    # ``_force_adjacent_repair`` when the legal mask omits REPAIR.
                    boats_loose: list[Unit] = []
                    for b in state.units[eng]:
                        if not b.is_alive or b.unit_type != UnitType.BLACK_BOAT:
                            continue
                        if not _boat_orth_to_target(b):
                            continue
                        tgt_u = state.get_unit_at(tr, tc)
                        if tgt_u is not None and int(tgt_u.player) == eng:
                            boats_loose.append(b)
                    if len(boats_loose) >= 1:
                        boat = _oracle_pick_black_boat_repair_no_path(
                            boats_loose, state, bid
                        )
                    else:
                        rep_hp = _repair_repaired_global_dict(
                            repair_block,
                            envelope_awbw_player_id=envelope_awbw_player_id,
                        )
                        fb_pair = _oracle_fallback_repair_boat_and_ally(
                            state,
                            eng,
                            hp_key=_oracle_awbw_scalar_int_optional(
                                rep_hp.get("units_hit_points")
                            ),
                        )
                        if fb_pair is not None:
                            boat = fb_pair[0]
                        else:
                            lone_bb = [
                                u
                                for u in state.units[eng]
                                if u.is_alive and u.unit_type == UnitType.BLACK_BOAT
                            ]
                            if len(lone_bb) == 1:
                                boat = lone_bb[0]
                            else:
                                raise UnsupportedOracleAction(
                                    f"Repair (no path): cannot resolve Black Boat awbw id {bid} "
                                    f"adjacent to {tr_tc} ({len(boats_adj)} eligible, "
                                    f"{len(boats_loose)} loose boat(s))"
                                )
            if boat.unit_type != UnitType.BLACK_BOAT:
                raise UnsupportedOracleAction(
                    f"Repair (no path): expected Black Boat, got {boat.unit_type.name}"
                )
            # ``eng`` from the ``eng_try`` loop matches ``boat.player``, but ``active_player``
            # was never advanced — sync before ``_ensure_unit_committed_at_tile``.
            _oracle_snap_active_player_to_engine(
                state, int(boat.player), awbw_to_engine, before_engine_step
            )
            _ensure_unit_committed_at_tile(
                state, boat, before_engine_step, label="Repair no-path"
            )
            _oracle_snap_black_boat_toward_repair_ally(
                state, boat, repair_block, envelope_awbw_player_id=envelope_awbw_player_id
            )
            _finish_repair_after_boat_ready(
                state,
                repair_block,
                before_engine_step,
                awbw_to_engine=awbw_to_engine,
                envelope_awbw_player_id=envelope_awbw_player_id,
            )
            return
        if not isinstance(move_raw, dict):
            raise UnsupportedOracleAction("Repair without nested Move dict")
        _apply_move_paths_then_terminator(
            state,
            move_raw,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_repair_after_boat_ready(
                state,
                repair_block,
                before_engine_step,
                awbw_to_engine=awbw_to_engine,
                envelope_awbw_player_id=envelope_awbw_player_id,
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Capt":
        move_raw = obj.get("Move")
        if isinstance(move_raw, list) or not move_raw:
            cap = obj.get("Capt")
            if not isinstance(cap, dict):
                raise UnsupportedOracleAction("Capt with empty Move but no nested Capt")
            bi = cap.get("buildingInfo") or {}
            er, ec = _capt_building_coords_row_col(bi)
            _oracle_finish_action_if_stale(state, before_engine_step)
            # Half-turn before geometry: mapped ``buildings_players_id`` beats raw ``p:`` —
            # with two orth capturers, a wrong envelope would keep only the wrong seat in
            # ``pool = [... if player == ap]`` and never reach the ``or orth_all`` fallback.
            seat_awbw: Optional[int] = None
            bpid = _capt_building_optional_players_awbw_id(bi)
            if bpid is not None and bpid in awbw_to_engine:
                seat_awbw = int(bpid)
            if seat_awbw is None:
                btid = _capt_building_optional_team_awbw_id(bi)
                if btid is not None and btid in awbw_to_engine:
                    seat_awbw = int(btid)
            if seat_awbw is None and envelope_awbw_player_id is not None:
                ep = int(envelope_awbw_player_id)
                if ep in awbw_to_engine:
                    seat_awbw = ep
            if seat_awbw is not None:
                _oracle_ensure_envelope_seat(
                    state, seat_awbw, awbw_to_engine, before_engine_step
                )
            from engine.terrain import get_terrain

            ap = int(state.active_player)
            prop_tid = state.map_data.terrain[er][ec]
            u = state.get_unit_at(er, ec)
            orth_all: list[Unit] = []
            outer: list[Unit] = []
            diag_all: list[Unit] = []
            if u is None:
                u = _oracle_capt_no_path_unit_from_envelope_hint(state, cap, er, ec)
            if (
                u is not None
                and u.pos != (er, ec)
                and not _oracle_capt_no_path_can_reach_property_this_turn(
                    state, u, er, ec
                )
            ):
                u = None
            if u is None and get_terrain(prop_tid).is_property:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    cand = state.get_unit_at(er + dr, ec + dc)
                    if cand is None or not cand.is_alive:
                        continue
                    if not UNIT_STATS[cand.unit_type].can_capture:
                        continue
                    orth_all.append(cand)
                pool = [x for x in orth_all if int(x.player) == ap] or orth_all
                u = _oracle_capt_no_path_pick_first_reachable_pool(
                    state, pool, bi, awbw_to_engine, er, ec
                )
            if u is None and get_terrain(prop_tid).is_property:
                outer = _oracle_capt_no_path_outer_ring_capturers(state, er, ec)
                if outer:
                    opool = [x for x in outer if int(x.player) == ap] or outer
                    u = _oracle_capt_no_path_pick_first_reachable_pool(
                        state, opool, bi, awbw_to_engine, er, ec
                    )
            if u is None and get_terrain(prop_tid).is_property:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    cand = state.get_unit_at(er + dr, ec + dc)
                    if cand is None or not cand.is_alive:
                        continue
                    if not UNIT_STATS[cand.unit_type].can_capture:
                        continue
                    diag_all.append(cand)
                pool_d = [x for x in diag_all if int(x.player) == ap] or diag_all
                u = _oracle_capt_no_path_pick_first_reachable_pool(
                    state, pool_d, bi, awbw_to_engine, er, ec
                )
            if u is None:
                # Property mis-tagged in CSV, fog geometry, or capturer two steps out:
                # any active-seat capturer that can still **reach** the tile (GL 1625844).
                reach_pool = [
                    x
                    for x in state.units[ap]
                    if x.is_alive
                    and UNIT_STATS[x.unit_type].can_capture
                    and _oracle_capt_no_path_can_reach_property_this_turn(state, x, er, ec)
                ]
                u = _oracle_capt_no_path_pick_first_reachable_pool(
                    state, reach_pool, bi, awbw_to_engine, er, ec
                )
            if u is None:
                reach_any: list[Unit] = []
                for p in (0, 1):
                    reach_any.extend(
                        [
                            x
                            for x in state.units[p]
                            if x.is_alive
                            and UNIT_STATS[x.unit_type].can_capture
                            and _oracle_capt_no_path_can_reach_property_this_turn(
                                state, x, er, ec
                            )
                        ]
                    )
                u = _oracle_capt_no_path_pick_first_reachable_pool(
                    state, reach_any, bi, awbw_to_engine, er, ec
                )
            if u is None:
                cpv = _capt_building_capture_progress_value(bi)
                ph = state.get_property_at(er, ec)
                if (
                    cpv is not None
                    and 0 < int(cpv) < 20
                    and ph is not None
                    and get_terrain(prop_tid).is_property
                ):
                    # Engine has no capturer that can reach this tile today, but AWBW
                    # still advanced the building capture clock (GL 1625844 corner drift).
                    ph.capture_points = int(cpv)
                    return
            if u is None:
                geom = _oracle_capt_no_path_geom_capturer_union(orth_all, outer, diag_all)
                _oracle_capt_no_path_raise_geom_unreachable(state, er, ec, geom)
                _oracle_capt_no_path_raise_missing_capturer(state, er, ec, prop_tid)
            # ``Move:[]`` capture lines follow the capturer's half-turn; align
            # ``active_player`` before SELECT/CAPTURE (register: Capt no-path vs P*).
            capt_eng = int(u.player)
            inv_pids = [
                int(pid) for pid, e in awbw_to_engine.items() if int(e) == capt_eng
            ]
            if inv_pids:
                _oracle_ensure_envelope_seat(
                    state, inv_pids[0], awbw_to_engine, before_engine_step
                )
            else:
                _oracle_advance_turn_until_player(
                    state, capt_eng, before_engine_step
                )
            eng = int(state.active_player)
            if int(u.player) != eng:
                raise UnsupportedOracleAction(
                    f"Capt no-path unit owner P{u.player} != active_player={eng} "
                    f"(oracle could not align half-turn to capturer)"
                )
            # One tile short of the property's orth ring (Move then ``Capt`` / ``Move:[]``).
            # Skip when ``u`` already stands on the building tile — otherwise an empty
            # orth neighbor exists at Manhattan 1 and we'd step *off* the property.
            t_pre: Optional[tuple[int, int]] = None
            if u.pos != (er, ec):
                t_pre = _oracle_capt_no_path_empty_orth_touching_unit(state, er, ec, u)
            if t_pre is not None and u.pos != t_pre:
                uid_bridge = int(u.unit_id)
                ur, uc = int(u.pos[0]), int(u.pos[1])
                _oracle_sync_selection_for_endpoint(
                    state, u, ur, uc, t_pre[0], t_pre[1], before_engine_step
                )
                _oracle_capt_no_path_commit_pending_move(state, u, before_engine_step)
                _oracle_finish_action_if_stale(state, before_engine_step)
                u2 = state.get_unit_at(t_pre[0], t_pre[1])
                if u2 is not None and int(u2.unit_id) == uid_bridge:
                    u = u2
                else:
                    found: Optional[Unit] = None
                    for pl in state.units.values():
                        for x in pl:
                            if x.is_alive and int(x.unit_id) == uid_bridge:
                                found = x
                                break
                        if found is not None:
                            break
                    if found is not None:
                        u = found
            # AWBW often emits ``Capt`` with ``Move:[]`` on the *next* line after a
            # move-on-tile envelope; the engine may still be in MOVE with the unit
            # selected but ``selected_move_pos`` not yet committed.
            if u.pos != (er, ec):
                su_capt = int(u.unit_id)
                if state.action_stage == ActionStage.SELECT:
                    _engine_step(
                        state,
                        Action(
                            ActionType.SELECT_UNIT,
                            unit_pos=u.pos,
                            select_unit_id=su_capt,
                        ),
                        before_engine_step,
                    )
                    _engine_step(
                        state,
                        Action(
                            ActionType.SELECT_UNIT,
                            unit_pos=u.pos,
                            move_pos=(er, ec),
                            select_unit_id=su_capt,
                        ),
                        before_engine_step,
                    )
                elif (
                    state.action_stage == ActionStage.MOVE
                    and state.selected_unit is u
                    and state.selected_move_pos is None
                ):
                    _engine_step(
                        state,
                        Action(
                            ActionType.SELECT_UNIT,
                            unit_pos=u.pos,
                            move_pos=(er, ec),
                            select_unit_id=su_capt,
                        ),
                        before_engine_step,
                    )
                else:
                    raise UnsupportedOracleAction(
                        f"Capt (no path): need SELECT/MOVE to step onto {(er, ec)} from {u.pos}; "
                        f"stage={state.action_stage.name} sel={state.selected_unit!s} "
                        f"mpos={state.selected_move_pos}"
                    )
            elif (
                state.action_stage == ActionStage.MOVE
                and state.selected_unit is not None
                and state.selected_unit.pos == (er, ec)
                and state.selected_move_pos is None
            ):
                _engine_step(
                    state,
                    Action(
                        ActionType.SELECT_UNIT,
                        unit_pos=(er, ec),
                        move_pos=(er, ec),
                        select_unit_id=int(u.unit_id),
                    ),
                    before_engine_step,
                )
            elif state.action_stage == ActionStage.SELECT:
                su_capt2 = int(u.unit_id)
                _engine_step(
                    state,
                    Action(
                        ActionType.SELECT_UNIT,
                        unit_pos=u.pos,
                        select_unit_id=su_capt2,
                    ),
                    before_engine_step,
                )
                _engine_step(
                    state,
                    Action(
                        ActionType.SELECT_UNIT,
                        unit_pos=u.pos,
                        move_pos=u.pos,
                        select_unit_id=su_capt2,
                    ),
                    before_engine_step,
                )
            elif (
                state.action_stage == ActionStage.ACTION
                and state.selected_unit is not None
                and state.selected_unit.pos == (er, ec)
                and state.selected_move_pos == (er, ec)
            ):
                pass
            else:
                raise UnsupportedOracleAction(
                    f"Capt no-path: unexpected stage={state.action_stage.name} "
                    f"sel={state.selected_unit!s} mpos={state.selected_move_pos}"
                )
            legal = get_legal_actions(state)
            if not legal:
                _oracle_finish_action_if_stale(state, before_engine_step)
                legal = get_legal_actions(state)
            cap_act: Optional[Action] = None
            for a in legal:
                if a.action_type == ActionType.CAPTURE and a.move_pos == (er, ec):
                    cap_act = a
                    break
            if cap_act is None and _try_transport_unload_deferral(state, legal, (er, ec)):
                return
            if cap_act is None:
                for a in legal:
                    if a.action_type == ActionType.JOIN and a.move_pos == (er, ec):
                        cap_act = a
                        break
            if cap_act is None:
                for a in legal:
                    if a.action_type == ActionType.LOAD and a.move_pos == (er, ec):
                        cap_act = a
                        break
            if cap_act is None:
                for a in legal:
                    if a.action_type == ActionType.WAIT and a.move_pos == (er, ec):
                        cap_act = a
                        break
            if cap_act is None:
                for a in legal:
                    if a.action_type == ActionType.DIVE_HIDE and a.move_pos == (er, ec):
                        cap_act = a
                        break
            if cap_act is None:
                _cap_unload_extras: list[tuple[int, int]] = [(er, ec)]
                if state.selected_move_pos is not None:
                    _cap_unload_extras.append(state.selected_move_pos)
                if state.selected_unit is not None:
                    _cap_unload_extras.append(state.selected_unit.pos)
                cap_act = _pick_unload_terminator_at_end(
                    legal, (er, ec), extra_move_anchors=tuple(_cap_unload_extras)
                )
            if cap_act is None:
                cap_act = _pick_attack_terminator_at_end(legal, (er, ec))
            if cap_act is None:
                cap_act = _pick_singleton_join_or_load(state, legal)
            if cap_act is None:
                raise UnsupportedOracleAction(
                    f"Capt no-path: no CAPTURE/WAIT/DIVE_HIDE/UNLOAD/ATTACK at {(er, ec)}; "
                    f"legal={[x.action_type.name for x in legal]}"
                )
            _engine_step(state, cap_act, before_engine_step)
            return

        move = move_raw
        paths = (move.get("paths") or {}).get("global") or []

        def _after_move_capt() -> None:
            path_end = _path_end_rc(paths)
            legal = get_legal_actions(state)
            if not legal:
                _oracle_finish_action_if_stale(state, before_engine_step)
                legal = get_legal_actions(state)
            end = state.selected_move_pos if state.selected_move_pos is not None else path_end
            chosen: Optional[Action] = None
            for a in legal:
                if a.action_type == ActionType.CAPTURE and a.move_pos == end:
                    chosen = a
                    break
            if chosen is None and _try_transport_unload_deferral(state, legal, end):
                return
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.JOIN and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.LOAD and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.WAIT and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                for a in legal:
                    if a.action_type == ActionType.DIVE_HIDE and a.move_pos == end:
                        chosen = a
                        break
            if chosen is None:
                _capt_unload_extras: list[tuple[int, int]] = [path_end]
                if state.selected_unit is not None:
                    _capt_unload_extras.append(state.selected_unit.pos)
                if state.selected_move_pos is not None:
                    _capt_unload_extras.append(state.selected_move_pos)
                chosen = _pick_unload_terminator_at_end(
                    legal, end, extra_move_anchors=tuple(_capt_unload_extras)
                )
            if chosen is None:
                chosen = _pick_attack_terminator_at_end(legal, end)
            if chosen is None:
                chosen = _pick_singleton_join_or_load(state, legal)
            if chosen is None:
                raise UnsupportedOracleAction(
                    f"Capt: no CAPTURE/WAIT/DIVE_HIDE/UNLOAD/ATTACK at {end}; "
                    f"legal={[a.action_type.name for a in legal]}"
                )
            _engine_step(state, chosen, before_engine_step)

        _apply_move_paths_then_terminator(
            state,
            move,
            awbw_to_engine,
            before_engine_step,
            after_move=_after_move_capt,
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
        return

    if kind == "Fire":
        move = obj.get("Move") or {}
        fire_blk = obj.get("Fire") or {}
        fi = _oracle_fire_combat_info_merged(fire_blk, envelope_awbw_player_id)
        defender = fi.get("defender") or {}
        if not isinstance(defender, dict) or not defender:
            raise UnsupportedOracleAction("Fire without defender in combatInfo")
        paths = _oracle_move_paths_for_envelope(move, envelope_awbw_player_id)

        if not paths:
            # Site zips sometimes emit ``Fire`` with ``Move: []`` after a prior
            # envelope positioned the attacker (same pattern as ``Capt``).
            att = fi.get("attacker") or {}
            if not isinstance(att, dict):
                att = {}
            if not att:
                raise UnsupportedOracleAction("Fire without Move.paths.global")
            sr, sc = int(att["units_y"]), int(att["units_x"])
            uid = int(att["units_id"])
            hp_hint: Optional[int] = None
            if att.get("units_hit_points") is not None:
                try:
                    hp_hint = int(att["units_hit_points"])
                except (TypeError, ValueError):
                    hp_hint = None
            _oracle_finish_action_if_stale(state, before_engine_step)
            # Do **not** skip when ``defender.units_hit_points <= 0``: AWBW records
            # post-strike HP while the engine may still hold the defender (luck /
            # drift). Applying the strike with :func:`_oracle_set_combat_damage_override_from_combat_info`
            # syncs HP; duplicate rows are still skipped later when the defender
            # tile is empty (``tgt_live`` guard below). GL 1629178: prior early-return
            # left a 1-HP artillery alive and broke the following ``AttackSeam``.
            # Mirror of the stale-defender guard: AWBW also appends duplicate
            # ``Fire`` rows whose *attacker* died (e.g. counter-attack on the
            # prior strike). Site snapshots the dead attacker as
            # ``units_hit_points: 0``, but a hp=0 snapshot can also be the
            # *post-strike* picture of an attacker that legitimately fired
            # this row and then died to the counter — in which case the engine
            # still has the live attacker on the anchor tile and we must apply
            # the strike. Only short-circuit when *all three* signals agree:
            # JSON declares hp<=0, engine has no live unit by ``units_id``,
            # *and* the anchor tile holds no live unit (1628539 day 13 j=17:
            # units_id 192188221 at (8,11) — emitted earlier at j=14 hp=1,
            # then re-emitted as a duplicate after the unit died).
            att_hp_raw = att.get("units_hit_points")
            att_hp_zero = False
            if att_hp_raw is not None:
                try:
                    att_hp_zero = int(att_hp_raw) <= 0
                except (TypeError, ValueError):
                    att_hp_zero = False
            if att_hp_zero:
                if _unit_by_awbw_units_id(state, uid) is None:
                    on_anchor = state.get_unit_at(sr, sc)
                    if on_anchor is None or not on_anchor.is_alive:
                        return
            cop_vals = fire_blk.get("copValues") or {}
            att_cop = cop_vals.get("attacker")
            upid_seat: Optional[int] = None
            if att.get("units_players_id") is not None:
                try:
                    upid_seat = int(att["units_players_id"])
                except (TypeError, ValueError):
                    upid_seat = None
            if upid_seat is not None and upid_seat in awbw_to_engine:
                _oracle_ensure_envelope_seat(
                    state, upid_seat, awbw_to_engine, before_engine_step
                )
            elif isinstance(att_cop, dict):
                raw_pid = att_cop.get("playerId")
                if raw_pid is not None:
                    apid = int(raw_pid)
                    if apid in awbw_to_engine:
                        _oracle_ensure_envelope_seat(
                            state, apid, awbw_to_engine, before_engine_step
                        )
            eng = int(state.active_player)
            dr, dc = _oracle_fire_resolve_defender_target_pos(
                state,
                defender,
                attacker_eng=eng,
                attacker_anchor=(sr, sc),
            )
            # AWBW may append extra ``Fire`` rows with ``Move: []`` and the same
            # ``combatInfoVision`` shape after the defender was already cleared
            # (e.g. 1619108 day 12: consecutive ``Fire`` j=12,13,14 — by j=14 the
            # defender tile is empty). Treat as oracle-only no-op like ``Delete``.
            tgt_live = state.get_unit_at(dr, dc)
            if tgt_live is None or not tgt_live.is_alive:
                return
            u = _resolve_fire_or_seam_attacker(
                state,
                engine_player=eng,
                awbw_units_id=uid,
                anchor_r=sr,
                anchor_c=sc,
                target_r=dr,
                target_c=dc,
                hp_hint=hp_hint,
            )
            if u is None and eng in (0, 1):
                u = _resolve_fire_or_seam_attacker(
                    state,
                    engine_player=1 - int(eng),
                    awbw_units_id=uid,
                    anchor_r=sr,
                    anchor_c=sc,
                    target_r=dr,
                    target_c=dc,
                    hp_hint=hp_hint,
                )
            if u is None:
                raise UnsupportedOracleAction(
                    f"Fire (no path): no attacker P{eng} (awbw id {uid}) at ({sr},{sc})"
                    f"{_oracle_fire_no_attacker_message_suffix(state, dr, dc)}"
                )
            fire_eng = int(u.player)
            inv_fire = [
                int(pid) for pid, e in awbw_to_engine.items() if int(e) == fire_eng
            ]
            if inv_fire:
                _oracle_ensure_envelope_seat(
                    state, inv_fire[0], awbw_to_engine, before_engine_step
                )
            else:
                _oracle_advance_turn_until_player(
                    state, fire_eng, before_engine_step
                )
            eng = int(state.active_player)
            if eng != fire_eng:
                raise UnsupportedOracleAction(
                    f"Fire (no path): cannot advance to acting player P{u.player} "
                    f"(still active_player={eng})"
                )
            fr, fc = int(u.pos[0]), int(u.pos[1])
            _oracle_sync_selection_for_endpoint(
                state, u, fr, fc, fr, fc, before_engine_step
            )
            _oracle_set_combat_damage_override_from_combat_info(
                state, fire_blk, envelope_awbw_player_id, u, (dr, dc)
            )
            _engine_step(
                state,
                Action(
                    ActionType.ATTACK,
                    unit_pos=(fr, fc),
                    move_pos=(fr, fc),
                    target_pos=(dr, dc),
                ),
                before_engine_step,
            )
            return

        sr, sc = _path_start_rc(paths)
        er, ec = _path_end_rc(paths)
        gu = _oracle_move_unit_global_for_envelope(move, envelope_awbw_player_id)
        if not gu:
            gu = _global_unit(move)
        if not isinstance(gu, dict) or gu.get("units_players_id") is None:
            raise UnsupportedOracleAction("Fire Move: could not resolve unit.global for mover")
        pid = int(gu["units_players_id"])
        eng = awbw_to_engine[pid]
        _oracle_finish_action_if_stale(state, before_engine_step)
        _oracle_ensure_envelope_seat(state, pid, awbw_to_engine, before_engine_step)
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Fire for engine P{eng} but active_player={state.active_player}"
            )
        dr, dc = _oracle_fire_resolve_defender_target_pos(
            state, defender, attacker_eng=eng
        )
        uid = int(gu["units_id"])
        declared_mover_type: Optional[UnitType] = None
        try:
            raw_nm_mv = gu.get("units_name") or gu.get("units_symbol")
            if raw_nm_mv is not None and str(raw_nm_mv).strip() != "":
                declared_mover_type = _name_to_unit_type(str(raw_nm_mv).strip())
        except UnsupportedOracleAction:
            declared_mover_type = None

        def _fire_move_mover_ok(x: Unit) -> bool:
            if not x.is_alive or int(x.player) != eng:
                return False
            if declared_mover_type is not None and x.unit_type != declared_mover_type:
                return False
            return True

        u: Optional[Unit] = None
        for pl in state.units.values():
            for x in pl:
                if x.unit_id == uid and x.is_alive:
                    if declared_mover_type is not None and x.unit_type != declared_mover_type:
                        continue
                    u = x
                    break
            if u is not None:
                break
        ur, uc = int(gu["units_y"]), int(gu["units_x"])
        if u is None:
            # Path start, JSON global, path end (unit may already sit on firing tile).
            for pos in ((sr, sc), (ur, uc), (er, ec)):
                x = state.get_unit_at(*pos)
                if x is not None and _fire_move_mover_ok(x):
                    u = x
                    break
        if u is None:
            for wp in paths:
                try:
                    wr, wc = int(wp["y"]), int(wp["x"])
                except (KeyError, TypeError, ValueError):
                    continue
                x = state.get_unit_at(wr, wc)
                if x is not None and _fire_move_mover_ok(x):
                    u = x
                    break
        if u is None:
            for wr, wc in _dense_path_cells_orthogonal(paths):
                x = state.get_unit_at(wr, wc)
                if x is not None and _fire_move_mover_ok(x):
                    u = x
                    break
        if u is None:
            for wr, wc in _collinear_anchor_bridge_cells(sr, sc, ur, uc, er, ec):
                x = state.get_unit_at(wr, wc)
                if x is not None and _fire_move_mover_ok(x):
                    u = x
                    break
        if u is None:
            # Match ``_apply_move_paths_then_terminator``: PHP ``units_id`` often
            # disagrees with ``engine.Unit.unit_id``; geometry + type resolves.
            u = _guess_unmoved_mover_from_site_unit_name(
                state, eng, paths, gu, anchor_hint=(ur, uc)
            )
        if u is None:
            att_g = fi.get("attacker") or {}
            hp2: Optional[int] = None
            if isinstance(att_g, dict) and att_g.get("units_hit_points") is not None:
                try:
                    hp2 = int(att_g["units_hit_points"])
                except (TypeError, ValueError):
                    hp2 = None
            u = _resolve_fire_or_seam_attacker(
                state,
                engine_player=eng,
                awbw_units_id=uid,
                anchor_r=er,
                anchor_c=ec,
                target_r=dr,
                target_c=dc,
                hp_hint=hp2,
            )
        if u is None and eng in (0, 1):
            u = _resolve_fire_or_seam_attacker(
                state,
                engine_player=1 - int(eng),
                awbw_units_id=uid,
                anchor_r=er,
                anchor_c=ec,
                target_r=dr,
                target_c=dc,
                hp_hint=hp2,
            )
        if u is None:
            raise UnsupportedOracleAction(
                f"Fire: no attacker for engine P{eng} (awbw id {uid}) "
                f"at path ({sr},{sc}) / global ({ur},{uc}) / end ({er},{ec})"
                f"{_oracle_fire_no_attacker_message_suffix(state, dr, dc)}"
            )
        strike_eng = int(u.player)
        if strike_eng != eng:
            inv_pf = [
                int(pid) for pid, e in awbw_to_engine.items() if int(e) == strike_eng
            ]
            if inv_pf:
                _oracle_ensure_envelope_seat(
                    state, inv_pf[0], awbw_to_engine, before_engine_step
                )
            else:
                _oracle_advance_turn_until_player(state, strike_eng, before_engine_step)
            eng = int(state.active_player)
            if eng != strike_eng:
                raise UnsupportedOracleAction(
                    f"Fire: resolved striker on engine P{strike_eng} but active_player={eng}"
                )
        # Always select the live unit at its true tile (path start, firing tile, or drift).
        start = u.pos
        fire_pos = _oracle_resolve_fire_move_pos(state, u, paths, (er, ec), (dr, dc))
        su_id = int(u.unit_id)
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=start,
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        _engine_step(
            state,
            Action(
                ActionType.SELECT_UNIT,
                unit_pos=start,
                move_pos=fire_pos,
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        # Must match ``get_legal_actions`` / ``_apply_attack``: unit is still on
        # ``start`` until the attack step runs; ``move_pos`` is the firing tile.
        _oracle_set_combat_damage_override_from_combat_info(
            state, fire_blk, envelope_awbw_player_id, u, (dr, dc)
        )
        _engine_step(
            state,
            Action(
                ActionType.ATTACK,
                unit_pos=start,
                move_pos=fire_pos,
                target_pos=(dr, dc),
            ),
            before_engine_step,
        )
        return

    if kind == "AttackSeam":
        move_raw = obj.get("Move")
        move = move_raw if isinstance(move_raw, dict) else {}
        aseam = obj.get("AttackSeam") or {}
        if not isinstance(aseam, dict):
            raise UnsupportedOracleAction("AttackSeam without AttackSeam dict")
        seam_row = int(aseam["seamY"])
        seam_col = int(aseam["seamX"])
        target = (seam_row, seam_col)

        def _after_attack_seam() -> None:
            paths_g = (move.get("paths") or {}).get("global") or []
            if not paths_g:
                raise UnsupportedOracleAction("AttackSeam after_move without paths")
            path_end = _path_end_rc(paths_g)
            legal = get_legal_actions(state)
            er, ec = state.selected_move_pos if state.selected_move_pos is not None else path_end
            chosen = _oracle_pick_attack_seam_terminator(
                state, legal, target, path_end=path_end
            )
            if chosen is None:
                raise UnsupportedOracleAction(
                    f"AttackSeam: no ATTACK to seam {target} from {(er, ec)}; "
                    f"legal={[x.action_type.name for x in legal]}"
                )
            _engine_step(state, chosen, before_engine_step)

        paths = (move.get("paths") or {}).get("global") or []
        if not paths:
            uwrap = aseam.get("unit") or {}
            gu: dict[str, Any] = {}
            if isinstance(uwrap, dict):
                if isinstance(uwrap.get("global"), dict):
                    gu = uwrap["global"]
                else:
                    # ``unit.global`` missing: ``combatInfo`` may live only under
                    # ``units_players_id``-keyed buckets (same idea as ``Move.unit``).
                    for v in uwrap.values():
                        if isinstance(v, dict) and isinstance(v.get("combatInfo"), dict) and v[
                            "combatInfo"
                        ]:
                            gu = v
                            break
                if not gu:
                    gu = _global_unit(uwrap) if isinstance(uwrap, dict) else {}
            ci = gu.get("combatInfo") if isinstance(gu, dict) else {}
            if not isinstance(ci, dict) or not ci:
                raise UnsupportedOracleAction("AttackSeam without Move.paths.global or combatInfo")
            sr, sc = int(ci["units_y"]), int(ci["units_x"])
            uid = int(ci["units_id"])
            hp_hint_as: Optional[int] = None
            if ci.get("units_hit_points") is not None:
                try:
                    hp_hint_as = int(ci["units_hit_points"])
                except (TypeError, ValueError):
                    hp_hint_as = None
            _oracle_finish_action_if_stale(state, before_engine_step)
            # Mirror of the Fire stale-attacker guard: only short-circuit when
            # JSON hp<=0 *and* engine has no live unit by ``units_id`` *and*
            # the anchor tile is empty — otherwise hp=0 could be a post-strike
            # snapshot of a real seam strike whose attacker died to the rubble
            # counter and which we still must apply.
            if (
                hp_hint_as is not None
                and hp_hint_as <= 0
                and _unit_by_awbw_units_id(state, uid) is None
            ):
                on_anchor_seam = state.get_unit_at(sr, sc)
                if on_anchor_seam is None or not on_anchor_seam.is_alive:
                    return
            if ci.get("units_players_id") is not None:
                try:
                    upid = int(ci["units_players_id"])
                except (TypeError, ValueError):
                    upid = None
                if upid is not None and upid in awbw_to_engine:
                    _oracle_ensure_envelope_seat(
                        state, upid, awbw_to_engine, before_engine_step
                    )
            eng = int(state.active_player)
            u = _resolve_attackseam_no_path_attacker(
                state,
                eng=eng,
                awbw_units_id=uid,
                anchor_r=sr,
                anchor_c=sc,
                seam_row=seam_row,
                seam_col=seam_col,
                hp_hint=hp_hint_as,
            )
            if u is None:
                raise UnsupportedOracleAction(
                    f"AttackSeam (no path): no attacker P{eng} (awbw id {uid}) at ({sr},{sc})"
                    f"{_oracle_fire_no_attacker_message_suffix(state, seam_row, seam_col)}"
                )
            seam_eng = int(u.player)
            inv_seam = [
                int(pid) for pid, e in awbw_to_engine.items() if int(e) == seam_eng
            ]
            if inv_seam:
                _oracle_ensure_envelope_seat(
                    state, inv_seam[0], awbw_to_engine, before_engine_step
                )
            else:
                _oracle_advance_turn_until_player(state, seam_eng, before_engine_step)
            eng = int(state.active_player)
            if eng != seam_eng:
                raise UnsupportedOracleAction(
                    f"AttackSeam (no path): cannot advance to acting player P{u.player} "
                    f"(still active_player={eng})"
                )
            fr, fc = int(u.pos[0]), int(u.pos[1])
            _oracle_sync_selection_for_endpoint(
                state, u, fr, fc, fr, fc, before_engine_step
            )
            _engine_step(
                state,
                Action(
                    ActionType.ATTACK,
                    unit_pos=(fr, fc),
                    move_pos=(fr, fc),
                    target_pos=target,
                ),
                before_engine_step,
            )
            return

        _apply_move_paths_then_terminator(
            state,
            move,
            awbw_to_engine,
            before_engine_step,
            after_move=_after_attack_seam,
            envelope_awbw_player_id=envelope_awbw_player_id,
            seam_attack_target=target,
        )
        return

    if kind == "Unload":
        raw_tid = obj.get("transportID")
        if raw_tid is None:
            raise UnsupportedOracleAction("Unload without transportID")
        tid = int(raw_tid)
        gu = _oracle_unload_unit_global_for_envelope(obj, envelope_awbw_player_id)
        gu_flat = _global_unit(obj)
        if isinstance(gu_flat, dict) and gu_flat:
            if not gu:
                gu = gu_flat
            else:
                gu = _merge_move_gu_fields(gu, gu_flat)
        elif not gu:
            gu = gu_flat if isinstance(gu_flat, dict) else {}
        if not isinstance(gu, dict) or gu.get("units_players_id") is None:
            raise UnsupportedOracleAction(
                "Unload: could not resolve cargo snapshot (unit.global or per-seat unit)"
            )
        target = (int(gu["units_y"]), int(gu["units_x"]))
        cargo_ut = _name_to_unit_type(str(gu["units_name"]))
        cargo_uid: Optional[int] = None
        raw_uid = gu.get("units_id")
        if raw_uid is not None:
            try:
                cargo_uid = int(raw_uid)
            except (TypeError, ValueError):
                cargo_uid = None
        pid = int(gu["units_players_id"])
        eng = awbw_to_engine[pid]
        _oracle_finish_action_if_stale(state, before_engine_step)
        _oracle_ensure_envelope_seat(state, pid, awbw_to_engine, before_engine_step)
        _oracle_snap_active_player_to_engine(state, eng, awbw_to_engine, before_engine_step)
        if int(state.active_player) != eng:
            raise UnsupportedOracleAction(
                f"Unload for engine P{eng} but active_player={state.active_player}"
            )
        try:
            transport = _resolve_unload_transport(
                state, tid, cargo_ut, target, eng, cargo_awbw_units_id=cargo_uid
            )
        except UnsupportedOracleAction as resolve_exc:
            # Drift recovery: AWBW unloads cargo whose carrier the engine has
            # empty (engine missed an earlier ``Load`` envelope). Spawn the
            # cargo on the unload tile directly when an empty friendly carrier
            # of the right ``carry_classes`` sits orth to ``target`` — that
            # signal mirrors AWBW's emission.
            if "no transport adjacent" not in str(resolve_exc):
                raise
            if _oracle_drift_spawn_unloaded_cargo(
                state, eng, cargo_ut, target, gu, cargo_uid
            ):
                return
            raise
        su_tr = int(transport.unit_id)
        if state.action_stage == ActionStage.SELECT:
            _engine_step(
                state,
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=transport.pos,
                    select_unit_id=su_tr,
                ),
                before_engine_step,
            )
            _engine_step(
                state,
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=transport.pos,
                    move_pos=transport.pos,
                    select_unit_id=su_tr,
                ),
                before_engine_step,
            )
        elif state.action_stage == ActionStage.MOVE:
            # ``Move`` envelope ended in UNLOAD-deferred ACTION hold, but the next
            # ``Unload`` JSON can land before the second ``SELECT_UNIT`` click is
            # replayed — same envelope-split pattern as ``Capt`` / ``Fire`` no-path.
            if state.selected_unit is not transport:
                raise UnsupportedOracleAction(
                    "Unload: MOVE stage but a different unit is selected than "
                    f"transport at {transport.pos}"
                )
            _engine_step(
                state,
                Action(
                    ActionType.SELECT_UNIT,
                    unit_pos=transport.pos,
                    move_pos=transport.pos,
                    select_unit_id=su_tr,
                ),
                before_engine_step,
            )
        elif state.action_stage == ActionStage.ACTION:
            if (
                state.selected_unit is not transport
                or state.selected_move_pos != transport.pos
            ):
                _engine_step(
                    state,
                    Action(
                        ActionType.SELECT_UNIT,
                        unit_pos=transport.pos,
                        select_unit_id=su_tr,
                    ),
                    before_engine_step,
                )
                _engine_step(
                    state,
                    Action(
                        ActionType.SELECT_UNIT,
                        unit_pos=transport.pos,
                        move_pos=transport.pos,
                        select_unit_id=su_tr,
                    ),
                    before_engine_step,
                )
        else:
            raise UnsupportedOracleAction(
                f"Unload: unexpected stage {state.action_stage.name}"
            )
        legal = get_legal_actions(state)
        chosen_u: Optional[Action] = None
        for a in legal:
            if (
                a.action_type == ActionType.UNLOAD
                and a.target_pos == target
                and a.unit_type == cargo_ut
            ):
                chosen_u = a
                break
        if chosen_u is None:
            for a in legal:
                if a.action_type == ActionType.UNLOAD and a.target_pos == target:
                    chosen_u = a
                    break
        if chosen_u is None:
            by_cargo = [
                a
                for a in legal
                if a.action_type == ActionType.UNLOAD and a.unit_type == cargo_ut
            ]
            # ``unit.global`` on the carrier tile: every legal drop is one step away;
            # pick deterministically (register: oracle_unload with wrong adjacency).
            if transport.pos == target and by_cargo:
                by_cargo.sort(key=lambda a: (a.target_pos[0], a.target_pos[1]))
                chosen_u = by_cargo[0]
            elif len(by_cargo) == 1:
                chosen_u = by_cargo[0]
        if chosen_u is None:
            # Site ``unit.global`` may name a tile that is not a legal orthogonal
            # drop from this transport; pick the closest legal UNLOAD for this cargo.
            unload_same = [
                a
                for a in legal
                if a.action_type == ActionType.UNLOAD
                and a.unit_type is not None
                and a.unit_type == cargo_ut
                and a.move_pos == transport.pos
                and a.target_pos is not None
            ]
            if unload_same:
                unload_same.sort(
                    key=lambda a: (
                        abs(a.target_pos[0] - target[0]) + abs(a.target_pos[1] - target[1]),
                        a.target_pos[0],
                        a.target_pos[1],
                    )
                )
                chosen_u = unload_same[0]
        if chosen_u is None:
            # The carrier engine-side does not have a legal UNLOAD to ``target``
            # (carrier wrong tile, target stacked, etc.). Try the same drift
            # recovery used when the resolver could not find a carrier — but
            # only when the carrier is mid-ACTION; otherwise the engine has
            # already advanced past the half-turn we would corrupt.
            if state.action_stage in (ActionStage.SELECT, ActionStage.ACTION):
                # Cancel the half-finished selection so drift spawn does not
                # collide with the still-selected transport.
                if state.action_stage == ActionStage.ACTION:
                    _oracle_finish_action_if_stale(state, before_engine_step)
                if _oracle_drift_spawn_unloaded_cargo(
                    state, eng, cargo_ut, target, gu, cargo_uid
                ):
                    return
            raise UnsupportedOracleAction(
                f"Unload: no UNLOAD to {target} for {cargo_ut.name}; "
                f"legal={[x.action_type.name for x in legal]}"
            )
        _engine_step(state, chosen_u, before_engine_step)
        return

    if kind == "Delete":
        # Viewer/site echo after combat or map edits; engine state already reflects removal.
        _oracle_finish_action_if_stale(state, before_engine_step)
        return

    raise UnsupportedOracleAction(f"unsupported oracle action {kind!r}")


def apply_oracle_action_json(
    state: GameState,
    obj: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook = None,
    *,
    envelope_awbw_player_id: Optional[int] = None,
) -> None:
    """Mutate ``state`` by applying one oracle viewer JSON action.

    ``envelope_awbw_player_id`` is the ``p:`` line's AWBW ``players[].id`` (PHP int).
    When set, we align ``active_player`` with that seat before applying the JSON —
    AWBW archives can interleave both players within one ``day`` without ``End``
    between envelopes (``oracle_turn_active_player`` cluster).

    Malformed numeric scalars in the archive (notably PHP ``'?'`` placeholders in
    fog / unknown unit fields) are surfaced as :class:`UnsupportedOracleAction`
    so batch audits classify them as ``oracle_gap`` instead of ``engine_bug``.
    """
    if state.done:
        return
    kind = obj.get("action")
    try:
        _apply_oracle_action_json_body(
            state,
            obj,
            awbw_to_engine,
            before_engine_step,
            envelope_awbw_player_id=envelope_awbw_player_id,
        )
    except ValueError as e:
        if _oracle_is_python_int_literal_parse_error(e):
            raise UnsupportedOracleAction(
                "Malformed AWBW action JSON: expected integer, got non-numeric text "
                f"(action={kind!r}). Detail: {e}. "
                "Site / PHP exports sometimes use ``'?'`` for hidden or unknown "
                "``units_id``, coordinates (``units_y``/``units_x``, waypoints), "
                "``units_hit_points``, ``transportID``, ``playerID``, seam coords, etc."
            ) from e
        raise


@dataclass
class OracleReplayResult:
    final_state: GameState
    envelopes_applied: int
    actions_applied: int


def replay_oracle_zip(
    zip_path: Path,
    *,
    map_pool: Path,
    maps_dir: Path,
    map_id: int,
    co0: int,
    co1: int,
    tier_name: str,
    on_skip: Optional[Callable[[str], None]] = None,
    before_engine_step: EngineStepHook = None,
) -> OracleReplayResult:
    frames = load_replay(zip_path)
    if not frames:
        raise ValueError("empty replay")
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    map_data = load_map(map_id, map_pool, maps_dir)
    envs = parse_p_envelopes_from_zip(zip_path)
    if not envs:
        raise ValueError(
            "no per-move action stream: zip has no a<game_id> gzip entry or no p: lines "
            "(ReplayVersion 1 snapshot-only; re-download or re-export with action stream)"
        )
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=tier_name,
        replay_first_mover=first_mover,
    )
    n_act = 0
    for _pid, _day, actions in envs:
        for obj in actions:
            if state.done:
                break
            try:
                apply_oracle_action_json(
                    state,
                    obj,
                    awbw_to_engine,
                    before_engine_step=before_engine_step,
                    envelope_awbw_player_id=_pid,
                )
                n_act += 1
            except UnsupportedOracleAction as e:
                if on_skip:
                    on_skip(str(e))
                raise
            if state.done:
                break
        if state.done:
            break
    return OracleReplayResult(
        final_state=state,
        envelopes_applied=len(envs),
        actions_applied=n_act,
    )
