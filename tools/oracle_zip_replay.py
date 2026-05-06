"""
Replay **AWBW Replay Player** ``p:`` action JSON through the engine (best-effort).

Action and snapshot layouts follow the desktop viewer / site zip contract
(`github.com/DeamonHunter/AWBW-Replay-Player`); ``load_replay`` parses the same
gzipped ``awbwGame`` lines the C# app uses for timeline state.

Designed first for zips **produced by this repo** (``write_awbw_replay_from_trace``),
where Move/Build/Fire/End shapes match ``tools/export_awbw_replay_actions.py``.
Live-site oracle zips may include extra action kinds; unmapped ones raise
``UnsupportedOracleAction``. Standalone ``Load`` / ``Unload`` / ``Supply`` / ``Repair`` /
``AttackSeam`` / ``Hide`` / ``Unhide`` (Sub dive / surface & Stealth hide / unhide → ``DIVE_HIDE`` toggle) / ``Power`` are mapped;
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
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
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
from engine.unit_naming import UnknownUnitName, to_unit_type

from tools.diff_replay_zips import load_replay
from tools.export_awbw_replay import _AWBW_UNIT_NAMES

EngineStepHook = Optional[Callable[[GameState, Action], None]]


class UnsupportedOracleAction(ValueError):
    pass


class OracleFireSeamNoAttackerCandidate(UnsupportedOracleAction):
    """Exhaustive :func:`_resolve_fire_or_seam_attacker` search found no striker.

    Narrower than :class:`UnsupportedOracleAction` so call sites can retry the
    opposite ``engine_player`` without catching Lane I pin / upstream-drift errors.
    """


def _engine_step(state: GameState, act: Action, hook: EngineStepHook) -> None:
    # STEP-GATE opt-out: every oracle replay action is reconstructed from an
    # AWBW zip envelope, not from ``get_legal_actions``. Many envelopes are
    # legitimately outside the mask (capture-timer convergence, drift snaps,
    # multi-action terminators, etc.). The legality gate inside
    # ``GameState.step`` (see ``IllegalActionError``) is therefore bypassed
    # here — and only here — by passing ``oracle_mode=True``. Phase 3 plan:
    # ``.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md`` STEP-GATE.
    if hook is not None:
        hook(state, act)
    state.step(act, oracle_mode=True)


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
    - **Direct**: Prefer waypoints along ``paths.global`` from the path tail toward
      the start that are both reachable and can strike ``target`` (Phase 8 Bucket A:
      the first reachable waypoint along the reversed path can be one step short of
      the ZIP tail when the tail hex is blocked in-engine, and that penultimate tile
      may not be a valid direct-fire stance — GL **1618770**). Then snap with
      :func:`_nearest_reachable_along_path` like :func:`_apply_move_paths_then_terminator`,
      rank all reachable strike tiles, and never use the snapped end as ``move_pos``
      when it cannot hit ``target`` (avoids a misleading ``_apply_attack`` range error).
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

    def attacks_from(pos: tuple[int, int]) -> bool:
        if pos not in costs:
            return False
        # Phase 11K-FIRE-STANCE-FRIENDLY-FIX: ``compute_reachable_costs`` includes
        # friendly tiles when the mover would JOIN (same-type injured ally) or
        # LOAD into a transport — both legal *terminators* but never legal *fire
        # stances*. A Fire stance must be a tile the attacker can occupy and
        # then attack from. Picking a JOIN/LOAD friendly tile here causes the
        # engine to ``_move_unit`` the attacker through a friendly occupant on
        # the way to ``_move_unit_forced(fire_pos)``, which silently resets the
        # property's ``capture_points`` to 20 (engine assumes the unit "left"
        # the friendly's tile). GL **1635679** env 22 ai=6: friendly-INFANTRY
        # capping (7,9) had its capture wiped because the resolver returned
        # (7,9) as fire_pos for a different infantry's strike on (7,8).
        occ = state.get_unit_at(*pos)
        if occ is not None and occ is not unit:
            return False
        if _oracle_fire_stance_would_stack_on_transport(state, unit, pos):
            return False
        return (tr, tc) in get_attack_targets(state, unit, pos)

    wps: list[tuple[int, int]] = []
    for wp in paths:
        try:
            wps.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    for pos in reversed(wps):
        if attacks_from(pos):
            return pos

    er, ec = _nearest_reachable_along_path(paths, costs, path_end, start)

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
    # Phase 11K-FIRE-STANCE-FRIENDLY-FIX: same friendly-occupant exclusion as
    # ``attacks_from`` — never return a stance that would stack on a different
    # friendly (JOIN/LOAD destination, not a Fire stance).
    _occ_er = state.get_unit_at(er, ec)
    if _occ_er is None or _occ_er is unit:
        if not _oracle_fire_stance_would_stack_on_transport(state, unit, (er, ec)):
            if (tr, tc) in get_attack_targets(state, unit, (er, ec)):
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
    except (TypeError, ValueError) as e:
        raise UnsupportedOracleAction(
            f"envelope seat: bad p: player id not int-convertible: {envelope_awbw_player_id!r}"
        ) from e
    if pid not in awbw_to_engine:
        raise UnsupportedOracleAction(
            f"envelope seat: unmapped p: player id {pid} (awbw_to_engine keys={sorted(awbw_to_engine)!r})"
        )
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


# ---------------------------------------------------------------------------
# ORACLE-PATH-ONLY: AWBW "Delete Unit" reproduction
# ---------------------------------------------------------------------------
# The functions/branches in this section reproduce the AWBW player-issued
# "Delete Unit" command for replay fidelity (Phase 11J-L2-BUILD-OCCUPIED-SHIP).
# They MUST NEVER be exposed to the RL agent. There is intentionally no
# ActionType.DELETE — see engine/action.py::_RL_LEGAL_ACTION_TYPES.
# Imperator directive 2026-04-20: bot must not learn to scrap own units.
def _oracle_kill_friendly_unit(state: GameState, u: Unit) -> None:
    """Mark a friendly unit dead (Phase 11J-L2-BUILD-OCCUPIED-SHIP).

    Mirrors the engine's standard death-cleanup pattern (see
    ``GameState._end_turn`` lines around ``hp = 0`` + ``[u for u in
    self.units[p] if u.is_alive]``): set ``hp = 0`` (drives ``is_alive``
    via the ``Unit.is_alive`` property at ``engine/unit.py``) and prune
    the dead unit from ``state.units[player]`` so subsequent
    ``get_unit_at`` calls treat the tile as empty. Any cargo aboard a
    deleted transport is dropped — same convention as
    ``_apply_attack``'s post-kill cleanup, which sets cargo ``hp = 0``
    and clears ``loaded_units`` (engine/game.py around line 981–984).
    """
    p = int(u.player)
    for cargo in list(getattr(u, "loaded_units", []) or []):
        cargo.hp = 0
    u.loaded_units = []
    u.hp = 0
    state.units[p] = [x for x in state.units[p] if x.is_alive]


def _oracle_nudge_eng_occupier_off_production_build_tile(
    state: GameState,
    r: int,
    c: int,
    eng: int,
    before_engine_step: EngineStepHook,
) -> bool:
    """Move a friendly unmoved blocker off the factory tile (legal orth step + WAIT).

    If the blocker already moved or is trapped (no reachable orth neighbour),
    return ``False`` so the Build handler surfaces drift instead of teleporting.
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
    return False


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


def _oracle_merge_global_move_unit_with_seat_hints(
    uwrap: dict[str, Any],
    gl: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """Overlay per-seat unit fields when ``unit.global`` uses fog sentinels (GL 1628722).

    Global League PHP can set ``units_hit_points`` to ``"?"`` in ``global`` while the
    envelope's player bucket (key ``players[].id``) still lists the real display HP.
    Downstream parsers and drift-spawn HP clamps need a numeric scalar; without this
    merge, ``int("?")`` can abort the action or leave ``gu`` inconsistent with the seat.

    Only merges when ``units_id`` matches between ``global`` and the seat dict.
    """
    if envelope_awbw_player_id is None:
        return gl
    pid = int(envelope_awbw_player_id)
    seat = uwrap.get(str(pid))
    if not isinstance(seat, dict):
        seat = uwrap.get(pid)
    if not isinstance(seat, dict):
        return gl
    try:
        if int(seat["units_id"]) != int(gl["units_id"]):
            return gl
    except (KeyError, TypeError, ValueError):
        return gl
    out = dict(gl)
    gh = out.get("units_hit_points")
    fog_hp = gh == "?" or (isinstance(gh, str) and gh.strip() == "?")
    if not fog_hp:
        return out
    sh = seat.get("units_hit_points")
    usable = False
    if isinstance(sh, int) and not isinstance(sh, bool):
        usable = True
    elif isinstance(sh, str) and sh.strip() not in ("", "?"):
        try:
            int(sh, 10)
            usable = True
        except ValueError:
            usable = False
    elif isinstance(sh, float) and sh == sh:
        try:
            int(sh)
            usable = True
        except (TypeError, ValueError):
            usable = False
    if usable:
        out["units_hit_points"] = sh
    return out


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
        return _oracle_merge_global_move_unit_with_seat_hints(
            u, gl, envelope_awbw_player_id
        )
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


def _oracle_resolve_nested_hide_unhide_units_id(
    nested: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> Optional[int]:
    """Resolve ``units_id`` from nested ``Hide`` / ``Unhide`` when ``Move`` is absent or ``[]``.

    AWBW may omit a movement segment when the unit already sits on the dive/hide tile.
    The nested block can carry full ``unit.global`` / per-seat dicts (via
    :func:`_oracle_resolve_move_global_unit`) or a compact vision map
    ``{ awbw_player_id: units_id, ... }`` with scalar values.
    """
    umap = nested.get("unit")
    if not isinstance(umap, dict):
        return None
    syn: dict[str, Any] = {"unit": umap}
    gu = _oracle_resolve_move_global_unit(syn, envelope_awbw_player_id)
    uid_raw = gu.get("units_id")
    if uid_raw is not None:
        try:
            return int(uid_raw)
        except (TypeError, ValueError):
            pass
    if envelope_awbw_player_id is not None:
        pid = int(envelope_awbw_player_id)
        raw = umap.get(str(pid), umap.get(pid))
        if isinstance(raw, int) and not isinstance(raw, bool):
            return int(raw)
    for _k, v in umap.items():
        if _k == "global":
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, dict) and v.get("units_id") is not None:
            try:
                return int(v["units_id"])
            except (TypeError, ValueError):
                pass
    return None


def _oracle_fire_combat_info_merged(
    fire_blk: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
) -> dict[str, Any]:
    """Merge ``combatInfoVision.global`` with per-seat data when attacker is ``?`` / non-dict.

    Phase 11J-FINAL-MNF (gid 1628722): when the merged combatInfo carries a fog
    sentinel ``units_hit_points = "?"`` for the attacker or defender role, fall
    through to **every** seat view in ``combatInfoVision`` to find a numeric HP
    for the same ``units_id``. AWBW canon: the unit's own owner always sees its
    own HP (no fog-of-war on own units — AWBW Fandom Wiki Fog_of_War,
    https://awbw.fandom.com/wiki/Fog_of_War: *"You can always see all of your
    own units regardless of vision range."*). So in a fog-of-war strike where
    the global view hides the defender's HP, the defender's owner seat still
    publishes the post-strike display HP in this same envelope. Without this
    second-pass merge, ``_oracle_set_combat_damage_override_from_combat_info``
    receives ``None`` for the defender HP and ``_apply_attack`` falls back to
    ``calculate_damage`` → ``random.randint(0, 9)`` luck — diverging from the
    AWBW server's actual rolled outcome (gid 1628722 day 15 P1→P0 Md.Tank
    triple-strike: AWBW left it at hp=1, engine RNG killed it; the resolver
    then mis-mapped a day-16 Move of a different P0 Md.Tank onto the freshly
    built id=65 and the day-17 Move of id=192427925 raised "mover not found").
    """
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

    def _is_fog(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "?":
            return True
        return False

    for role in ("attacker", "defender"):
        role_dict = out.get(role)
        if not isinstance(role_dict, dict):
            continue
        if not _is_fog(role_dict.get("units_hit_points")):
            continue
        target_uid = role_dict.get("units_id")
        try:
            target_uid_int = int(target_uid) if target_uid is not None else None
        except (TypeError, ValueError):
            target_uid_int = None
        for seat_key, seat_blob in civ.items():
            if seat_key == "global":
                continue
            if not isinstance(seat_blob, dict):
                continue
            ci2 = seat_blob.get("combatInfo")
            if not isinstance(ci2, dict):
                continue
            cand = ci2.get(role)
            if not isinstance(cand, dict):
                continue
            try:
                cand_uid = cand.get("units_id")
                cand_uid_int = int(cand_uid) if cand_uid is not None else None
            except (TypeError, ValueError):
                cand_uid_int = None
            # When ``units_id`` is present on both sides, only trust matching
            # ids; otherwise (sparse seat dump) accept the seat row as long as
            # the role itself matches — the per-seat ``combatInfo`` is keyed
            # by role anyway and AWBW never re-uses a role across same-frame
            # strikes.
            if (
                target_uid_int is not None
                and cand_uid_int is not None
                and target_uid_int != cand_uid_int
            ):
                continue
            cand_hp = cand.get("units_hit_points")
            if _is_fog(cand_hp):
                continue
            new_role = dict(role_dict)
            new_role["units_hit_points"] = cand_hp
            out[role] = new_role
            break
    return out


def _oracle_assert_fire_damage_table_compatible(
    state: GameState,
    attacker: Unit,
    defender_pos: tuple[int, int],
) -> None:
    """Phase 11J P-DRIFT-DEFENDER: refuse Fire when resolver picked a defender
    the attacker has no entry for in the engine damage table.

    AWBW combat involving a unit not present on the engine board causes
    ``_oracle_fire_resolve_defender_target_pos`` to fall back to a Chebyshev-1
    ring search and pick the *nearest* engine foe — sometimes a type the
    attacker fundamentally cannot damage (FIGHTER vs TANK in GL 1631494,
    ``base_damage = None``). If we let that pass, the override-bypass in
    ``_apply_attack`` would happily apply damage derived from a different
    AWBW unit's HP delta to the wrong engine unit. Instead we raise so the
    audit reclassifies this row from ``engine_bug`` to ``oracle_gap`` —
    the truthful bucket for "oracle cannot map this strike onto its current
    engine snapshot".
    """
    from engine.combat import get_base_damage

    defender = state.get_unit_at(*defender_pos)
    if defender is None or not defender.is_alive:
        return
    if get_base_damage(attacker.unit_type, defender.unit_type) is None:
        raise UnsupportedOracleAction(
            f"Fire: oracle resolved defender type {defender.unit_type.name} "
            f"at {defender_pos} but {attacker.unit_type.name} has no damage "
            f"entry against it per the AWBW damage chart "
            f"(https://awbw.amarriner.com/damage.php; '-' for this matchup). "
            f"AWBW combatInfo may refer to a unit missing from the engine "
            f"snapshot (resolver-miss fallback; possible site/replay drift)."
        )


def _oracle_assert_fire_defender_not_friendly(
    state: GameState,
    attacker: Unit,
    defender_pos: tuple[int, int],
) -> None:
    """Phase 11J-F4-FRIENDLY-FIRE-WAVE2: refuse Fire when the engine board has a
    friendly unit at the resolved defender tile.

    AWBW does not legalize friendly fire — ``attackUnit.php`` rejects any strike
    where the attacker and defender share a player seat (this invariant is
    mirrored by ``engine.game.GameState._apply_attack`` raising ``ValueError``
    on same-player target). Therefore, any Fire envelope where the engine
    resolves a friendly defender at ``defender_pos`` is by definition an
    upstream board-state divergence (e.g. a unit that the engine should have
    moved/killed earlier still sits on the tile, or owner index drifted on a
    previous load/build). Reclassify as ``oracle_gap`` rather than letting
    ``_apply_attack`` raise ``engine_bug``.

    Mirrors :func:`_oracle_assert_fire_damage_table_compatible`'s "snapshot
    drift" attribution: previously these rows surfaced as
    ``"_apply_attack: friendly fire from player N on X at (r,c)"`` in the
    desync register (Phase 11J-V2-936-AUDIT engine_bug rows 1629202, 1632825;
    both first-divergence migrations after MOVE-TRUNCATE-SHIP collapsed
    upstream Move drift that previously hid them).
    """
    defender = state.get_unit_at(*defender_pos)
    if defender is None or not defender.is_alive:
        return
    if int(defender.player) != int(attacker.player):
        return
    raise UnsupportedOracleAction(
        f"Fire: engine board holds friendly {defender.unit_type.name} at "
        f"{defender_pos} for attacker P{attacker.player} {attacker.unit_type.name} "
        f"id={attacker.unit_id} — AWBW Fire envelopes never legalize same-player "
        f"strikes (attackUnit.php rejects), so the engine snapshot has drifted "
        f"upstream (owner mis-mapped or stale unit on the defender tile). "
        f"Treat as oracle_gap (snapshot drift) rather than engine friendly-fire."
    )


def _oracle_set_combat_damage_override_from_combat_info(
    state: GameState,
    fire_blk: dict[str, Any],
    envelope_awbw_player_id: Optional[int],
    attacker: Unit,
    defender_pos: tuple[int, int],
    awbw_to_engine: Optional[dict[int, int]] = None,
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
    "no unit" drift cluster within ~3 turns). When AWBW HPs are missing or
    non-numeric, raise — do not fall back to engine RNG. Seam attacks
    (no defender unit) are not affected: this helper is only wired into the
    ``Fire`` paths.

    Phase 11J-CLOSE-1624082 — Sasha War Bonds pin: when ``awbw_to_engine``
    is supplied, also pin the per-fire War Bonds payout from PHP's
    ``combatInfoVision.global.combatInfo.gainedFunds`` block (a dict
    ``{awbw_player_id_str: gold | None}``). Engine consumes the pin in
    ``_apply_war_bonds_payout`` (one-shot per Fire). This is the canonical
    fix for game ``1624082`` env 33: Sasha's primary fires under the
    SCOP credited 500 g less than PHP because four engine defenders sat
    at a different pre-strike display HP than PHP (state-mismatch on
    pre-HP — combat-info HP override pulls them to the same post-HP, but
    pre-HP can still differ). PHP itself emits the per-fire WB credit in
    the same combatInfo block; using it directly removes the entire
    state-mismatch failure mode. See ``GameState
    ._oracle_war_bonds_payout_override`` for the empirical anchor and
    sign-corrected per-act table.
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

    def _refine_defender_hp_with_pin(base: Optional[int], pin_hp: int) -> int:
        """Merge defender post-envelope pin with lossy combatInfo ×10.

        Same ``frames[env_i+1]`` hazard as Phase 11K attacker pin, plus PHP
        day-start repair landing in the trailing snapshot while combatInfo is
        still post-strike (gid 1611364 env 21): reject ``pin_hp`` when it is a
        full display step above ``base`` (``pin_hp > base + 9``).  Sub-bar
        refinement (<10 internal) still trusts the pin (1635679).
        """
        if base is None:
            return pin_hp
        if abs(pin_hp - base) < 10:
            return pin_hp
        if pin_hp > base + 9:
            return base
        return pin_hp

    awbw_def_hp = _to_internal(def_ci.get("units_hit_points")) if isinstance(def_ci, dict) else None
    awbw_att_hp = _to_internal(att_ci.get("units_hit_points")) if isinstance(att_ci, dict) else None

    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — recover sub-display-HP counter
    # damage. AWBW combat is float; ``combatInfo.units_hit_points`` is the
    # integer display HP (= ``ceil(internal/10)``), which hides counters
    # of 1–9 internal HP. When the audit caller has populated
    # ``state._oracle_post_envelope_units_by_id`` with the post-envelope
    # snapshot's ``round(units.hit_points * 10)`` per AWBW ``units_id``,
    # consult it first to override the lossy display × 10 conversion. Per
    # AWBW combat rules (Wars Wiki Damage_Formula), the **attacker** can
    # take counter damage at most once per envelope (units act once /
    # turn), so the post-envelope HP for the attacker is unambiguously
    # the post-counter HP. The defender of a single ``Fire`` row may be
    # struck by multiple attackers within the same envelope (each with
    # its own ``combatInfo``), so post-envelope HP for the defender can
    # double-count: keep the per-fire ``combatInfo.units_hit_points`` × 10
    # for the defender side. Anchor: gid 1635679 env 10 RECON id
    # 192721109; closeout in
    # ``docs/oracle_exception_audit/phase11k_fire_frac_counter.md``.
    pin = getattr(state, "_oracle_post_envelope_units_by_id", None)
    if pin and isinstance(att_ci, dict):
        try:
            att_uid = int(att_ci.get("units_id"))
        except (TypeError, ValueError):
            att_uid = None
        if att_uid is not None and att_uid in pin:
            pin_hp = int(pin[att_uid])
            if awbw_att_hp is None:
                awbw_att_hp = pin_hp
            elif (
                pin_hp == 100
                and awbw_att_hp is not None
                and awbw_att_hp < 100
            ):
                # 1611364 env 21: trailing frame reflects post-Join full HP on
                # the striker while per-fire combatInfo is still post-counter.
                # A broad ``pin > base + N`` gate regressed legitimate strikes
                # (e.g. GL 1627324) where the pin is materially above combatInfo
                # for non-heal reasons — full internal HP is the safe sentinel.
                pass
            elif abs(pin_hp - awbw_att_hp) < 10:
                awbw_att_hp = pin_hp
            else:
                awbw_att_hp = pin_hp

    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — defender-side fractional pin.
    # The defender's post-envelope HP is unambiguous when this defender is
    # struck by exactly one ``Fire`` row in the envelope (units act once /
    # turn — a single defender can be hit by multiple attackers in the
    # same envelope, but in long-range / counter-rich Sturm replays the
    # common case is 1:1). The audit caller pre-scans the envelope and
    # stamps multi-hit defender ``units_id``s into
    # ``_oracle_post_envelope_multi_hit_defenders``; for those we keep the
    # integer display × 10 (combatInfo) which is per-fire ground truth.
    # Anchor: gid 1635679 env 25 day 13 TANK (13,13) id 192746072 — single
    # Hawke fire, PHP hp=4.2 (= 42 internal), engine over-rounded to 50
    # before this pin. Sturm-day-13 over-repair (+1600 g vs PHP +800 g)
    # collapses once defender HP carries the fractional residue.
    multi_def = getattr(state, "_oracle_post_envelope_multi_hit_defenders", None) or set()
    if pin and isinstance(def_ci, dict):
        try:
            def_uid = int(def_ci.get("units_id"))
        except (TypeError, ValueError):
            def_uid = None
        if def_uid is not None and def_uid in pin and def_uid not in multi_def:
            pin_hp = int(pin[def_uid])
            awbw_def_hp = _refine_defender_hp_with_pin(awbw_def_hp, pin_hp)

    dmg: Optional[int] = None
    counter: Optional[int] = None
    if awbw_def_hp is not None and defender_unit is not None and defender_unit.is_alive:
        dmg = max(0, int(defender_unit.hp) - awbw_def_hp)
    if awbw_att_hp is not None and attacker is not None and attacker.is_alive:
        counter = max(0, int(attacker.hp) - awbw_att_hp)

    if dmg is None and counter is None:
        raise UnsupportedOracleAction(
            "Fire: combatInfo missing numeric attacker/defender units_hit_points; "
            "cannot pin damage/counter to AWBW (oracle would fall back to engine RNG)"
        )
    state._oracle_combat_damage_override = (dmg, counter)
    _oracle_record_defender_killed(state, def_ci if isinstance(def_ci, dict) else {})

    # Phase 11J-CLOSE-1624082: pin War Bonds payout from PHP gainedFunds.
    if awbw_to_engine is not None:
        gf = fi.get("gainedFunds")
        if isinstance(gf, dict) and gf:
            wb_pin: dict[int, int] = {}
            for raw_pid, raw_gold in gf.items():
                if raw_gold is None:
                    continue
                try:
                    pid = int(raw_pid)
                    gold = int(raw_gold)
                except (TypeError, ValueError):
                    continue
                if gold <= 0:
                    continue
                eng_pid = awbw_to_engine.get(pid)
                if eng_pid is None:
                    continue
                wb_pin[int(eng_pid)] = gold
            if wb_pin:
                state._oracle_war_bonds_payout_override = wb_pin
            else:
                state._oracle_war_bonds_payout_override = None
        else:
            state._oracle_war_bonds_payout_override = None


def _oracle_fire_chebyshev1_neighbours(r: int, c: int) -> list[tuple[int, int]]:
    """Eight tiles adjacent to ``(r, c)`` (Chebyshev distance 1, excluding centre)."""
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _oracle_fire_defender_row_is_postkill_noop(
    state: GameState,
    defender: dict[str, Any],
    *,
    attacker_engine_player: Optional[int] = None,
) -> bool:
    """AWBW duplicate ``Fire`` after the defender died: JSON hp<=0 and tile empty (GL 1628985).

    Phase 11J-FIRE-MOVE-TERMINATOR-FINAL: when JSON already shows a dead defender
    but the engine tile holds a **live same-player** unit (silent-skip drift),
    treat as the same post-kill duplicate family as empty tiles — see
    phase11j_move_truncate_ship.md / phase11j_lane_l_widen_ship.md (Lane L /
    Fire-snap drift). Caller passes ``attacker_engine_player`` from the acting
    seat; omit it to preserve legacy two-arg behaviour byte-for-byte.
    """
    try:
        ry = int(defender["units_y"])
        rx = int(defender["units_x"])
    except (KeyError, TypeError, ValueError):
        return False
    raw_hp = defender.get("units_hit_points")
    if raw_hp is None:
        return False
    try:
        hp_i = int(raw_hp)
    except (TypeError, ValueError):
        return False
    if hp_i > 0:
        return False
    occ = state.get_unit_at(ry, rx)
    if occ is None or not occ.is_alive:
        return True
    if attacker_engine_player is not None:
        try:
            if int(occ.player) == int(attacker_engine_player):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _oracle_get_killed_awbw_ids(state: GameState) -> set[int]:
    """Per-state set of AWBW ``units_id`` values that the oracle has already
    applied as killed by a prior ``Fire`` row in this replay.

    Lazily attached to the state so we don't have to extend the engine
    dataclass. The zip-replay path does not stamp engine ``unit_id`` with the
    AWBW ``units_id``, so ``_unit_by_awbw_units_id`` cannot tell us whether a
    ``hp=0`` defender row is the original killing strike or a duplicate
    re-emit. The only ground truth available in this lane is whether *we*
    have already applied a strike on that AWBW id.
    """
    s = getattr(state, "_oracle_killed_awbw_units_ids", None)
    if s is None:
        s = set()
        state._oracle_killed_awbw_units_ids = s  # type: ignore[attr-defined]
    return s


def _oracle_record_defender_killed(state: GameState, defender: dict[str, Any]) -> None:
    """Record the JSON defender's AWBW ``units_id`` if AWBW reports it dead.

    Called from the combat-damage override helper, which runs immediately
    before every ``Fire`` envelope's ATTACK step. Idempotent: re-adding an
    id is a no-op.
    """
    if not isinstance(defender, dict):
        return
    raw_hp = defender.get("units_hit_points")
    if raw_hp is None:
        return
    try:
        if int(raw_hp) > 0:
            return
    except (TypeError, ValueError):
        return
    raw_id = defender.get("units_id")
    if raw_id is None:
        return
    try:
        did = int(raw_id)
    except (TypeError, ValueError):
        return
    _oracle_get_killed_awbw_ids(state).add(did)


def _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied(
    state: GameState,
    defender: dict[str, Any],
    attacker_eng: Optional[int] = None,
    attacker_anchor: Optional[tuple[int, int]] = None,
) -> bool:
    """Duplicate no-path ``Fire`` after the victim disappeared but another unit holds the tile.

    The original gate ``_unit_by_awbw_units_id(state, did) is None`` only works
    in the live-audit lane (``tools/desync_audit_amarriner_live``) where
    engine units carry AWBW ids. In the zip-replay lane engine units use small
    sequential ``unit_id``s, so that lookup is *always* ``None`` and the
    orphan check turned into "skip whenever any unit is on the recorded tile",
    which silently swallowed the legitimate killing strike (GL **1628985**:
    Artillery 192111507 -> Black Boat 192111511 at engine (10,9) on env 14).

    The faithful signals available in this lane:

    1. **Friendly occupant.** If the engine occupant is friendly to the JSON
       attacker, it cannot be the defender (drift case).
    2. **Already-killed id.** If the AWBW ``units_id`` is in the per-state
       kill set, an earlier Fire row in this replay already took it to 0
       hp; this row is a duplicate re-emit (lane B GL **1631194**).
    3. **Attacker anchor missing.** If the JSON attacker has no friendly
       unit at the recorded anchor, the engine state has drifted enough
       that the row's strike is unsafe to apply (GL **1631858** env 26
       j=16: AWBW expects a P0 attacker at engine (6,10) but engine has
       a P1 Tank stuck there from earlier turns; applying would target
       the wrong defender). Skip.
    4. **First kill, enemy occupant, attacker present.** Otherwise the
       engine occupant is most plausibly the original defender — apply
       the strike (GL 1628985).
    """
    try:
        ry = int(defender["units_y"])
        rx = int(defender["units_x"])
    except (KeyError, TypeError, ValueError):
        return False
    raw_hp = defender.get("units_hit_points")
    if raw_hp is None:
        return False
    try:
        hp_i = int(raw_hp)
    except (TypeError, ValueError):
        return False
    if hp_i > 0:
        return False
    raw_id = defender.get("units_id")
    if raw_id is None:
        return False
    try:
        did = int(raw_id)
    except (TypeError, ValueError):
        return False
    occ = state.get_unit_at(ry, rx)
    if occ is None or not occ.is_alive:
        return False
    if attacker_eng is not None and int(occ.player) == int(attacker_eng):
        return True
    if did in _oracle_get_killed_awbw_ids(state):
        return True
    if _unit_by_awbw_units_id(state, did) is not None:
        return False
    if attacker_eng is not None and attacker_anchor is not None:
        ar, ac = int(attacker_anchor[0]), int(attacker_anchor[1])
        au = state.get_unit_at(ar, ac)
        if au is None or not au.is_alive or int(au.player) != int(attacker_eng):
            return True
    return False


def _oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air(
    state: GameState,
    defender: dict[str, Any],
    sr: int,
    sc: int,
    dr: int,
    dc: int,
) -> bool:
    """Skip stale no-path ``Fire`` when orphan air defender hp is 1-2 and tile holds an unrelated live unit.

    Replay exports sometimes duplicate ``Move: []`` / ``Fire`` rows while the defender
    row is orphaned (lane B **1632124**, **1631068** resolves without this when the chart
    already damages air from the anchor). The original gate combined the orphan condition
    with ``get_base_damage(...) is None`` to scope the no-op to engine builds that could
    not model the strike at all. After agent4 filled Infantry/Mech vs B-/T-Copter
    (see ``data/damage_table.json``), the chart-None gate evaporates for the foot-vs-rotor
    case — but the underlying defect (PHP defender already dead, engine tile reoccupied
    by an unrelated live unit) still demands a no-op: re-firing would strike the wrong
    unit. The gate is therefore class-based (``air`` / ``copter``) on the orphan-tile-
    occupant, independent of chart contents. Lane B fixtures continue to assert this
    branch via ``test_low_hp_orphan_vs_air_when_get_base_damage_none`` (renamed
    semantically to "air" in docs; symbol kept stable for downstream callers).
    """
    raw_hp = defender.get("units_hit_points")
    raw_id = defender.get("units_id")
    if raw_hp is None or raw_id is None:
        return False
    try:
        hp_i = int(raw_hp)
        oid = int(raw_id)
    except (TypeError, ValueError):
        return False
    if hp_i not in (1, 2):
        return False
    if _unit_by_awbw_units_id(state, oid) is not None:
        return False
    ax = state.get_unit_at(int(sr), int(sc))
    tg = state.get_unit_at(int(dr), int(dc))
    if ax is None or tg is None or not ax.is_alive or not tg.is_alive:
        return False
    stg = UNIT_STATS[tg.unit_type]
    return stg.unit_class in ("air", "copter")


def _oracle_credit_skipped_fire_gained_funds(
    state: GameState,
    fire_blk: dict[str, Any],
    awbw_to_engine: Optional[dict[int, int]],
    envelope_awbw_player_id: Optional[int],
) -> None:
    """Phase 11J-CLOSE-1624082-WB-SKIP-CREDIT — credit PHP-recorded War Bonds
    when the no-path ``Fire`` row is silently skipped due to engine-side
    defender drift.

    AWBW canon (Tier 1, AWBW CO Chart Sasha row,
    https://awbw.amarriner.com/co.php):
      *"War Bonds — Returns 50% of damage dealt as funds (subject to a 9HP
      cap)."*

    Background: the no-path ``Fire`` branch (this module, ``kind == "Fire"``
    when ``not paths``) silently early-returns in several drift cases —
    ``_oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air`` is the most
    common, where the AWBW defender is a 1-2 HP orphan and the engine tile
    holds an unrelated live air/copter. Re-firing through the engine would
    strike the wrong unit, so the strike itself MUST be skipped. But PHP's
    ``combatInfoVision.global.combatInfo.gainedFunds`` records a real funds
    credit for the strike — silently dropping it strands War Bonds payouts
    that materialise as ``Build no-op (insufficient funds)`` later in the
    same envelope.

    Empirical anchor: gid ``1624082`` env 33 (Sasha day 17) Fire ``[6]``
    INF (14,11) hp=1 -> orphan B_COPTER (13,11) hp=2; PHP credits Sasha
    ``gainedFunds={'3753713': 450}`` (1 display HP loss × 9000 cost / 20).
    Engine pre-fix: silent skip drops 450 g, then ``Build NEO_TANK`` at
    (13,3) needs 22 000 g vs treasury 21 900 g — 100 g shortfall surfaces
    as ``oracle_gap`` in the canonical 936 audit. With the credit applied,
    treasury becomes 22 350 g and the build succeeds.

    Hybrid crediting model mirrors :meth:`GameState._apply_war_bonds_payout`:
    when the dealer is the active player (her own SCOP turn), credit
    immediately so in-turn builds can spend the bonds; otherwise defer to
    ``pending_war_bonds_funds`` for end-of-opp-turn settlement.

    Only credits when the dealer's CO is Sasha (co_id 19) and her War Bonds
    window is active — matches the engine formula path. Non-Sasha
    ``gainedFunds`` (e.g. Hachi Merchant Union, Colin Power of Money — both
    funds via different mechanisms) are not currently routed through this
    helper to avoid double-credit; if a future cluster surfaces those, the
    same hybrid pattern can extend.
    """
    if not isinstance(fire_blk, dict):
        return
    fi = _oracle_fire_combat_info_merged(fire_blk, envelope_awbw_player_id)
    if not isinstance(fi, dict):
        return
    gf = fi.get("gainedFunds")
    if not isinstance(gf, dict) or not gf:
        return
    if not awbw_to_engine:
        return
    for raw_pid, raw_gold in gf.items():
        if raw_gold is None:
            continue
        try:
            pid = int(raw_pid)
            gold = int(raw_gold)
        except (TypeError, ValueError):
            continue
        if gold <= 0:
            continue
        eng_pid_raw = awbw_to_engine.get(pid)
        if eng_pid_raw is None:
            continue
        eng_pid = int(eng_pid_raw)
        if eng_pid not in (0, 1):
            continue
        co = state.co_states[eng_pid]
        if co.co_id != 19 or not co.war_bonds_active:
            continue
        if eng_pid == int(state.active_player):
            state.funds[eng_pid] = min(999_999, state.funds[eng_pid] + gold)
        else:
            co.pending_war_bonds_funds += gold


def _oracle_fire_indirect_defender_from_attack_ring(
    state: GameState,
    *,
    aeng: int,
    anchor: tuple[int, int],
    record: tuple[int, int],
    hint_hp: Optional[int],
) -> Optional[tuple[int, int]]:
    """When vision names a Chebyshev neighbour, snap to a foe in the indirect Manhattan ring.

    Indirect units cannot strike Chebyshev-adjacent tiles; :func:`get_attack_targets`
    from the declared anchor is authoritative (GL **1609533**: artillery at ``(5,3)``
    vs defender recorded one step inside minimum range).
    """
    ar, ac = int(anchor[0]), int(anchor[1])
    u_a = state.get_unit_at(ar, ac)
    if u_a is None or not u_a.is_alive or int(u_a.player) != int(aeng):
        return None
    st = UNIT_STATS[u_a.unit_type]
    if not st.is_indirect:
        return None
    ring = get_attack_targets(state, u_a, u_a.pos)
    if not ring:
        return None
    pr, pc = int(record[0]), int(record[1])

    def _foe_at(tr: int, tc: int) -> Optional[Unit]:
        eu = state.get_unit_at(tr, tc)
        if eu is None or not eu.is_alive or int(eu.player) == int(aeng):
            return None
        return eu

    ring_foes: list[tuple[int, int, Unit]] = []
    for tr, tc in ring:
        eu = _foe_at(tr, tc)
        if eu is not None:
            ring_foes.append((tr, tc, eu))
    if not ring_foes:
        return None
    # Phase 11J-CLOSE-1617442 — when the recorded defender tile (``record``) is
    # itself in the indirect's strike ring AND holds a foe, prefer it. The
    # ``hint_hp`` heuristic below compares engine PRE-strike ``display_hp`` to
    # AWBW's ``combatInfo.units_hit_points``, which AWBW emits as the
    # **post-strike** HP (https://awbw.fandom.com/wiki/Damage_Formula —
    # ``combatInfoVision.combatInfo.{attacker,defender}.units_hit_points``).
    # That mismatch causes the ±1 relaxed band to snap to a *neighbour* foe
    # whose pre-strike display equals the actual defender's post-strike value
    # (gid 1617442 env 41 j=10: Artillery (4,7) → Infantry (5,6) hp10 mis-snapped
    # to Infantry (5,4) hp7 because hint=8 matched (5,4)'s display 7 within ±1
    # while the real defender at display 10 was 2 outside). The recorded tile is
    # AWBW's authoritative target; only fall through to the ring-foe heuristic
    # when ``record`` is unreachable for the indirect — see GL 1609533 docstring.
    rec_in_ring = next(
        ((tr, tc) for tr, tc, _ in ring_foes if tr == pr and tc == pc),
        None,
    )
    if rec_in_ring is not None:
        return rec_in_ring
    # GL 1629722: ``units_hit_points == 0`` is post-strike (defender destroyed).
    # Comparing ``want == 0`` to alive units' ``display_hp`` never matches; the
    # relaxed band can still pick a **neighbour** in ``ring_foes`` (e.g. Infantry
    # one tile off) when the true victim sat on ``record`` but was outside the
    # engine's computed indirect ``ring`` (LOS / min-range edge vs AWBW). Let
    # :func:`_oracle_fire_resolve_defender_target_pos` use ``get_unit_at(record)``
    # and Chebyshev / id fallbacks instead of snapping to the wrong ring tile.
    if hint_hp is not None and int(hint_hp) == 0:
        return None
    if hint_hp is not None:
        want = int(hint_hp)
        strict = [(tr, tc, u) for tr, tc, u in ring_foes if int(u.display_hp) == want]
        if len(strict) == 1:
            return (strict[0][0], strict[0][1])
        relaxed = [
            (tr, tc, u)
            for tr, tc, u in ring_foes
            if _repair_display_hp_matches_hint(int(u.display_hp), want)
        ]
        if len(relaxed) == 1:
            return (relaxed[0][0], relaxed[0][1])
    if len(ring_foes) == 1:
        return (ring_foes[0][0], ring_foes[0][1])
    ring_foes.sort(
        key=lambda t: (
            abs(t[0] - pr) + abs(t[1] - pc),
            t[0],
            t[1],
            int(t[2].unit_id),
        )
    )
    return (ring_foes[0][0], ring_foes[0][1])


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

    if attacker_anchor is not None:
        ir_snap = _oracle_fire_indirect_defender_from_attack_ring(
            state,
            aeng=aeng,
            anchor=attacker_anchor,
            record=(dr, dc),
            hint_hp=hint_hp,
        )
        if ir_snap is not None:
            return ir_snap

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


def _repair_pairs_filter_by_php_hit_points_hint(
    pairs: list[tuple[Unit, Unit]], w: int
) -> list[tuple[Unit, Unit]]:
    """Narrow (Black Boat, ally) pairs using PHP ``repaired.global.units_hit_points``.

    The hint is the **post**-heal display bar after a Black Boat +1 bar tick, so
    the canonical **pre**-heal display is ``w - 1``. A plain ±1 fuzzy match
    would also admit ``w + 1`` (e.g. full-HP allies at display 10 when ``w=9``),
    picking the wrong boat+ally when several Black Boats are on the map (GL
    1624307 env 36).
    """
    want_pre = max(0, min(10, w - 1))
    pre = [(b, t) for b, t in pairs if t.display_hp == want_pre]
    if pre:
        return pre
    exact = [(b, t) for b, t in pairs if t.display_hp == w]
    if exact:
        return exact
    return [
        (b, t) for b, t in pairs if _repair_display_hp_matches_hint(t.display_hp, w)
    ]


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
            # Several Black Boats each sit ``best_d`` tiles from a different same-HP ally
            # (GL 1624764: three INF at display 10 all Manhattan 3 from distinct BBs). The
            # old ``continue`` left ``_oracle_fallback_nearest_allied_repair_target_pos`` as
            # ``None`` and broke ``Repair`` resolution. Deterministic (boat_id, ally_id) tie
            # break matches AWBW's stable id ordering in practice for this class of ties.
            chosen = min(tops, key=lambda t: (int(t[1].unit_id), int(t[2].unit_id)))
        # Black Boats cover long sea legs; long-range drift vs AWBW is surfaced, not forced.
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
        pairs = _repair_pairs_filter_by_php_hit_points_hint(pairs, int(hp_key))
    # Site ``units_hit_points`` can disagree with bars after prior-step drift;
    # restore unfiltered pairs when HP hints eliminate everyone.
    if len(pairs) == 0 and len(pairs_unc) > 0:
        pairs = pairs_unc
    if len(pairs) > 1:
        tposes = {t.pos for _, t in pairs}
        if len(tposes) == 1:
            return next(iter(tposes))
        if hp_key is not None:
            w = int(hp_key)
            want_pre = max(0, min(10, w - 1))
            hp_pairs = [(b, t) for b, t in pairs if t.display_hp == want_pre]
            if len(hp_pairs) != 1:
                hp_pairs = [
                    (b, t)
                    for b, t in pairs
                    if _repair_display_hp_matches_hint(t.display_hp, w)
                ]
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
            loose = _repair_pairs_filter_by_php_hit_points_hint(loose, int(hp_key))
        if len(loose) == 0 and len(loose_unc) > 0:
            loose = loose_unc
        if len(loose) > 1:
            tposes = {t.pos for _, t in loose}
            if len(tposes) == 1:
                return next(iter(tposes))
            if hp_key is not None:
                w = int(hp_key)
                want_pre = max(0, min(10, w - 1))
                hp_loose = [(b, t) for b, t in loose if t.display_hp == want_pre]
                if len(hp_loose) != 1:
                    hp_loose = [
                        (b, t)
                        for b, t in loose
                        if _repair_display_hp_matches_hint(t.display_hp, w)
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


def oracle_set_php_id_tile_cache(state: GameState, frame: dict[str, Any]) -> None:
    """Cache AWBW ``units[].id`` → (row, col) from a PHP snapshot **before** an envelope.

    Predeployed engine ``unit_id`` values do not match PHP database ids, so
    ``Delete``'s ``unitId.global`` cannot resolve via :func:`_unit_by_awbw_units_id`.
    The pre-envelope frame's tile for that id is authoritative for which friendly
    occupier to scrap (gid 1628198: Delete 192112547 before Jess SCOP).
    """
    m: dict[int, tuple[int, int]] = {}
    for u in (frame.get("units") or {}).values():
        if not isinstance(u, dict):
            continue
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        try:
            uid = int(u.get("id") or 0)
            row = int(u["y"])
            col = int(u["x"])
        except (KeyError, TypeError, ValueError):
            continue
        if uid > 0:
            m[uid] = (row, col)
    setattr(state, "_oracle_php_id_to_rc", m)


def _oracle_sole_dive_hide_actor_for_player(state: GameState, eng: int) -> Optional[Unit]:
    """Fallback when AWBW ``units_id`` does not match engine ``unit_id`` (predeploy uses
    monotonic ids) and the compact ``Move: []`` envelope carries no tile mirror.

    If exactly one alive dive/hide-capable unit (``UNIT_STATS.*.can_dive``) exists
    for ``eng``, it is the unique candidate for ``DIVE_HIDE`` / hide in place.
    """
    hits: list[Unit] = []
    for u in state.units[eng]:
        if not u.is_alive:
            continue
        if not UNIT_STATS[u.unit_type].can_dive:
            continue
        hits.append(u)
    if len(hits) == 1:
        return hits[0]
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

    When ``_unit_by_awbw_units_id`` finds the declared ``units_id`` on the active
    engine seat but that unit cannot strike the resolved defender, we **do not**
    substitute a different friendly (Phase 8 Lane I): that produced misleading
    ``_apply_attack`` errors when AWBW pinned one unit id and another unit was
    closer in the old tie-break.
    """
    from engine.action import _can_attack_submerged_or_hidden
    from engine.combat import get_base_damage

    eng = int(engine_player)
    tr, tc = int(target_r), int(target_c)
    target_rc = (tr, tc)
    ar, ac = int(anchor_r), int(anchor_c)

    u_id = _unit_by_awbw_units_id(state, int(awbw_units_id))
    pin_active = u_id is not None and int(u_id.player) == eng
    id_pin = int(awbw_units_id)
    if u_id is not None and int(u_id.player) == eng:
        if _oracle_unit_can_attack_target_cell(state, u_id, eng, target_rc):
            return u_id
        u_grit = _oracle_unit_grit_jake_probe_for_target(state, u_id, eng, target_rc)
        if u_grit is not None:
            return u_grit

    def _anchor_unit_matches_awbw_id(xu: Unit) -> bool:
        return not pin_active or int(xu.unit_id) == id_pin

    x = state.get_unit_at(ar, ac)
    if x is not None and x.is_alive:
        cross_hit = False
        for mp in _oracle_fire_attack_move_pos_candidates(state, x):
            if target_rc in get_attack_targets(state, x, mp):
                cross_hit = True
                break
        if cross_hit:
            if int(x.player) == eng:
                if _anchor_unit_matches_awbw_id(x):
                    return x
            # Legal strike from anchor but occupant is the other engine seat (envelope lag).
            du = state.get_unit_at(tr, tc)
            if du is None or int(du.player) != int(x.player):
                if _anchor_unit_matches_awbw_id(x):
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
    if pin_active and cands:
        # ``u_id`` is on ``eng`` but cannot strike ``target_rc``; any ``cands`` entry
        # must be a different engine unit (same AWBW id cannot appear twice). Picking
        # the closest alternate caused Bucket B wrong-attacker ``_apply_attack`` noise.
        raise UnsupportedOracleAction(
            f"Fire: AWBW attacker units_id {awbw_units_id} not among eligible strikers "
            f"for target ({tr},{tc}); refusing alternate unit (upstream drift)"
        )
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
                # AWBW canon: direct units must be Manhattan-1 (orthogonally
                # adjacent) to the defender — never diagonally. Phase 6 bug
                # fix: the prior `max(abs(...))==1` (Chebyshev) admitted
                # diagonal candidates that AWBW would never have produced.
                ok_adj = False
                for er, ec in _oracle_fire_attack_move_pos_candidates(state, cand):
                    if abs(er - tr) + abs(ec - tc) == 1:
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
        if pin_active and adj_one:
            id_adj = [c for c in adj_one if int(c.unit_id) == id_pin]
            if len(id_adj) == 1:
                return id_adj[0]
            raise UnsupportedOracleAction(
                f"Fire: AWBW attacker units_id {awbw_units_id} not among eligible strikers "
                f"for target ({tr},{tc}); refusing alternate unit (upstream drift)"
            )
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
    raise OracleFireSeamNoAttackerCandidate(
        f"Fire/seam: no attacker candidate for awbw id {awbw_units_id} anchor=({anchor_r},{anchor_c}) "
        f"target=({target_r},{target_c}) hp_hint={hp_hint!r}"
    )


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
        try:
            return _resolve_fire_or_seam_attacker(
                state,
                engine_player=e_try,
                awbw_units_id=int(awbw_units_id),
                anchor_r=ar,
                anchor_c=ac,
                target_r=tr,
                target_c=tc,
                hp_hint=hp_hint,
            )
        except OracleFireSeamNoAttackerCandidate:
            continue

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
    """Append to ``UnsupportedOracleAction`` when attacker resolution fails (no unit)."""
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
    """Resolve a site/PHP/JSON unit-name string to a UnitType.

    Phase 11Z: thin wrapper around ``engine.unit_naming.to_unit_type``.
    All alias dicts (formerly local) live in ``engine/unit_naming.py``;
    add new spellings there. The oracle-error contract (``UnsupportedOracleAction``
    on unknown name) is preserved here so callers do not need to change.
    """
    try:
        return to_unit_type(name)
    except UnknownUnitName:
        raise UnsupportedOracleAction(f"unknown AWBW unit name {name!r}") from None


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
        pid = _oracle_awbw_scalar_int_optional(p.get("id"))
        if pid is None:
            continue
        order_raw = p.get("order", 0)
        # PHP may use 'N' placeholder for hidden/unknown fields (fog, unknown CO).
        # ``_oracle_awbw_scalar_int_optional`` swallows these safely elsewhere;
        # do the same here so fogged games don't crash the audit harness.
        order = _oracle_awbw_scalar_int_optional(order_raw)
        if order is None:
            order = 0
        cid_raw = p.get("co_id")
        cid = _oracle_awbw_scalar_int_optional(cid_raw)
        if cid is None:
            # Fogged or unknown CO — use the caller-supplied co0/co1 as fallback.
            # We still need a stable order; use the order field (0/1) to assign.
            cid = co0 if order == 0 else co1
        rows.append((order, pid, cid))
    rows.sort(key=lambda t: t[0])
    if len(rows) < 2:
        raise ValueError(f"expected >=2 players in snapshot, got {rows!r}")
    rows = rows[:2]
    (_, id_a, c_a), (_, id_b, c_b) = rows
    # Fix: compare CO ids (c_a, c_b) to (co0, co1) to determine seat mapping
    # The player with co0 gets seat 0, the player with co1 gets seat 1
    if c_a == co0 and c_b == co1:
        return {id_a: 0, id_b: 1}
    elif c_a == co1 and c_b == co0:
        return {id_a: 1, id_b: 0}
    else:
        # Fallback: use CO id matching
        if c_a == co0:
            return {id_a: 0, id_b: 1}
        else:
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
        if raw is not None and str(raw).strip():
            raise
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


def _oracle_move_med_tank_label_engine_tank_drift(
    state: GameState,
    eng: int,
    declared_mover_type: Optional[UnitType],
    paths: list[Any],
    gu: dict[str, Any],
    path_start: tuple[int, int],
    global_rc: tuple[int, int],
    path_end: tuple[int, int],
) -> Optional[Unit]:
    """Resolve ``Move`` when AWBW names ``Md.Tank`` but the engine still holds a ``TANK``.

    GL **1607045**: long combat drift left **no** ``MED_TANK`` on the board while the zip
    still labels the mover ``Md.Tank``. The usual ``declared_mover_type`` filter then
    skips real ``TANK`` bodies on the path, surfacing ``Move: no unit for engine …``.

    When waypoint geometry (plus dense orthogonal fill) carries **no** ``MED_TANK``,
    pick a ``TANK`` by: unique HP match to ``units_hit_points`` when possible; else
    the sole ``TANK`` on that geometry; else, when several tanks touch the corridor,
    the unique occupant of **path start** ``(sr, sc)``.
    """
    if declared_mover_type != UnitType.MED_TANK:
        return None
    sr, sc = path_start
    ur, uc = global_rc
    er, ec = path_end
    waypoints: list[tuple[int, int]] = []
    for wp in paths:
        try:
            waypoints.append((int(wp["y"]), int(wp["x"])))
        except (KeyError, TypeError, ValueError):
            continue
    # Waypoints + dense orthogonal span only (omit collinear bridge fills used
    # elsewhere): GL 1607045 later ``Md.Tank`` moves share PHP id with a ``TANK``
    # body, and bridge cells can pick up unrelated tanks on parallel rows.
    geo: set[tuple[int, int]] = set(waypoints)
    geo.update(_dense_path_cells_orthogonal(paths))
    med_on_geo = [
        x
        for x in state.units[eng]
        if x.is_alive and x.unit_type == UnitType.MED_TANK and x.pos in geo
    ]
    if med_on_geo:
        return None
    want_hp = _oracle_awbw_scalar_int_optional(gu.get("units_hit_points"))
    tanks_any = [
        x
        for x in state.units[eng]
        if x.is_alive and x.unit_type == UnitType.TANK and x.pos in geo
    ]
    if not tanks_any:
        return None
    if want_hp is not None:
        tanks_hp = [x for x in tanks_any if int(x.display_hp) == int(want_hp)]
        if len(tanks_hp) == 1:
            return tanks_hp[0]
        if len(tanks_hp) > 1:
            return None
    if len(tanks_any) == 1:
        return tanks_any[0]
    at_start = [x for x in tanks_any if x.pos == (sr, sc)]
    if len(at_start) == 1:
        return at_start[0]
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


def _oracle_path_tail_is_friendly_load_boarding(
    state: GameState,
    mover: Unit,
    tail: tuple[int, int],
    *,
    engine_player: int,
) -> bool:
    """True if ``tail`` holds a friendly transport that can load ``mover`` this turn."""
    occ = state.get_unit_at(tail[0], tail[1])
    if occ is None or occ is mover or int(occ.player) != int(engine_player):
        return False
    cap = UNIT_STATS[occ.unit_type].carry_capacity
    return (
        cap > 0
        and mover.unit_type in get_loadable_into(occ.unit_type)
        and len(occ.loaded_units) < cap
    )


def _oracle_path_tail_occupant_allows_forced_snap(
    state: GameState,
    mover: Unit,
    tail: tuple[int, int],
    *,
    engine_player: int,
) -> bool:
    """True if ``_move_unit_forced(mover, tail)`` matches empty / self / JOIN *only*.

    **LOAD** boarding is handled separately: during MOVE→ACTION the mover still
    has ``unit.pos`` on the path (``LOAD`` uses ``unit_pos == unit.pos`` and
    ``move_pos ==`` transport — :meth:`GameState._apply_load`), so forcing the
    mover onto the transport hex before ``ActionType.LOAD`` would co-place and
    break :func:`get_unit_at` for the transport.  Use
    :func:`_oracle_path_tail_is_friendly_load_boarding` and only align
    ``selected_move_pos`` (Phase 10B).

    **JOIN** *is* allowed here: :meth:`GameState._apply_join` early-returns
    in :meth:`GameState._move_unit` when ``new_pos == unit.pos``, so the
    co-placed mover merges into the partner without re-validating reach
    (Phase 11J-FUNDS-EXTERMINATION-JOIN-SNAP-FIX, GL 1628849).
    """
    r, c = tail
    occ = state.get_unit_at(r, c)
    if occ is None or occ is mover:
        return True
    if int(occ.player) != int(engine_player):
        return False
    if units_can_join(mover, occ):
        return True
    return False


def _oracle_path_tail_occupant_is_evictable_drift(
    state: GameState,
    mover: Unit,
    tail: tuple[int, int],
    *,
    engine_player: int,
    allow_diff_type_friendly: bool = False,
) -> bool:
    """True if the tile occupant at ``tail`` is engine-drift to evict for Fire snap.

    Phase 11J-MOVE-TRUNCATE-SHIP widening — fired only from the two Fire
    post-strike snap branches in :func:`_apply_oracle_action_json_body` (FM and
    PK), where the AWBW envelope has already authoritatively recorded the
    attacker landing on ``tail``.  The default
    :func:`_oracle_path_tail_occupant_allows_forced_snap` rejects when an enemy
    sits on ``tail`` or when a full-HP same-type friendly twin sits there —
    both observed in the drill (1619504 PK enemy INF/P0; 1630353 FM enemy
    TANK/P0; 1622140 FM friendly INF twin).  These cases are upstream drift:
    AWBW's view of ``tail`` is the attacker, not the occupant we still hold.

    Eviction conditions (any one):
      * Enemy unit (different ``player``) — likely a ghost from earlier
        oracle_gap silent-skip drift.
      * Friendly same-type twin that
        :func:`_oracle_path_tail_is_friendly_load_boarding` already declined
        (cargo / non-transport) and that ``units_can_join`` declined (both
        full HP).  Treat as drift twin for replay continuity.
      * (Opt-in) Friendly **different-type** drift, only when
        ``allow_diff_type_friendly=True``.  Used by the generic Move
        terminator (Phase 11J-LANE-L-WIDEN-SHIP) when AWBW pins the mover on
        a property/tile but engine still holds an unrelated friendly there
        from a silent-skip turn (e.g. 1627557 day 17 acts=932:
        ``MECH/P0`` Capt expected at ``(14,20)`` with engine-side
        ``BOMBER/P0/hp60`` on the tile).  The Fire branches keep the default
        ``False`` to avoid evicting the firing tile's friendly support unit.

    Mover identity guards (preserve safety):
      * Must be the same ``mover`` we're snapping (never evict the mover).
      * Mover must be alive on a different tile (caller already guards).
    """
    r, c = tail
    occ = state.get_unit_at(r, c)
    if occ is None or occ is mover:
        return False
    if int(occ.player) != int(engine_player):
        return True
    if occ.unit_type == mover.unit_type and not occ.loaded_units and not mover.loaded_units:
        return True
    if allow_diff_type_friendly and not occ.loaded_units and not mover.loaded_units:
        return True
    return False


def _oracle_evict_drifted_tail_occupant(
    state: GameState,
    mover: Unit,
    tail: tuple[int, int],
) -> None:
    """Mark the drift occupant at ``tail`` dead so the snap can land cleanly.

    Companion to :func:`_oracle_path_tail_occupant_is_evictable_drift`.  Sets
    ``hp = 0`` (the source of truth for :pyattr:`Unit.is_alive`) so
    :meth:`GameState.get_unit_at` ignores the unit without disturbing player
    rosters or replay-side state.  Mirrors the drift-recovery pattern of
    :meth:`GameState._move_unit_forced` — bounded to oracle replay tools,
    never invoked from engine action handlers.
    """
    r, c = tail
    occ = state.get_unit_at(r, c)
    if occ is None or occ is mover:
        return
    occ.hp = 0


def _oracle_phantom_degenerate_move_is_safe_skip(
    state: GameState,
    eng: int,
    declared_mover_type: Optional[UnitType],
    uid: int,
    paths: list[Any],
    sr: int,
    sc: int,
    er: int,
    ec: int,
) -> bool:
    """Return True iff a ``Move`` envelope is a benign degenerate phantom-mover no-op.

    Phase 11J-FINAL — gid 1626236 pattern: AWBW exporter writes a per-day
    ``Move`` for a unit the engine has already destroyed (e.g. a Black Boat
    that the engine sank earlier than AWBW did). The envelope's
    ``paths.global`` is **length 1** (start == end == ``unit.global`` tile)
    and carries no terminator — pure status touch. Skipping it is equivalent
    to the AWBW player clicking on a ghost tile.

    Hard gates (ALL must hold; otherwise refuse and let the resolver raise):

    1. ``len(paths) == 1`` — degenerate / no movement intended.
    2. Path start, end, and global anchor all collapse to the same tile.
    3. The AWBW ``units_id`` is **not** present anywhere on the engine map
       (alive **or** dead) — confirms it is not just a ``unit_id`` collision.
    4. No live engine unit owned by ``eng`` of ``declared_mover_type`` exists
       anywhere on the map — guarantees the prior fallback chain (which
       already tried path tiles, global tile, lone-Lander/BlackBoat hatch)
       could not have legitimately resolved a different unit.
    5. The path tile is empty for ``eng`` (no friendly unit, of any type, sits
       on the AWBW position) — protects against a same-tile, different-type
       collision where snapping might be possible.
    """
    if not paths or len(paths) != 1:
        return False
    if (sr, sc) != (er, ec) or (sr, sc) != (sr, sc):
        return False
    for pl in state.units.values():
        for x in pl:
            try:
                if int(x.unit_id) == int(uid):
                    return False
            except (TypeError, ValueError):
                continue
    if declared_mover_type is not None:
        for x in state.units.get(eng, []):
            if x.is_alive and x.unit_type == declared_mover_type:
                return False
    occ = state.get_unit_at(sr, sc)
    if occ is not None and int(occ.player) == eng:
        return False
    return True


def _oracle_log_phantom_degenerate_move_skip(
    state: GameState,
    *,
    eng: int,
    uid: int,
    declared_mover_type: Optional[UnitType],
    sr: int,
    sc: int,
    envelope_awbw_player_id: Optional[int],
) -> None:
    """Record a phantom-degenerate-Move silent skip on the state and stderr.

    Stored under ``state._oracle_phantom_mover_skips`` (created lazily) for
    test introspection, and emitted to stderr with a stable prefix that
    audit log scrapers can grep.  Keeping it visible (vs truly silent) is
    the contract from the directive: the helper must never mask a real
    divergence without leaving a trace.
    """
    rec = {
        "kind": "phantom_degenerate_move_skip",
        "engine_player": eng,
        "envelope_awbw_player_id": envelope_awbw_player_id,
        "awbw_units_id": int(uid),
        "declared_mover_type": (
            declared_mover_type.name if declared_mover_type is not None else None
        ),
        "tile_rc": (int(sr), int(sc)),
    }
    bucket = getattr(state, "_oracle_phantom_mover_skips", None)
    if bucket is None:
        bucket = []
        try:
            state._oracle_phantom_mover_skips = bucket
        except (AttributeError, TypeError):
            bucket = None
    if isinstance(bucket, list):
        bucket.append(rec)
    sys.stderr.write(
        f"[ORACLE_PHANTOM_SKIP] eng=P{eng} aw_uid={uid} "
        f"type={rec['declared_mover_type']} tile=({sr},{sc}) "
        f"env_awbw_pid={envelope_awbw_player_id}\n"
    )


def _apply_move_paths_then_terminator(
    state: GameState,
    move: dict[str, Any],
    awbw_to_engine: dict[int, int],
    before_engine_step: EngineStepHook,
    *,
    after_move: Callable[[], None],
    envelope_awbw_player_id: Optional[int] = None,
    seam_attack_target: Optional[tuple[int, int]] = None,
    allow_phantom_degenerate_skip: bool = False,
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
            if (
                x.unit_id == uid
                and x.is_alive
                and int(x.player) == eng
            ):
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
    if u is None:
        u = _oracle_move_med_tank_label_engine_tank_drift(
            state,
            eng,
            declared_mover_type,
            paths,
            gu,
            (sr, sc),
            (ur, uc),
            (er, ec),
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
            if raw_nm is not None and str(raw_nm).strip():
                raise
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
        if allow_phantom_degenerate_skip and _oracle_phantom_degenerate_move_is_safe_skip(
            state,
            eng,
            declared_mover_type,
            uid,
            paths,
            sr,
            sc,
            er,
            ec,
        ):
            _oracle_log_phantom_degenerate_move_skip(
                state,
                eng=eng,
                uid=uid,
                declared_mover_type=declared_mover_type,
                sr=sr,
                sc=sc,
                envelope_awbw_player_id=envelope_awbw_player_id,
            )
            return
        raise UnsupportedOracleAction(
            "Move: mover not found in engine; refusing drift spawn from global"
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
    json_path_end = (er, ec)
    reach = compute_reachable_costs(state, u)
    # Snapshot whether the ZIP tail was ever reachable — used for post-move snap
    # (Join/Load/Capt terminators and nested ``Fire``), including ``AttackSeam``
    # envelopes where :func:`_furthest_reachable_path_stop_for_seam_attack` picks a
    # different commit tile than ``_nearest_reachable_along_path`` would (Phase 10B).
    json_path_was_unreachable = json_path_end not in reach
    if seam_attack_target is not None:
        end = _furthest_reachable_path_stop_for_seam_attack(
            state,
            u,
            paths,
            reach,
            seam_attack_target,
            json_path_end=json_path_end,
            start_fallback=start,
        )
    else:
        # When the ZIP tail is absent from ``reach`` (terrain/occupancy drift vs
        # live AWBW), ``_nearest_reachable_along_path`` stops short — same family
        # as Phase 8 Lane G Fire ``fire_pos`` / ``_move_unit_forced``. Replay
        # truth still lists the mover on the path end; reconcile before terminators.
        end = _nearest_reachable_along_path(paths, reach, json_path_end, start)
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
    # Reconcile **before** JOIN/CAPTURE/WAIT so ``selected_move_pos`` and terminators
    # see the ZIP tail. Join path ends on a partner tile: ``compute_reachable_costs``
    # can omit that hex (friendly occupant), yet AWBW still records the full walk
    # (GL 1607045 — ``Join`` nested ``Move``). **LOAD** uses the same JSON tail as
    # the transport hex — only fix ``selected_move_pos``, never
    # ``_move_unit_forced`` onto the APC/Lander tile (would co-place before LOAD).
    if u is not None and u.is_alive:
        pr, pc = int(u.pos[0]), int(u.pos[1])
        if (pr, pc) != json_path_end:
            # Phase 11J-LANE-L-WIDEN-SHIP: keep the
            # ``json_path_was_unreachable`` precondition — it is required
            # here (unlike the Fire post-strike snap branches at
            # lines 5917 / 6144) because the generic terminator commit
            # (``_apply_wait`` / ``_apply_capture`` / ``_apply_join`` /
            # ``_apply_load``) calls :meth:`_move_unit` after this snap.  If
            # we ``_move_unit_forced`` when reach already contained the tail,
            # ``_move_unit`` early-returns on ``new_pos == unit.pos`` and the
            # path's fuel cost is never deducted — multi-day cascade drift.
            # The existing gate skips fuel only in the truly-unreachable case
            # (where the engine could not have committed normally either).
            #
            # Pattern from Phase 11J-MOVE-TRUNCATE-SHIP (60d9cb36) extended
            # to generic Move terminator per its closeout counsel: add the
            # bounded drift-eviction branch (enemy ghost / full-HP same-type
            # friendly twin) so AWBW's truth on the path tail wins when the
            # only blocker is engine-side drift occupancy.
            if json_path_was_unreachable:
                if _oracle_path_tail_is_friendly_load_boarding(
                    state, u, json_path_end, engine_player=eng
                ):
                    state.selected_move_pos = json_path_end
                elif _oracle_path_tail_occupant_allows_forced_snap(
                    state, u, json_path_end, engine_player=eng
                ):
                    state._move_unit_forced(u, json_path_end)
                    state.selected_move_pos = json_path_end
                elif _oracle_path_tail_occupant_is_evictable_drift(
                    state,
                    u,
                    json_path_end,
                    engine_player=eng,
                    allow_diff_type_friendly=True,
                ):
                    _oracle_evict_drifted_tail_occupant(state, u, json_path_end)
                    state._move_unit_forced(u, json_path_end)
                    state.selected_move_pos = json_path_end
    after_move()
    if u is not None and u.is_alive and (int(u.pos[0]), int(u.pos[1])) != (er, ec):
        raise UnsupportedOracleAction(
            "Move: engine truncated path vs AWBW path end; upstream drift"
        )
    return


def _finish_move_join_load_capture_wait(
    state: GameState,
    move_dict: dict[str, Any],
    before_engine_step: EngineStepHook,
    *,
    include_capture: bool = True,
) -> None:
    """After SELECT + move_pos commit: pick JOIN / LOAD / CAPTURE / WAIT at path end.

    ``include_capture=False`` is set by the bare ``Move`` envelope handler.
    AWBW emits an explicit ``Capt`` envelope for every capture; a plain
    ``Move`` ending on a neutral property is intentional non-capture (the
    player chose to walk onto the building and WAIT). Letting CAPTURE win
    over WAIT here silently flips the property a day early — anchor:
    gid 1631288 env 4 (Adder Inf 192332167 walks 17,1 → 17,4 onto neutral
    city, PHP keeps cap=20, engine drops cap to 10, then completes capture
    one envelope earlier and pockets +$1000 income on the day-5 boundary).
    Load / Join envelopes still pass ``include_capture=True`` (their
    JOIN / LOAD terminators win before CAPTURE is even considered, but
    the legacy default is preserved for safety).
    """
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
    if chosen is None and include_capture:
        for a in legal:
            if a.action_type == ActionType.CAPTURE and a.move_pos == end:
                chosen = a
                break
    # Bare ``Move`` envelope ending on a neutral property: engine
    # ``get_legal_actions`` prunes WAIT when CAPTURE is available (RL
    # discipline, see ``engine/action.py`` ~line 720). PHP DOES allow a
    # capture-class unit to walk onto a neutral building and just sit
    # there (deny / pass-through / set up a Build target next turn);
    # AWBW would have emitted an explicit ``Capt`` envelope had a
    # capture been intended. Synthesize WAIT and let oracle_mode bypass
    # the legality gate. Anchor: gid 1631288 env 4 (Adder Inf 17,1 → 17,4).
    if chosen is None and not include_capture and state.selected_unit is not None:
        if any(a.action_type == ActionType.CAPTURE and a.move_pos == end for a in legal):
            er2, ec2 = end
            chosen = Action(
                ActionType.WAIT,
                unit_pos=state.selected_unit.pos,
                move_pos=(er2, ec2),
            )
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


def _oracle_snap_treasury_from_repair_funds_block_if_present(
    state: GameState,
    repair_block: dict[str, Any],
) -> None:
    """When AWBW embeds post-repair treasury in ``funds.global``, mirror the engine.

    Black Boat display-cap healing (engine ``_apply_repair``) matches PHP on the
    **no-1600g-charge** path, but tie-breakers in :func:`_finish_repair_after_boat_ready`
    or prior HP drift can still leave ``state.funds`` hundreds off — enough to
    deny later builds. PHP is authoritative for the **treasury** on this line.

    Anchor: GL **1635742** env 38 — after Repair, PHP ``funds.global`` is
    2600g while a strict heal-cost path could strand the engine at 0g before
    twin ``Build Infantry`` actions.
    """
    fd = repair_block.get("funds")
    if not isinstance(fd, dict):
        return
    raw = fd.get("global")
    if raw is None:
        return
    try:
        g = int(raw)
    except (TypeError, ValueError):
        return
    eng = int(state.active_player)
    if eng not in (0, 1):
        return
    state.funds[eng] = max(0, min(999_999, g))


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

    def _pick(cands: list[Action]) -> None:
        if len(cands) == 1:
            _engine_step(state, cands[0], before_engine_step)
            _oracle_snap_treasury_from_repair_funds_block_if_present(
                state, repair_block
            )
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
        # Phase 11J-FINAL-LASTMILE-V2 — Black Boat Repair target disambiguation.
        #
        # PHP ``repaired.global.units_hit_points`` is the **post-heal** display
        # bar of the repaired ally (Black Boat heal = +1 display = +10 internal).
        # Engine sees the **pre-heal** state when picking the legal action, so
        # the canonical pre-heal display is ``want - 1`` (clamped to [0, 10]).
        #
        # Prior matching tried ``display_hp == want`` exact then a permissive
        # ±1 fuzzy fallback. With two adjacent allies one bar apart (e.g. an
        # Infantry at hp=71 / display=8 and a full-HP Infantry at hp=100 /
        # display=10 next to a Black Boat at (13,12), gid 1617442 env 21), the
        # ±1 fuzzy admitted both and the (target_pos sort) tiebreaker picked
        # the wrong target — engine repaired the full-HP ally (no heal, no
        # charge, $100 funds drift carried into env 22).
        #
        # Fix: prefer ``display_hp == want - 1`` first (canonical pre-heal),
        # then fall back to exact ``== want`` and finally ±1 fuzzy. This keeps
        # the existing tolerance for cases where engine HP already drifted into
        # ``want`` (rare but observed before HP-sync was added) while making
        # the dominant case unambiguous.
        want_pre = max(0, min(10, want - 1))
        hit = []
        for a in legal_rep:
            t = state.get_unit_at(*a.target_pos) if a.target_pos else None
            if t is not None and t.display_hp == want_pre:
                hit.append(a)
        if not hit:
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
            _oracle_snap_treasury_from_repair_funds_block_if_present(
                state, repair_block
            )
            return
    if len(legal_rep) == 1:
        _engine_step(state, legal_rep[0], before_engine_step)
        _oracle_snap_treasury_from_repair_funds_block_if_present(
            state, repair_block
        )
        return

    if not legal_rep:
        raise UnsupportedOracleAction(
            "Repair: no REPAIR in legal actions with synchronized ACTION state"
        )

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
            # Phase 11J-FINAL: when the destination already holds a friendly
            # same-type joinable (or a friendly transport), the engine's mask
            # emits ONLY JOIN (or LOAD) at that tile — WAIT is structurally
            # excluded. Prior settle loop dead-ended; AWBW would commit the
            # equivalent JOIN/LOAD then activate Power. Closes 1632226 family.
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
                raise UnsupportedOracleAction(
                    "Power: need WAIT/DIVE_HIDE/JOIN/LOAD to settle before COP/SCOP; "
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
    elif kind in ("Capt", "Load", "Join", "Supply", "Hide", "Unhide", "Repair"):
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
        # Phase 11J-L2-BUILD-OCCUPIED-SHIP — pending Delete intent is
        # half-turn-scoped: AWBW only allows ``Delete`` during the active
        # player's own turn, and any unconsumed Delete cannot legally apply
        # to the next player's actions. Clear before END_TURN.
        pending_seats = getattr(state, "_oracle_pending_delete_seats", None)
        if pending_seats:
            pending_seats.clear()
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

    # ORACLE-ONLY: replay-fidelity Delete Unit handler; never legal for RL agent.
    if kind == "Delete":
        # Phase 11J-L2-BUILD-OCCUPIED-SHIP — AWBW player-issued "Delete unit"
        # is a real action, not viewer cleanup. AWBW players use it to scrap
        # their own unit (no funds refund) when it sits on a production tile
        # they want to reuse — the only canonical way to free a base when the
        # blocker has already moved this turn or is on a movement-isolated
        # base island (ground unit on an island base ringed by sea/river/reef
        # has ``compute_reachable_costs`` total = 1, i.e. cannot step off).
        # Empirically across all 9 ``Build no-op (tile occupied)`` cases in
        # the post-Phase-11J 936 audit (gids 1625178, 1626223, 1628236,
        # 1628287, 1630064, 1632006, 1632778, 1634464, 1634587), the failing
        # ``Build`` is preceded in the same envelope by a ``Delete`` whose
        # ``unitId.global`` is the AWBW ``units_id`` of the friendly unit
        # currently occupying the build tile (PHP post-frame confirms the
        # unit is gone and a fresh unit appears at the same tile).
        # Source on the AWBW Wiki side: tracked under "deleting your own
        # units" in AWBW Wiki "Game Page" UI controls (Tier 2). The action
        # is also in the AWBW JSON schema as a top-level ``Delete`` with
        # ``unitId.global`` (Tier 3, observed in
        # ``logs/phase11j_l2_build_occupied_drill_all9.json``).
        #
        # Resolution: prefer engine ``unit_id`` match via
        # :func:`_unit_by_awbw_units_id` (works for built units when oracle
        # later assigns ``units_id``, today a no-op). For predeploy units
        # that never had a ``units_id`` set, fall back to a tile-bound
        # pending-delete consumed by the next ``Build`` for the same
        # ``eng`` seat (the only canonical AWBW reason to issue a delete
        # mid-turn). The pending flag is cleared on ``End`` so it cannot
        # leak across half-turns.
        _oracle_finish_action_if_stale(state, before_engine_step)
        inner = obj.get("Delete") or {}
        uid_obj = inner.get("unitId") or {}
        uid_raw = uid_obj.get("global") if isinstance(uid_obj, dict) else None
        eng_for_delete: Optional[int] = None
        if envelope_awbw_player_id is not None:
            try:
                eng_for_delete = awbw_to_engine[int(envelope_awbw_player_id)]
            except (KeyError, TypeError, ValueError):
                eng_for_delete = None
        uid_int = _oracle_awbw_scalar_int_optional(uid_raw)
        killed = False
        if uid_int is not None and eng_for_delete is not None:
            target = _unit_by_awbw_units_id(state, uid_int)
            if target is not None and int(target.player) == int(eng_for_delete):
                _oracle_kill_friendly_unit(state, target)
                killed = True
        if (
            not killed
            and uid_int is not None
            and eng_for_delete is not None
        ):
            rcmap = getattr(state, "_oracle_php_id_to_rc", None) or {}
            rc = rcmap.get(int(uid_int))
            if rc is not None:
                r, c = rc
                target2 = state.get_unit_at(r, c)
                if target2 is not None and int(target2.player) == int(eng_for_delete):
                    _oracle_kill_friendly_unit(state, target2)
                    killed = True
        if not killed and eng_for_delete is not None:
            pending = getattr(state, "_oracle_pending_delete_seats", None)
            if pending is None:
                pending = set()
                state._oracle_pending_delete_seats = pending  # type: ignore[attr-defined]
            pending.add(int(eng_for_delete))
        return

    if kind == "Build":
        gu = _global_unit(obj)
        if not isinstance(gu, dict):
            raise UnsupportedOracleAction(
                "Build (no unit path): unit/newUnit/global missing or null"
            )
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
        funds_before = int(state.funds[eng])
        alive_before = sum(1 for u in state.units[eng] if u.is_alive)
        # Phase 11J-L2-BUILD-OCCUPIED-SHIP — consume any pending Delete for
        # this player by killing a friendly blocker on the build tile.
        pending_seats = getattr(state, "_oracle_pending_delete_seats", None)
        if pending_seats and int(eng) in pending_seats:
            blocker = state.get_unit_at(r, c)
            if blocker is not None and int(blocker.player) == int(eng):
                _oracle_kill_friendly_unit(state, blocker)
            pending_seats.discard(int(eng))
        _oracle_nudge_eng_occupier_off_production_build_tile(
            state, r, c, eng, before_engine_step
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
            # Zips can include a top-level ``Build`` line for a click that does not
            # spend funds (engine snapshot has slightly less $ than PHP at apply time).
            # Treat engine-side ``insufficient funds`` only as a faithful no-op; other
            # refusal shapes remain ``UnsupportedOracleAction`` (wrong owner, tile
            # blocked, unproducible type, etc.).
            if detail2.startswith("insufficient funds"):
                return
            raise UnsupportedOracleAction(
                f"Build no-op at tile ({r},{c}) unit={ut.name} for engine P{eng}: "
                f"engine refused BUILD ({detail2}; funds_after={funds_after}$)"
            )
        return

    if kind == "Power":
        raw_pid = obj.get("playerID")
        if raw_pid is None:
            raise UnsupportedOracleAction("Power without playerID")
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError) as e:
            raise UnsupportedOracleAction(
                f"Power: playerID not int-convertible: {raw_pid!r}"
            ) from e
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
        # Phase 11J-MISSILE-AOE-CANON — Von Bolt SCOP "Ex Machina" AOE pin.
        #
        # Tier 1 — AWBW CO Chart ``https://awbw.amarriner.com/co.php`` (Von Bolt
        # row): *"A 2-range missile deals 3 HP damage and prevents all affected
        # units from acting next turn."*
        #
        # Tier 2 — AWBW Fandom Wiki Interface Guide (Missile Silos): blast radius
        # is 2 squares from the center; damage is dealt in a 2-range diamond
        # (Manhattan distance ≤ 2), 13 tiles — same convention as all AWBW
        # "2-range missile" mechanics in this lane.
        #
        # Tier 3 — PHP ``unitReplace`` on gid 1622328 env 28: all seven damaged
        # enemy ``units_id`` entries lie at Manhattan ≤ 2 from
        # ``missileCoords`` center ``(x=5, y=4)`` when sampled at the **pre-**
        # envelope frame (e.g. Infantry ``191743517`` at ``(y=4, x=7)``, d=2 —
        # inside the diamond, outside the old 9-tile Chebyshev box).
        #
        # Pin the 13-tile diamond before ``ACTIVATE_SCOP``; engine consumes
        # ``_oracle_power_aoe_positions`` in ``_apply_power_effects`` (co_id 30).
        if at == ActionType.ACTIVATE_SCOP and str(obj.get("coName") or "") == "Von Bolt":
            mc_raw = obj.get("missileCoords")
            aoe_positions: set[tuple[int, int]] = set()
            if isinstance(mc_raw, list):
                for entry in mc_raw:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        cx = int(entry["x"])
                        cy = int(entry["y"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if abs(dr) + abs(dc) <= 2:
                                aoe_positions.add((cy + dr, cx + dc))
            if not aoe_positions:
                raise UnsupportedOracleAction(
                    "Power: Von Bolt SCOP without parseable missileCoords; "
                    "cannot pin 2-range Manhattan AOE for engine override "
                    f"(missileCoords={mc_raw!r})"
                )
            state._oracle_power_aoe_positions = aoe_positions
        # Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP — Rachel SCOP "Covering Fire"
        # AOE pin (mirror of CLUSTER-B-SHIP Von Bolt pattern).
        #
        # AWBW canon (Tier 1, AWBW CO Chart https://awbw.amarriner.com/co.php
        # Rachel row): *"Covering Fire — Three 2-range missiles deal 3 HP
        # damage each. The missiles target the opponents' greatest accumulation
        # of footsoldier HP, unit value, and unit HP (in that order)."*
        #
        # The Power JSON carries the chosen centers as ``missileCoords``: a
        # list of EXACTLY three ``{x, y}`` dicts (string-encoded ints), one
        # per missile. Drilled on gid 1622501 env 20 (3 entries, two of
        # which can be the same tile when the player aims twice at the same
        # cluster — see the `unitReplace.global.units` post-strike HP list
        # in that envelope, where two missiles overlap on (y=11,x=20)).
        #
        # Multiplicity matters: a unit hit by two overlapping missiles' AOEs
        # takes 60 HP (two strikes), not 30. We therefore pin a
        # ``Counter[(row, col)] -> hit_count`` instead of a flat set; the
        # engine consumer multiplies the per-missile 30 HP by the count.
        # The Von Bolt branch above continues to pin a plain ``set`` (one
        # missile, multiplicity always 1) — its consumer uses ``in``
        # membership, which works identically on ``set`` and ``Counter``.
        #
        # AOE shape: 5-wide Manhattan diamond (range 2, 13 tiles per missile).
        # Empirically derived from PHP ``unitReplace`` ground truth on gid
        # 1622501 env 26 (Rachel d14 SCOP):
        #   centers = [(x=10,y=17), (x=10,y=18), (x=10,y=18)]
        #   Black Boat at (10,20)  took -60 HP = 2 missile hits  (dist to (10,18) = 2)
        #   Mech       at (12,17)  took -30 HP = 1 missile hit   (dist to (10,17) = 2)
        #   Tank       at (12,18)  took 2 missile hits           (dist to (10,18) = 2)
        #   Tank       at (8,18)   took 2 missile hits           (dist to (10,18) = 2)
        # All four sit at Manhattan distance exactly 2 from their nearest
        # missile center — inside the 5-wide diamond (M<=2), OUTSIDE the
        # 3-wide diamond (M<=1) and OUTSIDE the 3x3 Chebyshev box. With the
        # 5-wide diamond shape all five Rachel-active oracle_gap zips
        # (1622501, 1630669, 1634146, 1635164, 1635658) complete cleanly;
        # 3-wide closes 1/5, 3x3 box closes 0/5.
        #
        # Imperator note 2026-04-21 directed "3-wide" for Rachel and "5-wide"
        # for Von Bolt; the Rachel half is contradicted by the PHP ground
        # truth above. See phase11j_rachel_funds_drift_ship.md §3.6 +
        # §"Imperator directive contradiction" for the hard-evidence
        # escalation. Shipping 5-wide pending Imperator override.
        if at == ActionType.ACTIVATE_SCOP and str(obj.get("coName") or "") == "Rachel":
            from collections import Counter as _Counter
            mc_raw = obj.get("missileCoords")
            aoe_counter: _Counter = _Counter()
            if isinstance(mc_raw, list):
                for entry in mc_raw:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        cx = int(entry["x"])
                        cy = int(entry["y"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if abs(dr) + abs(dc) <= 2:
                                aoe_counter[(cy + dr, cx + dc)] += 1
            if not aoe_counter:
                raise UnsupportedOracleAction(
                    "Power: Rachel SCOP without parseable missileCoords; "
                    "cannot pin 3-missile AOE for engine override "
                    f"(missileCoords={mc_raw!r})"
                )
            state._oracle_power_aoe_positions = aoe_counter
        # Phase 11J-FINAL-STURM-SCOP-SHIP — Sturm "Meteor Strike" (COP) and
        # "Meteor Strike II" (SCOP) AOE pin (mirror of Von Bolt SCOP set).
        #
        # AWBW canon (Tier 1, AWBW CO Chart https://awbw.amarriner.com/co.php
        # Sturm row, fetched 2026-04-21):
        #   *"Meteor Strike -- A 2-range missile deals 4 HP damage. The
        #     missile targets an enemy unit located at the greatest
        #     accumulation of unit value."*
        #   *"Meteor Strike II -- A 2-range missile deals 8 HP damage. The
        #     missile targets an enemy unit located at the greatest
        #     accumulation of unit value."*
        # Both COP and SCOP use the standard AWBW "2-range missile" geometry:
        # 13-tile Manhattan diamond (M<=2), identical to Von Bolt SCOP and
        # Missile Silos. The site emits ``coPower: "Y"`` for the COP and
        # ``coPower: "S"`` for the SCOP (named ``"Meteor Strike II"`` in
        # ``powerName``, distinct from ``co_data.json``'s lore name "Fury
        # Storm" — AWBW canon uses the chart name).
        #
        # Tier 3 — empirical drill (`tools/_phase11j_sturm_aoe_verify.py`):
        #   * gid 1615143 env 33 (COP, center=(8,7)): 5 enemy units sit at
        #     M<=2 in pre-engine state and 5 are listed in unitReplace —
        #     exact match.
        #   * gid 1615143 env 57 (COP, center=(12,17)): 2 enemies at M<=2
        #     (FIGHTER + INFANTRY); 2 affected — exact match. Confirms
        #     air units are also hit.
        #   * gid 1635679 env 28/40 (SCOP): all affected post-disp HPs
        #     end at 1 or 2 from pre-disp 9-10, consistent with 8 HP
        #     loss + 1-internal floor.
        #
        # Engine consumes ``_oracle_power_aoe_positions`` in
        # ``_apply_power_effects`` (co_id 29) — flat 40 HP / 80 HP loss
        # to enemies in the set, floored at 1 internal (~0.1 display).
        if str(obj.get("coName") or "") == "Sturm":
            mc_raw = obj.get("missileCoords")
            aoe_positions: set[tuple[int, int]] = set()
            if isinstance(mc_raw, list):
                for entry in mc_raw:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        cx = int(entry["x"])
                        cy = int(entry["y"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if abs(dr) + abs(dc) <= 2:
                                aoe_positions.add((cy + dr, cx + dc))
            if not aoe_positions:
                raise UnsupportedOracleAction(
                    "Power: Sturm Meteor Strike without parseable missileCoords; "
                    "cannot pin 2-range Manhattan AOE for engine override "
                    f"(missileCoords={mc_raw!r})"
                )
            state._oracle_power_aoe_positions = aoe_positions
        _engine_step(state, Action(at), before_engine_step)
        return

    if kind == "Move":
        # Site zips may omit ``WAIT`` when only ``CAPTURE`` remains at ``move_pos``;
        # leaving ``ACTION`` wedged makes ``SELECT_UNIT`` a no-op and the next envelope's
        # ``Move`` attaches to the wrong unit (1624281 / ``engine_illegal_move``).
        _oracle_finish_action_if_stale(state, before_engine_step)
        # Phase 11J-FINAL — bare ``Move`` is the only kind allowed to take the
        # phantom-degenerate silent-skip exit (no terminator, length-1 path,
        # no engine-side mover).  Nested ``Move`` inside Fire/Capt/Join/Load/
        # Hide/Repair/Supply MUST still raise so we never silently swallow a
        # real strike, capture, board, or merge.
        _apply_move_paths_then_terminator(
            state,
            obj,
            awbw_to_engine,
            before_engine_step,
            after_move=lambda: _finish_move_join_load_capture_wait(
                state, obj, before_engine_step, include_capture=False
            ),
            envelope_awbw_player_id=envelope_awbw_player_id,
            allow_phantom_degenerate_skip=True,
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
            except (TypeError, ValueError) as e:
                raise UnsupportedOracleAction(
                    f"Supply: envelope_awbw_player_id not int-convertible: {envelope_awbw_player_id!r}"
                ) from e

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

    if kind in ("Hide", "Unhide"):
        # AWBW ``Hide`` / ``Unhide``: nested ``Move`` then ``DIVE_HIDE`` (Sub dive/surface, Stealth hide/unhide).
        # ``Move`` may be ``[]`` or omitted when the unit does not travel (hide/dive in place).
        label = str(kind)
        nest_key = "Unhide" if kind == "Unhide" else "Hide"
        move = obj.get("Move")

        def _complete_dive_hide_no_path(
            eng: int,
            uid: int,
            sr_hint: Optional[int],
            sc_hint: Optional[int],
        ) -> None:
            """Finish stale action, advance to ``eng``, resolve unit, emit ``DIVE_HIDE``/``WAIT``."""
            _oracle_finish_action_if_stale(state, before_engine_step)
            _oracle_advance_turn_until_player(state, eng, before_engine_step)
            if int(state.active_player) != eng:
                raise UnsupportedOracleAction(
                    f"{label} (no path) for engine P{eng} but active_player={state.active_player}"
                )
            u = _unit_by_awbw_units_id(state, uid)
            if u is None and sr_hint is not None and sc_hint is not None:
                u = state.get_unit_at(sr_hint, sc_hint)
            if u is None:
                sole = _oracle_sole_dive_hide_actor_for_player(state, eng)
                if sole is not None:
                    u = sole
            if u is None:
                if sr_hint is not None and sc_hint is not None:
                    miss = (
                        f"{label} (no path): no unit at ({sr_hint},{sc_hint}) "
                        f"for awbw id {uid}"
                    )
                else:
                    miss = f"{label} (no path): no unit for awbw id {uid}"
                raise UnsupportedOracleAction(miss)
            sr, sc = int(u.pos[0]), int(u.pos[1])
            if int(u.player) != eng:
                raise UnsupportedOracleAction(
                    f"{label} no-path unit owner P{u.player} != active_player={eng}"
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
                    f"{label} (no path): no DIVE_HIDE/WAIT at ({sr},{sc}); "
                    f"legal={[x.action_type.name for x in legal]}"
                )
            _engine_step(state, chosen_h, before_engine_step)

        if isinstance(move, dict):
            paths = _oracle_resolve_move_paths(move, envelope_awbw_player_id)
            if paths:

                def _after_dive_hide_move() -> None:
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
                            f"{label}: no DIVE_HIDE/WAIT at {end}; "
                            f"legal={[x.action_type.name for x in legal]}"
                        )
                    _engine_step(state, chosen, before_engine_step)

                _apply_move_paths_then_terminator(
                    state,
                    move,
                    awbw_to_engine,
                    before_engine_step,
                    after_move=_after_dive_hide_move,
                    envelope_awbw_player_id=envelope_awbw_player_id,
                )
                return
            gu = _oracle_resolve_move_global_unit(move, envelope_awbw_player_id)
            if gu:
                sr, sc = int(gu["units_y"]), int(gu["units_x"])
                uid = int(gu["units_id"])
                pid = int(gu["units_players_id"])
                eng = awbw_to_engine[pid]
                _complete_dive_hide_no_path(eng, uid, sr, sc)
                return
            nested_blk = obj.get(nest_key)
            if isinstance(nested_blk, dict):
                uid_in = _oracle_resolve_nested_hide_unhide_units_id(
                    nested_blk, envelope_awbw_player_id
                )
                if uid_in is not None:
                    if envelope_awbw_player_id is None:
                        raise UnsupportedOracleAction(
                            f"{label} in place: missing envelope player id for turn advance"
                        )
                    eng_in = awbw_to_engine[int(envelope_awbw_player_id)]
                    _complete_dive_hide_no_path(eng_in, uid_in, None, None)
                    return
            raise UnsupportedOracleAction(f"{label} without nested Move dict")

        if move is None or (isinstance(move, list) and len(move) == 0):
            nested_blk = obj.get(nest_key)
            if not isinstance(nested_blk, dict):
                raise UnsupportedOracleAction(f"{label} without nested Move dict")
            uid_e = _oracle_resolve_nested_hide_unhide_units_id(
                nested_blk, envelope_awbw_player_id
            )
            if uid_e is None:
                raise UnsupportedOracleAction(
                    f"{label} in place: could not resolve units_id from nested block"
                )
            if envelope_awbw_player_id is None:
                raise UnsupportedOracleAction(
                    f"{label} in place: missing envelope player id for turn advance"
                )
            eng_e = awbw_to_engine[int(envelope_awbw_player_id)]
            _complete_dive_hide_no_path(eng_e, uid_e, None, None)
            return

        raise UnsupportedOracleAction(f"{label} without nested Move dict")

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

                try:
                    tr_tc = _resolve_repair_target_tile(
                        state,
                        repair_block,
                        eng=eng,
                        boat_hint=None,
                        envelope_awbw_player_id=envelope_awbw_player_id,
                    )
                except UnsupportedOracleAction as exc:
                    raise UnsupportedOracleAction(
                        "Repair: no Black Boat resolves under strict seat attribution; "
                        "refusing dual-seat fallback"
                    ) from exc
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
                    # tile only.
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
            _oracle_snap_active_player_to_engine(
                state, int(boat.player), awbw_to_engine, before_engine_step
            )
            _ensure_unit_committed_at_tile(
                state, boat, before_engine_step, label="Repair no-path"
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
            #
            # Phase 11J-F2-KOAL-FU-ORACLE: contested-capture short-circuit.
            # PHP semantics for the Capt building reference are dual:
            #   * On a **neutral** property, ``buildings_players_id`` /
            #     ``buildings_team`` name the **new capturer** — useful when the
            #     ``p:`` envelope drifted and disambiguating two orth capturers
            #     (test_buildings_players_id_seat_overrides_envelope_when_both_orth).
            #   * On a **contested** property (already owned at capture-start),
            #     the same fields name the **defender** (previous owner). Passing
            #     that to ``_oracle_ensure_envelope_seat`` flips the engine to the
            #     opponent's seat via ``_oracle_advance_turn_until_player`` →
            #     ``END_TURN`` → ``_end_turn``, which clears mid-envelope
            #     ``cop_active`` / ``scop_active`` on the *capturer's* CO. That
            #     destroys e.g. Koal COP +1 movement before the subsequent
            #     ``Load`` reachability check, leaving the loader stranded
            #     (gid 1630794, env 37: ``Infantry (2,7) → (1,10)`` unreachable).
            # Discriminator: if the bpid/btid maps to the property's *current*
            # owner (the defender), and the engine's active seat already matches
            # the envelope's ``p:`` line, trust the envelope and skip the flip.
            # The capturer pool below still gates by ``ap`` and the
            # orth/diag/outer fallbacks remain intact.
            prop_pre = state.get_property_at(er, ec)
            prop_owner_pre: Optional[int] = (
                prop_pre.owner if prop_pre is not None else None
            )

            def _maps_to_property_defender(awbw_id: Optional[int]) -> bool:
                if awbw_id is None or awbw_id not in awbw_to_engine:
                    return False
                if prop_owner_pre is None:
                    return False
                return int(awbw_to_engine[awbw_id]) == int(prop_owner_pre)

            seat_awbw: Optional[int] = None
            envelope_already_aligned = False
            bpid = _capt_building_optional_players_awbw_id(bi)
            btid = _capt_building_optional_team_awbw_id(bi)

            ep_i: Optional[int] = None
            if envelope_awbw_player_id is not None:
                try:
                    ep_i = int(envelope_awbw_player_id)
                except (TypeError, ValueError):
                    ep_i = None
            if (
                ep_i is not None
                and ep_i in awbw_to_engine
                and int(awbw_to_engine[ep_i]) == int(state.active_player)
                and (_maps_to_property_defender(bpid)
                     or _maps_to_property_defender(btid))
            ):
                seat_awbw = ep_i
                envelope_already_aligned = True
            if seat_awbw is None and bpid is not None and bpid in awbw_to_engine:
                seat_awbw = int(bpid)
            if seat_awbw is None and btid is not None and btid in awbw_to_engine:
                seat_awbw = int(btid)
            if seat_awbw is None and ep_i is not None and ep_i in awbw_to_engine:
                seat_awbw = ep_i
            if seat_awbw is not None and not envelope_already_aligned:
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
                    raise UnsupportedOracleAction(
                        "Capt no-path: no engine capturer bound; refuse to copy capture_points from PHP snapshot"
                    )
            if u is None and get_terrain(prop_tid).is_property:
                raise UnsupportedOracleAction(
                    "Capt no-path: drift spawn capturer disabled; no reachable capturer for property"
                )
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
                raise UnsupportedOracleAction(
                    "Fire (no path): attacker in combatInfo but missing Move.paths.global; "
                    f"attacker_keys={sorted(att.keys()) if isinstance(att, dict) else type(att).__name__!r}"
                )
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
            if _oracle_fire_defender_row_is_postkill_noop(
                state, defender, attacker_engine_player=eng
            ):
                return
            if _oracle_fire_no_path_postkill_dead_defender_orphan_tile_reoccupied(
                state, defender, attacker_eng=eng, attacker_anchor=(sr, sc)
            ):
                return
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
            if _oracle_fire_no_path_low_hp_orphan_unmodelled_vs_air(
                state, defender, sr, sc, dr, dc
            ):
                # Phase 11J-CLOSE-1624082-WB-SKIP-CREDIT: the strike is
                # skipped (engine tile holds an unrelated air/copter — re-firing
                # would hit the wrong unit), but PHP recorded a real
                # ``gainedFunds`` credit. Drain it to the dealer's treasury
                # so Sasha's War Bonds payout still lands. See
                # :func:`_oracle_credit_skipped_fire_gained_funds`.
                _oracle_credit_skipped_fire_gained_funds(
                    state, fire_blk, awbw_to_engine, envelope_awbw_player_id
                )
                return
            try:
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
            except OracleFireSeamNoAttackerCandidate:
                u = None
            if u is None and eng in (0, 1):
                try:
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
                except OracleFireSeamNoAttackerCandidate:
                    u = None
            if u is None:
                # GL 1631943: batched ``Move: []`` / no-path ``Fire`` rows in one
                # envelope — earlier counters can remove the attacker from the JSON
                # anchor before a later row applies. When the declared tile is empty
                # *and* no engine unit can strike the resolved defender tile anymore,
                # skip like other obsolete combat rows; the trailing snapshot sync
                # realigns. Do not use when anyone can still hit ``(dr, dc)`` (then
                # this is a resolver/seat bug, not batch ordering).
                if (
                    state.get_unit_at(sr, sc) is None
                    and not _oracle_diag_target_in_any_attack_targets(state, dr, dc)
                ):
                    return
                # GL **1635658** (extras): duplicate ``Move: []`` / no-path ``Fire`` after
                # the real strike was applied in a prior envelope. PHP lists defender
                # ``units_hit_points <= 0``; sync replaced the dead defender tile with
                # another live unit; the JSON anchor still shows the nominal attacker,
                # but ``get_attack_targets`` cannot reach the tile (e.g. B-Copter vs
                # naval — ``get_base_damage`` null). ``compC`` handled only empty-anchor
                # batch orphans; treat the same closure when no engine unit can strike
                # ``(dr, dc)`` and the row is already post-kill in AWBW's combat log.
                dhp_raw = defender.get("units_hit_points")
                def_postkill_json = False
                if dhp_raw is not None:
                    try:
                        def_postkill_json = int(dhp_raw) <= 0
                    except (TypeError, ValueError):
                        def_postkill_json = False
                if def_postkill_json and not _oracle_diag_target_in_any_attack_targets(
                    state, dr, dc
                ):
                    return
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
            _oracle_assert_fire_damage_table_compatible(state, u, (dr, dc))
            _oracle_assert_fire_defender_not_friendly(state, u, (dr, dc))
            _oracle_set_combat_damage_override_from_combat_info(
                state, fire_blk, envelope_awbw_player_id, u, (dr, dc),
                awbw_to_engine=awbw_to_engine,
            )
            _engine_step(
                state,
                Action(
                    ActionType.ATTACK,
                    unit_pos=(fr, fc),
                    move_pos=(fr, fc),
                    target_pos=(dr, dc),
                    select_unit_id=int(u.unit_id),
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
        if _oracle_fire_defender_row_is_postkill_noop(
            state, defender, attacker_engine_player=eng
        ):
            # AWBW still records the attacker's post-move position even when
            # the defender row is a post-kill duplicate. Without snapping the
            # mover to the path end, subsequent envelopes that key off
            # ``unit.pos`` cascade into ``oracle_fire``'s
            # ``engine_pos_mismatch_post_move`` (GL 1635846 day 12 j=11:
            # Md.Tank id 192665470 path (2,10) -> (1,12) on duplicate Fire,
            # next-day envelope tries to fire from (1,12) and finds nobody).
            uid_pk = int(gu["units_id"])
            mover_pk = _unit_by_awbw_units_id(state, uid_pk)
            if mover_pk is None:
                sr_pk, sc_pk = _path_start_rc(paths)
                cand_pk = state.get_unit_at(sr_pk, sc_pk)
                if (
                    cand_pk is not None
                    and cand_pk.is_alive
                    and int(cand_pk.player) == eng
                ):
                    mover_pk = cand_pk
            if mover_pk is not None and mover_pk.is_alive:
                if (int(mover_pk.pos[0]), int(mover_pk.pos[1])) != (er, ec):
                    rpk = compute_reachable_costs(state, mover_pk)
                    tail_pk = (er, ec)
                    # Phase 11J-MOVE-TRUNCATE-SHIP: drop the historical
                    # ``tail_pk not in rpk`` precondition.  AWBW post-kill duplicate
                    # ``Fire`` rows pin the attacker on ``tail_pk`` regardless of
                    # engine reachability — engine reach simply mirrors live-site
                    # drift here.  3-of-3 drilled GIDs (1619504 PK + 1630353 / 1622140
                    # FM) needed the relaxation; baseline kept the snap at 0/24 effective.
                    if not UNIT_STATS[mover_pk.unit_type].is_indirect:
                        # Phase 11J-MOVE-TRUNCATE-RESIDUALS-SHIP: hoist the
                        # diff-type evict_drift branch above the
                        # ``stance_stack`` gate.  Eviction marks the tail
                        # occupant ``hp = 0`` *before* the snap, so by the
                        # time ``_move_unit_forced`` runs there is no
                        # transport on the tile to "stack" onto.  Mirror of
                        # the LANE-L-WIDEN-SHIP (d176d5ad) opt-in pattern;
                        # closes the diff-type-friendly drift residuals
                        # observed in the residuals drill (1605367 B_COPTER
                        # over TANK; 1626181 TANK over LANDER).
                        if _oracle_path_tail_occupant_is_evictable_drift(
                            state,
                            mover_pk,
                            tail_pk,
                            engine_player=eng,
                            allow_diff_type_friendly=True,
                        ):
                            _oracle_evict_drifted_tail_occupant(
                                state, mover_pk, tail_pk
                            )
                            state._move_unit_forced(mover_pk, tail_pk)
                        elif not _oracle_fire_stance_would_stack_on_transport(
                            state, mover_pk, tail_pk
                        ):
                            if _oracle_path_tail_is_friendly_load_boarding(
                                state, mover_pk, tail_pk, engine_player=eng
                            ):
                                state.selected_move_pos = tail_pk
                            elif _oracle_path_tail_occupant_allows_forced_snap(
                                state, mover_pk, tail_pk, engine_player=eng
                            ):
                                state._move_unit_forced(mover_pk, tail_pk)
                if (int(mover_pk.pos[0]), int(mover_pk.pos[1])) != (er, ec):
                    raise UnsupportedOracleAction(
                        "Move: engine truncated path vs AWBW path end; upstream drift"
                    )
            return
        dr, dc = _oracle_fire_resolve_defender_target_pos(
            state, defender, attacker_eng=eng
        )
        uid = int(gu["units_id"])
        declared_mover_type: Optional[UnitType] = None
        raw_nm_mv = gu.get("units_name") or gu.get("units_symbol")
        if raw_nm_mv is not None and str(raw_nm_mv).strip() != "":
            declared_mover_type = _name_to_unit_type(str(raw_nm_mv).strip())

        def _fire_move_mover_ok(x: Unit) -> bool:
            if not x.is_alive or int(x.player) != eng:
                return False
            if declared_mover_type is not None and x.unit_type != declared_mover_type:
                return False
            return True

        u: Optional[Unit] = None
        for pl in state.units.values():
            for x in pl:
                if (
                    x.unit_id == uid
                    and x.is_alive
                    and int(x.player) == eng
                ):
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
            try:
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
            except OracleFireSeamNoAttackerCandidate:
                u = None
        if u is None and eng in (0, 1):
            try:
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
            except OracleFireSeamNoAttackerCandidate:
                u = None
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
        # Snapshot reachability to the JSON path tail **before** SELECT/move/attack:
        # after the strike, AWBW still records the mover on ``paths.global``[-1]; when
        # that tail was never in ``compute_reachable_costs`` here (drift vs live site),
        # ``u.pos`` can remain short of ``(er, ec)`` — reconcile like plain ``Move``
        # (Phase 9 Lane L; complements Lane G pre-attack ``fire_pos`` work).
        reach_at_fire_open = compute_reachable_costs(state, u)
        json_fire_path_end = (er, ec)
        json_fire_path_was_unreachable = json_fire_path_end not in reach_at_fire_open
        # Always select the live unit at its true tile (path start, firing tile, or drift).
        fire_pos = _oracle_resolve_fire_move_pos(state, u, paths, (er, ec), (dr, dc))
        # When the resolver snaps to a waypoint that cannot strike but the JSON path
        # end can (Manhattan direct-fire), prefer the ZIP tail — same class as GL 1618770.
        if (dr, dc) not in get_attack_targets(state, u, fire_pos):
            if (
                (dr, dc) in get_attack_targets(state, u, (er, ec))
                and not _oracle_fire_stance_would_stack_on_transport(state, u, (er, ec))
            ):
                fire_pos = (er, ec)
        costs_fire = compute_reachable_costs(state, u)
        if fire_pos not in costs_fire:
            if (
                (dr, dc) in get_attack_targets(state, u, fire_pos)
                and not _oracle_fire_stance_would_stack_on_transport(state, u, fire_pos)
            ):
                # ZIP path end / chosen stance: engine reachability omits that hex but
                # AWBW recorded the strike — snap for replay (Phase 8 Bucket A).
                state._move_unit_forced(u, fire_pos)
        start = u.pos
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
        _oracle_assert_fire_damage_table_compatible(state, u, (dr, dc))
        _oracle_assert_fire_defender_not_friendly(state, u, (dr, dc))
        _oracle_set_combat_damage_override_from_combat_info(
            state, fire_blk, envelope_awbw_player_id, u, (dr, dc),
            awbw_to_engine=awbw_to_engine,
        )
        _engine_step(
            state,
            Action(
                ActionType.ATTACK,
                unit_pos=start,
                move_pos=fire_pos,
                target_pos=(dr, dc),
                select_unit_id=su_id,
            ),
            before_engine_step,
        )
        if u is not None and u.is_alive:
            pr, pc = int(u.pos[0]), int(u.pos[1])
            if (pr, pc) != json_fire_path_end:
                st_mv = UNIT_STATS[u.unit_type]
                # Phase 11J-MOVE-TRUNCATE-SHIP: drop the
                # ``json_fire_path_was_unreachable`` precondition (mirror the PK
                # branch above).  After ATTACK the mover is pinned on ``fire_pos``
                # which can differ from ``json_fire_path_end`` even when reach
                # nominally covered the tail; AWBW's path tail is the truth.
                if not st_mv.is_indirect:
                    # Phase 11J-MOVE-TRUNCATE-RESIDUALS-SHIP: hoist the
                    # diff-type evict_drift branch above the ``stance_stack``
                    # gate.  Eviction marks the tail occupant ``hp = 0``
                    # *before* the snap, so by the time ``_move_unit_forced``
                    # runs there is no transport on the tile to "stack" onto.
                    # Mirror of the LANE-L-WIDEN-SHIP (d176d5ad) opt-in
                    # pattern; closes the diff-type-friendly drift residuals
                    # observed in the residuals drill (1605367 B_COPTER over
                    # TANK; 1626181 TANK over LANDER).  Same dominant guard
                    # rejection mode for both: ``allow_snap=False`` because
                    # occupant is a different-type friendly that prior Fire
                    # eviction defaulted to leave alone.
                    if _oracle_path_tail_occupant_is_evictable_drift(
                        state,
                        u,
                        json_fire_path_end,
                        engine_player=eng,
                        allow_diff_type_friendly=True,
                    ):
                        _oracle_evict_drifted_tail_occupant(state, u, json_fire_path_end)
                        state._move_unit_forced(u, json_fire_path_end)
                        state.selected_move_pos = json_fire_path_end
                    elif not _oracle_fire_stance_would_stack_on_transport(
                        state, u, json_fire_path_end
                    ):
                        if _oracle_path_tail_is_friendly_load_boarding(
                            state, u, json_fire_path_end, engine_player=eng
                        ):
                            state.selected_move_pos = json_fire_path_end
                        elif _oracle_path_tail_occupant_allows_forced_snap(
                            state, u, json_fire_path_end, engine_player=eng
                        ):
                            state._move_unit_forced(u, json_fire_path_end)
                            state.selected_move_pos = json_fire_path_end
            pr, pc = int(u.pos[0]), int(u.pos[1])
            if (pr, pc) != (er, ec):
                raise UnsupportedOracleAction(
                    "Move: engine truncated path vs AWBW path end; upstream drift"
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
                raise UnsupportedOracleAction(
                    "AttackSeam (no path): need combatInfo on resolved unit payload when "
                    "Move.paths.global is empty; "
                    f"gu_keys={sorted(gu.keys()) if isinstance(gu, dict) else type(gu).__name__!r}; "
                    f"unit_wrap_keys={sorted(uwrap.keys()) if isinstance(uwrap, dict) else 'n/a'}; "
                    f"AttackSeam_keys={sorted(aseam.keys()) if isinstance(aseam, dict) else type(aseam).__name__!r}"
                )
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
                    select_unit_id=int(u.unit_id),
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
        try:
            tid = int(raw_tid)
        except (TypeError, ValueError) as e:
            raise UnsupportedOracleAction(
                f"Unload: transportID not int-convertible: {raw_tid!r}"
            ) from e
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
            uwrap_u = obj.get("unit") or {}
            raise UnsupportedOracleAction(
                "Unload: could not resolve cargo snapshot (need unit.global or per-seat unit "
                "with units_players_id); "
                f"unit_wrap_type={type(uwrap_u).__name__!r}; "
                f"unit_wrap_keys={sorted(uwrap_u.keys()) if isinstance(uwrap_u, dict) else 'n/a'}; "
                f"merged_gu_keys={sorted(gu.keys()) if isinstance(gu, dict) else type(gu).__name__!r}; "
                f"flat_global_keys={sorted(gu_flat.keys()) if isinstance(gu_flat, dict) else type(gu_flat).__name__!r}"
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
            if "no transport adjacent" not in str(resolve_exc):
                raise
            raise UnsupportedOracleAction(
                "Unload: drift recovery disabled; transport/target/loaded cargo do not support UNLOAD"
            ) from resolve_exc
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
            raise UnsupportedOracleAction(
                "Unload: drift recovery disabled; transport/target/loaded cargo do not support UNLOAD"
            )
        _engine_step(state, chosen_u, before_engine_step)
        return

    # ORACLE-ONLY: replay-fidelity Delete Unit handler; never legal for RL agent.
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
    luck_seed: Optional[int] = None,
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
        luck_seed=luck_seed,
    )
    n_act = 0
    for env_i, (_pid, _day, actions) in enumerate(envs):
        if env_i < len(frames):
            oracle_set_php_id_tile_cache(state, frames[env_i])
        else:
            setattr(state, "_oracle_php_id_to_rc", {})
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
