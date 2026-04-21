"""
Batch desync audit: drive the Python engine from each downloaded AWBW replay zip
via the oracle action stream, capture the first divergence, and emit a reviewable
register (JSONL + CSV summary).

What this measures
------------------
For each ``replays/amarriner_gl/{games_id}.zip`` whose ``map_id`` is in the
Global League **std** rotation (``type == \"std\"`` in ``data/gl_map_pool.json``):

1. Look up ``games_id`` in ``data/amarriner_gl_std_catalog.json`` for ``map_id``,
   ``tier``, and CO ids.
2. Step through every ``p:`` envelope using the same code path as
   ``tools/oracle_zip_replay.py`` — but instrumented so we can record *where*
   the engine first refuses to follow AWBW's recorded actions.
3. Classify the failure into a fixed taxonomy (see ``Classification`` below)
   and append one row to the register.

Games whose catalog row is missing ``co_p0_id`` or ``co_p1_id`` are **not**
replayed; they emit ``catalog_incomplete`` so you can fix the scrape or JSON
first — the engine cannot create a game without two CO ids.

What this does NOT do (yet)
---------------------------
Per-day diffing of engine state vs the embedded PHP snapshot lines. The current
oracle pipeline only steps the action stream; adding state-vs-snapshot
assertions is a future enhancement and is reserved for the
``state_mismatch_investigate`` class.

Examples::

  python tools/desync_audit.py
  python tools/desync_audit.py --max-games 10 --register logs/desync_misery.jsonl
  python tools/desync_audit.py --max-games 10 --from-bottom --register logs/desync_tail.jsonl
  python tools/desync_audit.py --games-id 272176
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

# Canonical seed for the regression gate. Pin this and never touch it without
# coordinating a new baseline — the gate compares register diffs and any
# borderline luck-roll-sensitive game (e.g. ``Fire (no path)`` strikes that
# fall back to engine RNG when AWBW combatInfo is missing) will flip class on
# any seed change. See ``logs/desync_regression_log.md`` for the rationale.
CANONICAL_SEED = 1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionStage, ActionType  # noqa: E402
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from rl.paths import LOGS_DIR, ensure_logs_dir  # noqa: E402

from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    _oracle_advance_turn_until_player,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)


def _audit_before_engine_step(*_args, **_kwargs) -> None:
    """No-op hook satisfying the EngineStepHook signature for audit-internal
    pre-roll calls. The desync audit does not need a per-step trace; oracle-
    side helpers expect a callable to fire before each engine step."""
    return None
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.replay_snapshot_compare import (  # noqa: E402
    compare_funds,
    compare_snapshot_to_engine,
    replay_snapshot_pairing,
)
from engine.unit import UNIT_STATS  # noqa: E402

CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"
REGISTER_DEFAULT = LOGS_DIR / "desync_register.jsonl"


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    """Catalog rows may use JSON ``null`` for missing CO or map ids."""
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
# Fixed strings keep downstream review (filtering, dashboards) stable.
CLS_OK = "ok"                                  # engine ran every envelope without raising
CLS_ORACLE_GAP = "oracle_gap"                  # action kind not yet mapped in oracle_zip_replay
CLS_LOADER_ERROR = "loader_error"              # snapshot CO/player mapping or zip layout problem
CLS_REPLAY_NO_ACTION_STREAM = "replay_no_action_stream"  # RV1 zip: PHP snapshot only, no p: stream (not a corrupt zip)
CLS_ENGINE_BUG = "engine_bug"                  # engine raised under a mapped action
# state_mismatch_* family — opt-in via --enable-state-mismatch. Phase 11STATE-MISMATCH:
# replay completed without exception but the engine state diverged from the PHP
# snapshot frame at the post-envelope cadence (Option B per design spec). The
# specific suffix records which axis (or combination) drifted first.
CLS_STATE_MISMATCH_FUNDS = "state_mismatch_funds"
CLS_STATE_MISMATCH_UNITS = "state_mismatch_units"
CLS_STATE_MISMATCH_MULTI = "state_mismatch_multi"
CLS_STATE_MISMATCH_INVESTIGATE = "state_mismatch_investigate"  # comparator failure / unknown layout
CLS_CATALOG_INCOMPLETE = "catalog_incomplete"  # missing co_p0_id / co_p1_id in JSON — cannot build GameState


# ---------------------------------------------------------------------------
# Instrumented replay (mirrors oracle_zip_replay.replay_oracle_zip but tracks
# day / action index at the moment of the exception)
# ---------------------------------------------------------------------------
@dataclass
class _ReplayProgress:
    envelopes_total: int = 0
    envelopes_applied: int = 0
    actions_applied: int = 0
    last_day: Optional[int] = None
    last_action_kind: Optional[str] = None
    last_envelope_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Phase 11STATE-MISMATCH — per-envelope PHP snapshot diff (opt-in)
# ---------------------------------------------------------------------------
# These knobs are OFF by default. When --enable-state-mismatch is passed,
# ``_run_replay_instrumented`` calls ``_diff_engine_vs_snapshot`` after every
# successful ``p:`` envelope (Option B cadence per phase11state_mismatch_design
# §2). The first non-empty diff stops the replay and surfaces a
# ``state_mismatch_*`` row instead of the silent ``ok`` we'd otherwise emit
# (Phase 10F: 78% of "ok" rows hide PHP drift; Phase 11K: 74.5% on n=200).
class StateMismatchError(Exception):
    """Sentinel raised by the snapshot-diff hook in ``_run_replay_instrumented``.

    Carries enough metadata for ``_audit_one`` to populate the register row's
    ``state_mismatch`` payload (envelope/frame index, PHP day, pairing mode,
    structured ``diff_summary``). Caught and re-classified by ``_classify``
    /``_classify_state_mismatch``; never escapes the audit boundary.
    """

    def __init__(
        self,
        *,
        env_i: int,
        snap_i: int,
        day_php: Optional[int],
        pairing: Optional[str],
        diff_summary: dict[str, Any],
    ) -> None:
        msg = "; ".join((diff_summary.get("human_readable") or [])[:2]) or "state mismatch"
        super().__init__(msg)
        self.env_i = env_i
        self.snap_i = snap_i
        self.day_php = day_php
        self.pairing = pairing
        self.diff_summary = diff_summary


# Name aliases the PHP comparator already tolerates (mirror
# ``replay_snapshot_compare.compare_units``). Kept local so we don't import a
# private map and so the audit's diff stays self-describing.
_PHP_NAME_ALIASES: dict[str, str] = {
    "Md.Tank": "Medium Tank",
    "Md. Tank": "Medium Tank",
}


def _canonicalize_unit_type_name(name: str) -> str:
    """Normalize unit display names for state-mismatch **type** equality only.

    Cosmetic engine vs PHP naming (spacing, abbreviations, one plural) was
    polluting ``state_mismatch_units``; see ``docs/oracle_exception_audit/
    phase11j_state_mismatch_name_normalize.md``. Empirical sample from
    ``desync_register_state_mismatch_936_retune.jsonl``: ``Megatank`` vs
    ``Mega Tank`` (21 rows), ``Missiles`` vs ``Missile`` (9 rows).

    Does **not** affect ``compare_units`` / default desync classification —
    only ``_diff_engine_vs_snapshot`` when ``--enable-state-mismatch`` is on.

    Steps: apply ``_PHP_NAME_ALIASES`` on exact strings; lowercase; strip
    spaces, hyphens, periods, underscores; fold the missile unit plural
    ``missiles`` → ``missile`` (AWBW singular/plural pair only — not generic
    ``s`` stripping); fold medium-tank spellings to one token.
    """
    s = str(name).strip()
    s = _PHP_NAME_ALIASES.get(s, s)
    core = (
        s.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("_", "")
    )
    if core == "missiles":
        core = "missile"
    if core in ("mediumtank", "mdtank", "medtank"):
        core = "mediumtank"
    return core


def _snapshot_line_is_cosmetic_type_only(line: str) -> bool:
    """True if a ``compare_snapshot_to_engine`` type line differs only cosmetically."""
    m = re.search(r"type engine='([^']+)' php='([^']+)'", line)
    if not m:
        return False
    return _canonicalize_unit_type_name(m.group(1)) == _canonicalize_unit_type_name(
        m.group(2)
    )


def _diff_engine_vs_snapshot(
    state: GameState,
    php_frame: dict[str, Any],
    awbw_to_engine: dict[int, int],
    *,
    hp_internal_tolerance: int = 0,
) -> dict[str, Any]:
    """Compare a single PHP frame to the engine ``state``.

    Returns a structured diff dict (axes + per-seat funds + count + first-K
    human-readable lines) or an empty dict when everything matches within the
    configured tolerance. ``hp_internal_tolerance`` is the maximum absolute
    delta on **internal HP** (engine ``Unit.hp`` vs ``round(php.hit_points*10)``)
    that we silently absorb. CLI default is 9 (Phase 11J-STATE-MISMATCH-
    RETUNE-SHIP: sub-display rounding-noise filter); the function-level
    default kept at 0 here so direct in-process callers (tests, ad-hoc
    scripts) still get EXACT semantics unless they opt in.

    Carried units (``carried == 'Y'`` in PHP) are excluded — they live inside
    transports' ``loaded_units`` in the engine and would otherwise duplicate
    the carrier's tile (see ``compare_units`` for the same exclusion).
    """
    funds_lines = compare_funds(php_frame, state, awbw_to_engine)
    funds_axis_present = bool(funds_lines)

    funds_engine: dict[str, int] = {}
    funds_php: dict[str, int] = {}
    funds_delta: dict[str, int] = {}
    for _k, pl in (php_frame.get("players") or {}).items():
        if not isinstance(pl, dict):
            continue
        try:
            pid = int(pl["id"])
        except (KeyError, TypeError, ValueError):
            continue
        eng = awbw_to_engine.get(pid)
        if eng is None:
            continue
        try:
            php_f = int(pl.get("funds", 0) or 0)
        except (TypeError, ValueError):
            continue
        eng_f = int(state.funds[eng])
        funds_engine[str(eng)] = eng_f
        funds_php[str(eng)] = php_f
        funds_delta[str(eng)] = eng_f - php_f

    php_by_tile: dict[tuple[int, int, int], dict[str, Any]] = {}
    for _k, u in (php_frame.get("units") or {}).items():
        if not isinstance(u, dict):
            continue
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        try:
            col, row = int(u["x"]), int(u["y"])
            pid = int(u["players_id"])
        except (KeyError, TypeError, ValueError):
            continue
        eng_seat = awbw_to_engine.get(pid)
        if eng_seat is None:
            continue
        php_by_tile[(eng_seat, row, col)] = u

    eng_by_tile: dict[tuple[int, int, int], Any] = {}
    for seat in (0, 1):
        for u in state.units[seat]:
            if u.is_alive:
                r, c = u.pos
                eng_by_tile[(seat, r, c)] = u

    units_count_mismatch = set(php_by_tile) != set(eng_by_tile)
    units_type_mismatch_count = 0
    units_hp_mismatch_count = 0
    # Keep our own human-readable lines for HP/type/count drift. Required
    # because ``compare_snapshot_to_engine`` only flags **display-bar**
    # mismatches (ceil(php.hit_points)) and silently misses internal-HP drift
    # like engine.hp=50 vs php.hit_points=4.5 (both ceil to bar 5). The diff
    # spec (§4) treats internal HP as the comparison axis, so we have to emit
    # our own lines or downstream triage sees ``state_mismatch_units`` rows
    # with no diagnostic body.
    own_lines: list[str] = []
    for key in sorted(set(php_by_tile) & set(eng_by_tile)):
        pu, eu = php_by_tile[key], eng_by_tile[key]
        php_name = str(pu.get("name", "")).strip()
        eng_name = UNIT_STATS[eu.unit_type].name
        if (
            php_name
            and _canonicalize_unit_type_name(eng_name)
            != _canonicalize_unit_type_name(php_name)
        ):
            units_type_mismatch_count += 1
            own_lines.append(
                f"at {key} type engine={eng_name!r} php={php_name!r}"
            )
            continue
        php_hp = pu.get("hit_points")
        if php_hp is None:
            continue
        try:
            php_internal = int(round(float(php_hp) * 10))
        except (TypeError, ValueError):
            continue
        delta = int(eu.hp) - php_internal
        if abs(delta) > hp_internal_tolerance:
            units_hp_mismatch_count += 1
            own_lines.append(
                f"at {key} hp engine={int(eu.hp)} php_internal={php_internal} "
                f"(php_hit_points={float(php_hp)}) delta={delta}"
            )

    if units_count_mismatch:
        only_php = sorted(set(php_by_tile) - set(eng_by_tile))
        only_eng = sorted(set(eng_by_tile) - set(php_by_tile))
        own_lines.insert(
            0,
            f"unit tile set mismatch only_in_php={only_php[:8]}"
            f"{'…' if len(only_php) > 8 else ''} only_in_engine={only_eng[:8]}"
            f"{'…' if len(only_eng) > 8 else ''}",
        )

    axes: list[str] = []
    if funds_axis_present:
        axes.append("funds")
    if units_count_mismatch:
        axes.append("units_count")
    if units_type_mismatch_count > 0:
        axes.append("units_type")
    if units_hp_mismatch_count > 0:
        axes.append("units_hp")

    if not axes:
        return {}

    # Stitched human_readable preview: funds_lines first (already structured by
    # compare_funds), then unit-axis lines from our own scan. Reuses
    # ``compare_snapshot_to_engine`` only as a fallback for lines we haven't
    # generated locally (e.g. comparator-side aliases or future axes).
    human: list[str] = list(funds_lines)
    for ln in own_lines:
        if ln not in human:
            human.append(ln)
        if len(human) >= 16:
            break
    if len(human) < 4:
        for ln in compare_snapshot_to_engine(php_frame, state, awbw_to_engine):
            if _snapshot_line_is_cosmetic_type_only(ln):
                continue
            if ln not in human:
                human.append(ln)
            if len(human) >= 16:
                break
    return {
        "axes": axes,
        "funds_engine_by_seat": funds_engine,
        "funds_php_by_seat": funds_php,
        "funds_delta_by_seat": funds_delta,
        "unit_mismatch_count": (
            units_hp_mismatch_count
            + units_type_mismatch_count
            + (1 if units_count_mismatch else 0)
        ),
        "unit_hp_mismatch_count": units_hp_mismatch_count,
        "unit_type_mismatch_count": units_type_mismatch_count,
        "unit_count_mismatch": bool(units_count_mismatch),
        "human_readable": human[:16],
    }


def _classify_state_mismatch(diff_summary: dict[str, Any]) -> str:
    """Pick the most specific ``state_mismatch_*`` class for a non-empty diff.

    Heuristic: funds-only -> ``state_mismatch_funds``; units-only ->
    ``state_mismatch_units``; both families -> ``state_mismatch_multi``;
    empty/garbled -> ``state_mismatch_investigate``. ``meta`` and
    ``properties`` are reserved (Section 3 of the design spec) and not
    emitted in this initial implementation — diff_summary today doesn't
    populate those axes.
    """
    axes = diff_summary.get("axes") or []
    has_funds = "funds" in axes
    has_units = any(a.startswith("units") for a in axes)
    if has_funds and has_units:
        return CLS_STATE_MISMATCH_MULTI
    if has_funds:
        return CLS_STATE_MISMATCH_FUNDS
    if has_units:
        return CLS_STATE_MISMATCH_UNITS
    return CLS_STATE_MISMATCH_INVESTIGATE


def _run_replay_instrumented(
    state: GameState,
    envelopes: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
    progress: _ReplayProgress,
    *,
    frames: Optional[list[dict[str, Any]]] = None,
    enable_state_mismatch: bool = False,
    hp_internal_tolerance: int = 0,
) -> Optional[Exception]:
    """
    Step the engine through ``envelopes``. On the first exception, populate
    ``progress`` with the divergence location and return the exception. Return
    ``None`` if the entire stream replayed cleanly (including resign / terminal).

    When ``enable_state_mismatch`` is True and ``frames`` is provided with a
    valid ``replay_snapshot_pairing`` (trailing or tight), a per-envelope diff
    runs after each successful envelope. The first non-empty diff is returned
    as a ``StateMismatchError`` (treated as an "exception" by the caller for
    return-type uniformity; ``_classify`` routes it to a state_mismatch_* row).
    """
    progress.envelopes_total = len(envelopes)
    n_frames = len(frames) if frames is not None else 0
    pairing = (
        replay_snapshot_pairing(n_frames, len(envelopes))
        if enable_state_mismatch and frames is not None
        else None
    )
    diff_active = enable_state_mismatch and frames is not None and pairing is not None
    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — precompute per-envelope post-frame
    # ``units_id → internal_hp`` maps. The override consumer in
    # ``oracle_zip_replay._oracle_set_combat_damage_override_from_combat_info``
    # uses these to recover sub-display-HP counter damage that
    # ``combatInfo.units_hit_points`` (integer display) silently rounds
    # away. ``frames`` is always available in the audit path (see
    # ``audit_one`` below) but kept guarded for safety.
    can_pin_post_frame = frames is not None and len(frames) >= len(envelopes) + 1
    for env_i, (_pid, day, actions) in enumerate(envelopes):
        if can_pin_post_frame:
            post_frame = frames[env_i + 1]
            pin: dict[int, int] = {}
            for u in (post_frame.get("units") or {}).values():
                try:
                    uid = int(u["id"])
                    hp = float(u["hit_points"])
                except (TypeError, ValueError, KeyError):
                    continue
                pin[uid] = max(0, min(100, int(round(hp * 10))))
            # Phase 11J-FINAL-LASTMILE — End-repaired post-frame exclusion.
            #
            # When an envelope ends with an explicit ``End`` action, AWBW
            # PHP processes the next-turn player's day-start income and
            # property repair AS PART OF the End action, then captures
            # ``frames[env_i + 1]``. Units that were repaired in that
            # tick appear in ``post_frame`` with their POST-REPAIR
            # ``hit_points`` — NOT the post-strike value the pin is
            # meant to convey. PHP exposes this directly via
            # ``End.updatedInfo.repaired`` (a list of
            # ``{units_id, units_hit_points}`` per unit healed). The
            # FIRE-FRAC-COUNTER pin (added Phase 11K) was anchored on
            # defender HPs that did NOT get repaired between strike and
            # post-frame; for End-repaired units it actively poisons
            # the override by setting ``awbw_def_hp = post_repair_hp``,
            # so engine post-strike comes out the same as PHP
            # post-repair, then engine ALSO applies its own day-start
            # repair → over-heal by exactly +1 display bar (= +20 / +30
            # internal HP for non-Rachel / Rachel COs respectively).
            # The over-heal compounds across turns and bleeds funds.
            #
            # Anchor: gid 1607045 env 17 day 9, P1 (Rachel) ends turn.
            # Drake (P0) units 190277871 (Inf @ 0,11) and 190289865
            # (Inf @ 6,16) appear in End.updatedInfo.repaired. The
            # post-frame ``hit_points`` (5.80 / 8.70) reflect Drake
            # day-10 post-repair. Engine pinned defender HP to 58/87,
            # then Drake's ``_resupply_on_properties`` added +20 each
            # → engine 78/97 vs PHP 58/87 (+20 internal each = +2
            # display bars over PHP). Compound funds drift saves
            # engine $100 (one Inf bar) at env 17 boundary, snowballs
            # to $180 by env 27 BUILD where engine TANK at (14,2)
            # needs $7000 / has $6820 → Build no-op oracle_gap.
            #
            # Fix: any unit that PHP repaired in this envelope's End
            # action is excluded from ``pin`` (and consequently from
            # ``multi`` propagation) — the consumer in
            # ``_oracle_set_combat_damage_override_from_combat_info``
            # then falls back to the per-fire combatInfo display × 10
            # for these units, which is post-strike ground truth and
            # rounds within the same display bar PHP repairs from
            # (cost identity preserved).
            end_repaired_ids: set[int] = set()
            for obj in actions:
                if not isinstance(obj, dict):
                    continue
                if obj.get("action") != "End":
                    continue
                ui = obj.get("updatedInfo")
                if not isinstance(ui, dict):
                    continue
                rep = ui.get("repaired")
                if isinstance(rep, dict):
                    rep = rep.get("global")
                if not isinstance(rep, list):
                    continue
                for r in rep:
                    if not isinstance(r, dict):
                        continue
                    raw_uid = r.get("units_id")
                    if raw_uid is None:
                        continue
                    try:
                        end_repaired_ids.add(int(raw_uid))
                    except (TypeError, ValueError):
                        continue
            for uid in end_repaired_ids:
                pin.pop(uid, None)
            # Phase 11K-FIRE-FRAC-COUNTER-SHIP — defender multi-hit guard.
            # Pre-scan the envelope's Fire combatInfo to count defender
            # appearances; only single-hit defenders get the post-frame
            # pin (multi-hit defenders fall back to per-fire combatInfo
            # display × 10, which is per-act ground truth).
            def_hits: dict[int, int] = {}
            for obj in actions:
                if not isinstance(obj, dict):
                    continue
                if obj.get("action") not in ("Fire", "AttackSeam"):
                    continue
                ci = obj.get("combatInfo")
                if not isinstance(ci, dict):
                    continue
                d = ci.get("defender")
                if not isinstance(d, dict):
                    continue
                try:
                    d_uid = int(d.get("units_id"))
                except (TypeError, ValueError):
                    continue
                def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
            multi = {uid for uid, c in def_hits.items() if c > 1}
            state._oracle_post_envelope_units_by_id = pin
            state._oracle_post_envelope_multi_hit_defenders = multi
        else:
            state._oracle_post_envelope_units_by_id = None
            state._oracle_post_envelope_multi_hit_defenders = None
        for obj in actions:
            if state.done:
                state._oracle_post_envelope_units_by_id = None
                state._oracle_post_envelope_multi_hit_defenders = None
                return None
            progress.last_day = day
            progress.last_action_kind = str(obj.get("action") or "?")
            progress.last_envelope_index = env_i
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=_pid
                )
            except Exception as exc:  # noqa: BLE001 — we classify upstream
                state._oracle_post_envelope_units_by_id = None
                state._oracle_post_envelope_multi_hit_defenders = None
                return exc
            progress.actions_applied += 1
            if state.done:
                progress.envelopes_applied = env_i + 1
                return None
        progress.envelopes_applied = env_i + 1

        # Phase 11J-FINAL-LASTMILE-V2 — post-envelope HP sync (comparator hygiene).
        #
        # The per-fire damage override (``_oracle_set_combat_damage_override_from_combat_info``)
        # syncs engine defender / attacker HP to PHP whenever a Fire action
        # carries a ``combatInfo`` block. That covers the dominant divergence
        # path but leaves a residual class:
        #
        #   * Units whose HP changed in PHP via a non-Fire path that also
        #     diverges from the engine's view (CO power AOE, Sturm Meteor,
        #     Hawke / Drake / Olaf / Von Bolt power damage, capture HP-lock,
        #     join + repair, fractional-internal carry-over from prior fires
        #     where the per-fire ``units_hit_points`` rounded the residue
        #     away).
        #   * Multi-hit defenders (excluded from the per-fire pin to avoid
        #     post-frame HP double-counting) where engine and PHP agree on
        #     the END HP but diverge mid-stream and never re-converge.
        #
        # When this drift sits dormant inside the same display bar PHP
        # repairs from, it costs nothing (cost identity preserved). When
        # the drift crosses a display-bar boundary on the very next
        # day-start, ``_resupply_on_properties`` charges a different
        # display-step than PHP and silently bleeds funds into the next
        # envelope — eventually denying a BUILD that PHP allows. Anchor:
        # gid 1607045 env 40 day 21 (Drake End triggers Rachel day-start);
        # engine charges $2600 in repair vs PHP $1300, leaving Rachel
        # $300 short on a $1000 INFANTRY build at env 41.
        #
        # Fix: at the end of every envelope, mirror the canonical PHP
        # post-envelope frame's per-unit ``hit_points`` onto the engine's
        # units when the engine has a same-seat unit at the same tile.
        # This is HP-only — positions, ownership, ammo, fuel, and unit
        # creation / death stay engine-authored. The PHP post-frame is
        # the same one already used for the per-fire pin (``frames[env_i + 1]``).
        #
        # Match key: AWBW ``units_id`` first (when the engine unit's
        # ``unit_id`` was already aliased to AWBW via a Build action
        # during this replay), then ``(player_seat, x, y)`` as a
        # fallback for predeploy units that still carry the engine's
        # monotonic ``unit_id``.
        #
        # Hard rules:
        #   * Skip when ``can_pin_post_frame`` is False (no PHP frame
        #     available — short-form replays without trailing snapshots).
        #   * Skip if engine has multiple units at the same tile (should
        #     never happen in AWBW; defensive).
        #   * Match seat by ``awbw_to_engine[php_unit.players_id]``.
        #   * Never create or remove engine units here (positional
        #     mismatches resolve via the existing oracle rails, not here).
        #
        # Validation: drives gid 1607045 from oracle_gap → ok and holds
        # the canonical 935 ok / 1 oracle_gap floor at 936 / 0 / 0.
        if can_pin_post_frame:
            post_frame_for_sync = frames[env_i + 1]
            php_units_iter = post_frame_for_sync.get("units") or {}
            if isinstance(php_units_iter, dict):
                php_units_list = list(php_units_iter.values())
            elif isinstance(php_units_iter, list):
                php_units_list = php_units_iter
            else:
                php_units_list = []
            php_by_awbw_id: dict[int, tuple[int, int, int, int]] = {}
            php_by_seat_pos: dict[tuple[int, int, int], int] = {}
            for pu in php_units_list:
                if not isinstance(pu, dict):
                    continue
                try:
                    raw_id = int(pu["id"])
                    raw_hp = float(pu["hit_points"])
                    raw_x = int(pu["x"])
                    raw_y = int(pu["y"])
                    raw_pid = int(pu["players_id"])
                except (TypeError, ValueError, KeyError):
                    continue
                seat = awbw_to_engine.get(raw_pid)
                if seat is None:
                    continue
                hp_int = max(0, min(100, int(round(raw_hp * 10))))
                php_by_awbw_id[raw_id] = (seat, raw_x, raw_y, hp_int)
                # Skip seat/pos cache for tiles already claimed (engine
                # AWBW would never put two units on one tile).
                php_by_seat_pos.setdefault((seat, raw_x, raw_y), hp_int)
            # Engine unit -> seat lookup. Use state.units which is keyed
            # by player seat already.
            for seat in (0, 1):
                for u in state.units[seat]:
                    if not getattr(u, "is_alive", True):
                        continue
                    new_hp: Optional[int] = None
                    try:
                        uid = int(u.unit_id)
                    except (TypeError, ValueError):
                        uid = None
                    if uid is not None and uid in php_by_awbw_id:
                        ps, px, py, php_hp_int = php_by_awbw_id[uid]
                        if ps == seat:
                            new_hp = php_hp_int
                    if new_hp is None:
                        # Position fallback: engine unit.pos is (row, col).
                        # PHP units use (x, y) where x=col, y=row.
                        try:
                            row, col = u.pos
                            key = (seat, int(col), int(row))
                        except (TypeError, ValueError):
                            key = None
                        if key is not None and key in php_by_seat_pos:
                            new_hp = php_by_seat_pos[key]
                    if new_hp is not None and new_hp != int(u.hp):
                        u.hp = new_hp

        if diff_active:
            snap_i = env_i + 1
            if snap_i < n_frames:
                php_frame = frames[snap_i]
                # Phase 11J-FUNDS-EXTERMINATION — PHP cadence pre-roll.
                #
                # AWBW PHP snapshots include the implicit end-of-turn that
                # the server applies between envelopes when a player's
                # action stream lacks an explicit ``End`` (timeouts, AET,
                # short-form replays). Concrete trigger: game ``1618984``
                # env_i=5 is a single ``Capt`` with no ``End``; the next
                # PHP frame has ``day == envelope.day + 1`` and reflects
                # the next player's start-of-turn income (P0 +8000g).
                # The engine catches up via
                # ``_oracle_advance_turn_until_player`` only at the START
                # of envelope env_i+1 — too late for this snapshot diff.
                #
                # When ``php_frame.day`` exceeds the envelope's ``day``,
                # AWBW has crossed an end-of-turn boundary that the
                # engine has not. Roll the engine to match the same
                # cadence before diffing — funds, fuel, supply and
                # comm-tower bookkeeping all run via ``_end_turn``.
                # ``_oracle_advance_turn_until_player`` is idempotent
                # when the engine is already on the requested seat, so
                # this is a no-op when the envelope ended cleanly with
                # an explicit ``End``.
                #
                # This is *comparator cadence alignment*, not gate logic:
                # the StateMismatchError class, ``_classify``, and the
                # ``--state-mismatch-hp-tolerance`` floor are unchanged.
                # No per-engine code path moves; only the
                # already-existing oracle helper fires one envelope
                # earlier.
                try:
                    php_day_pre = int(php_frame.get("day") or 0)
                except (TypeError, ValueError):
                    php_day_pre = 0
                env_day = int(day) if day is not None else 0
                env_player_eng = awbw_to_engine.get(int(_pid))
                # Only pre-roll when the engine has NOT yet rolled past
                # the envelope's player (i.e. the envelope ended without
                # an explicit ``End`` action, so engine is still seated
                # on ``_pid``) AND the PHP snapshot's day has advanced
                # past the envelope. This is the cadence-mismatch
                # signature: AWBW implicit AET / timeout boundary that
                # the envelope stream does not encode.
                if (
                    not state.done
                    and env_player_eng is not None
                    and int(state.active_player) == int(env_player_eng)
                    and php_day_pre > env_day
                    and state.action_stage == ActionStage.SELECT
                ):
                    other_eng = 1 - int(env_player_eng)
                    try:
                        _oracle_advance_turn_until_player(
                            state, other_eng, _audit_before_engine_step
                        )
                    except Exception:
                        pass
                ds = _diff_engine_vs_snapshot(
                    state,
                    php_frame,
                    awbw_to_engine,
                    hp_internal_tolerance=hp_internal_tolerance,
                )
                if ds:
                    php_day = php_frame.get("day")
                    try:
                        php_day_int: Optional[int] = (
                            int(php_day) if php_day is not None else None
                        )
                    except (TypeError, ValueError):
                        php_day_int = None
                    return StateMismatchError(
                        env_i=env_i,
                        snap_i=snap_i,
                        day_php=php_day_int,
                        pairing=pairing,
                        diff_summary=ds,
                    )
    return None


def _classify(exc: Optional[Exception]) -> tuple[str, str, str]:
    """Return (class, exception_type, message) for the register row."""
    if exc is None:
        return CLS_OK, "", ""
    et = type(exc).__name__
    msg = str(exc)
    if isinstance(exc, StateMismatchError):
        return _classify_state_mismatch(exc.diff_summary), et, msg
    if isinstance(exc, UnsupportedOracleAction):
        return CLS_ORACLE_GAP, et, msg
    # Snapshot / player mapping problems vs zip layout (keep patterns tight:
    # a bare ``"co" in msg`` false-positive'd on **Recon** / **recover** etc.)
    if isinstance(exc, (FileNotFoundError, KeyError)) or (
        isinstance(exc, ValueError)
        and (
            "snapshot" in msg.lower()
            or "players" in msg.lower()
            or "co_id" in msg.lower()
            or "co mapping" in msg.lower()
        )
    ):
        return CLS_LOADER_ERROR, et, msg
    return CLS_ENGINE_BUG, et, msg


# ---------------------------------------------------------------------------
# Catalog + zip selection
# ---------------------------------------------------------------------------
def _load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_catalog_files(paths: list[Path]) -> dict[str, Any]:
    """
    Load each catalog path in order and merge ``games`` by ``games_id``; the
    last file wins when the same ``games_id`` appears in more than one catalog.
    """
    by_gid: dict[int, dict[str, Any]] = {}
    for path in paths:
        raw = _load_catalog(path)
        games = raw.get("games") or {}
        for _k, g in games.items():
            if isinstance(g, dict) and "games_id" in g:
                by_gid[int(g["games_id"])] = g
    return {"games": {str(gid): by_gid[gid] for gid in sorted(by_gid)}}


def _count_zip_filter_stats(
    *,
    zips_dir: Path,
    catalog: dict[str, Any],
    games_ids: Optional[set[int]],
    std_map_ids: set[int],
) -> tuple[int, int]:
    """
    Walk ``zips_dir`` (same stem/games_ids rules as ``_iter_zip_targets``).
    For zips that have a catalog row, count how many are excluded only by the
    std map pool vs how many pass the pool but lack both CO ids (CO completeness).
    """
    games = catalog.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
    filtered_map_pool = 0
    filtered_co = 0
    if not zips_dir.is_dir():
        return 0, 0
    for p in sorted(zips_dir.glob("*.zip")):
        stem = p.stem
        if not stem.isdigit():
            continue
        gid = int(stem)
        if games_ids is not None and gid not in games_ids:
            continue
        meta = by_id.get(gid)
        if meta is None:
            continue
        mid = _meta_int(meta, "map_id", -1)
        if mid not in std_map_ids:
            filtered_map_pool += 1
            continue
        if not catalog_row_has_both_cos(meta):
            filtered_co += 1
    return filtered_map_pool, filtered_co


def _iter_zip_targets(
    *,
    zips_dir: Path,
    catalog: dict[str, Any],
    games_ids: Optional[set[int]],
    max_games: Optional[int],
    from_bottom: bool,
    std_map_ids: set[int],
) -> Iterator[tuple[int, Path, dict[str, Any]]]:
    games = catalog.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g
    if not zips_dir.is_dir():
        return
    rows: list[tuple[int, Path, dict[str, Any]]] = []
    for p in sorted(zips_dir.glob("*.zip")):
        stem = p.stem
        if not stem.isdigit():
            continue
        gid = int(stem)
        if games_ids is not None and gid not in games_ids:
            continue
        meta = by_id.get(gid)
        if meta is None:
            continue  # zip without catalog metadata — cannot pick map_id/COs
        mid = _meta_int(meta, "map_id", -1)
        if mid not in std_map_ids:
            continue
        rows.append((gid, p, meta))
    rows.sort(key=lambda t: t[0])
    if max_games is not None:
        n = max(0, max_games)
        if from_bottom:
            rows = rows[-n:]
        else:
            rows = rows[:n]
    for row in rows:
        yield row


# ---------------------------------------------------------------------------
# Per-game audit
# ---------------------------------------------------------------------------
@dataclass
class AuditRow:
    games_id: int
    map_id: int
    tier: str
    co_p0_id: int
    co_p1_id: int
    matchup: str
    zip_path: str
    status: str
    cls: str
    exception_type: str
    message: str
    approx_day: Optional[int]
    approx_action_kind: Optional[str]
    approx_envelope_index: Optional[int]
    envelopes_total: int
    envelopes_applied: int
    actions_applied: int
    # Optional structured payload populated only when --enable-state-mismatch is on
    # AND the snapshot diff hook fires. Keep ``None`` everywhere else so the
    # default-OFF JSONL is byte-identical to pre-Phase-11 audits (regression
    # gate #7 in the campaign rules).
    state_mismatch: Optional[dict[str, Any]] = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "games_id": self.games_id,
            "map_id": self.map_id,
            "tier": self.tier,
            "co_p0_id": self.co_p0_id,
            "co_p1_id": self.co_p1_id,
            "matchup": self.matchup,
            "zip_path": self.zip_path,
            "status": self.status,
            "class": self.cls,
            "exception_type": self.exception_type,
            "message": self.message,
            "approx_day": self.approx_day,
            "approx_action_kind": self.approx_action_kind,
            "approx_envelope_index": self.approx_envelope_index,
            "envelopes_total": self.envelopes_total,
            "envelopes_applied": self.envelopes_applied,
            "actions_applied": self.actions_applied,
        }
        if self.state_mismatch is not None:
            out["state_mismatch"] = self.state_mismatch
        return out


def _audit_catalog_incomplete(
    games_id: int,
    zip_path: Path,
    meta: dict[str, Any],
) -> AuditRow:
    a, b = meta.get("co_p0_id"), meta.get("co_p1_id")
    msg = (
        "catalog row missing co_p0_id and/or co_p1_id "
        f"(co_p0_id={a!r}, co_p1_id={b!r}); cannot run engine without both COs. "
        "Re-run `python tools/amarriner_gl_catalog.py build` or edit the catalog JSON."
    )
    return AuditRow(
        games_id=games_id,
        map_id=_meta_int(meta, "map_id"),
        tier=str(meta.get("tier", "")),
        co_p0_id=int(a) if a is not None else -1,
        co_p1_id=int(b) if b is not None else -1,
        matchup=str(meta.get("matchup", "")),
        zip_path=str(zip_path),
        status="skipped",
        cls=CLS_CATALOG_INCOMPLETE,
        exception_type="CatalogIncompleteCOIds",
        message=msg,
        approx_day=None,
        approx_action_kind=None,
        approx_envelope_index=None,
        envelopes_total=0,
        envelopes_applied=0,
        actions_applied=0,
    )


def _seed_for_game(seed: int, games_id: int) -> int:
    """Mix the audit's process-wide seed with the games_id so each game has a
    deterministic-but-distinct RNG stream. Bit-mixing (rather than a string
    seed) keeps reseeding cheap and avoids hash-randomization sensitivity."""
    return ((int(seed) & 0xFFFFFFFF) << 32) | (int(games_id) & 0xFFFFFFFF)


def _audit_one(
    *,
    games_id: int,
    zip_path: Path,
    meta: dict[str, Any],
    map_pool: Path,
    maps_dir: Path,
    seed: int,
    enable_state_mismatch: bool = False,
    state_mismatch_hp_tolerance: int = 0,
) -> AuditRow:
    # Pin Python's process-wide RNG to a value derived from games_id (mixed
    # with the audit's --seed). engine.combat.calculate_damage falls back to
    # ``random.randint(0, 9)`` whenever AWBW's per-strike combatInfo override
    # is missing (seam attacks, missing units_hit_points, etc.). Without this
    # reseed every audit run rolled a different luck stream, cascading into
    # unit-position drift and flipping borderline games (e.g. 1634965)
    # between ``ok`` and ``oracle_gap`` from one process to the next.
    random.seed(_seed_for_game(seed, games_id))

    co_p0, co_p1 = pair_catalog_cos_ids(meta)
    map_id = _meta_int(meta, "map_id")
    tier = str(meta.get("tier", ""))
    matchup = str(meta.get("matchup", ""))
    base = AuditRow(
        games_id=games_id,
        map_id=map_id,
        tier=tier,
        co_p0_id=co_p0,
        co_p1_id=co_p1,
        matchup=matchup,
        zip_path=str(zip_path),
        status="ok",
        cls=CLS_OK,
        exception_type="",
        message="",
        approx_day=None,
        approx_action_kind=None,
        approx_envelope_index=None,
        envelopes_total=0,
        envelopes_applied=0,
        actions_applied=0,
    )

    try:
        frames = load_replay(zip_path)
        if not frames:
            base.status = "first_divergence"
            base.cls = CLS_LOADER_ERROR
            base.exception_type = "ValueError"
            base.message = "empty replay (no PHP snapshot frames)"
            return base
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co_p0, co_p1)
        map_data = load_map(map_id, map_pool, maps_dir)
        envelopes = parse_p_envelopes_from_zip(zip_path)
        if not envelopes:
            # Distinct from loader_error: zip layout is valid AWBW RV1; site never shipped a<games_id>.
            base.status = "skipped"
            base.cls = CLS_REPLAY_NO_ACTION_STREAM
            base.exception_type = "ReplaySnapshotOnly"
            base.message = (
                "Replay zip has PHP turn snapshots only (no a<game_id> gzip with p: action lines). "
                "ReplayVersion 1 style — oracle cannot step moves; mirror may only offer this format."
            )
            base.envelopes_total = 0
            base.envelopes_applied = 0
            base.actions_applied = 0
            return base
        first_mover = resolve_replay_first_mover(envelopes, frames[0], awbw_to_engine)
        state = make_initial_state(
            map_data,
            co_p0,
            co_p1,
            starting_funds=0,
            tier_name=tier or "T2",
            replay_first_mover=first_mover,
        )
    except Exception as exc:  # noqa: BLE001 — pre-replay setup failures
        base.status = "first_divergence"
        cls, et, msg = _classify(exc)
        base.cls = cls if cls != CLS_ENGINE_BUG else CLS_LOADER_ERROR
        base.exception_type = et
        base.message = msg
        return base

    progress = _ReplayProgress()
    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — always pass ``frames`` so the
    # post-envelope ``units_id → internal_hp`` pin populates even when
    # ``enable_state_mismatch`` is False (the standard 936 audit path).
    # ``enable_state_mismatch`` still controls only the per-envelope
    # state-diff snapshotting; the pin is a separate, cheap precompute.
    exc = _run_replay_instrumented(
        state,
        envelopes,
        awbw_to_engine,
        progress,
        frames=frames,
        enable_state_mismatch=enable_state_mismatch,
        hp_internal_tolerance=state_mismatch_hp_tolerance,
    )
    base.envelopes_total = progress.envelopes_total
    base.envelopes_applied = progress.envelopes_applied
    base.actions_applied = progress.actions_applied

    if exc is None:
        base.status = "ok"
        base.cls = CLS_OK
        return base

    if isinstance(exc, StateMismatchError):
        # Snapshot-diff lane: NOT a first oracle divergence — the engine
        # consumed every action without raising; the divergence is silent
        # (Phase 10F / 11K class). Use a distinct status string so dashboards
        # and the cluster_desync_register can split state-mismatch rows from
        # ``first_divergence`` exception rows.
        base.status = "snapshot_divergence"
        cls, et, msg = _classify(exc)
        base.cls = cls
        base.exception_type = et
        base.message = msg
        base.approx_day = progress.last_day
        base.approx_action_kind = progress.last_action_kind
        base.approx_envelope_index = exc.env_i
        base.state_mismatch = {
            "first_mismatch_envelope": exc.env_i,
            "first_mismatch_frame_index": exc.snap_i,
            "first_mismatch_day_php": exc.day_php,
            "pairing": exc.pairing,
            "diff_summary": exc.diff_summary,
        }
        return base

    base.status = "first_divergence"
    base.approx_day = progress.last_day
    base.approx_action_kind = progress.last_action_kind
    base.approx_envelope_index = progress.last_envelope_index
    cls, et, msg = _classify(exc)
    base.cls = cls
    base.exception_type = et
    base.message = msg
    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the configured ``argparse.ArgumentParser`` for the audit CLI.

    Extracted from ``main()`` so regression tests (e.g.
    ``tests/test_state_mismatch_tolerance.py``) can introspect defaults
    without invoking ``sys.argv`` parsing.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--catalog",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Catalog JSON (repeatable). Merges ``games`` in order; duplicate "
            f"games_id rows use the last file. Default: {CATALOG_DEFAULT}."
        ),
    )
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--register", type=Path, default=REGISTER_DEFAULT)
    ap.add_argument("--games-id", type=int, action="append", default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument(
        "--from-bottom",
        action="store_true",
        help=(
            "With --max-games, audit the highest games_id zips (last in ascending sort) "
            "instead of the lowest."
        ),
    )
    ap.add_argument(
        "--print-traceback",
        action="store_true",
        help="Print full Python tracebacks to stderr for engine_bug rows",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_SEED,
        help=(
            "Process-wide RNG seed mixed with each games_id before that game's "
            "replay (default: CANONICAL_SEED=%(default)s). Required for the "
            "regression gate to be deterministic; see logs/desync_regression_log.md."
        ),
    )
    ap.add_argument(
        "--enable-state-mismatch",
        action="store_true",
        help=(
            "Phase 11STATE-MISMATCH: after each successful p: envelope, diff "
            "engine state vs the matching PHP snapshot frame. First mismatch "
            "stops the replay and emits a state_mismatch_{funds,units,multi} "
            "row instead of the silent ``ok`` we otherwise produce. Default OFF "
            "for backward compat with the canonical regression register; "
            "expect ~1.5-3x wall time when enabled."
        ),
    )
    ap.add_argument(
        "--state-mismatch-hp-tolerance",
        type=int,
        default=9,
        help=(
            "Maximum absolute internal-HP delta (engine.Unit.hp vs round("
            "php.hit_points*10)) absorbed silently by the state-mismatch hook. "
            "Default 9 = sub-display rounding-noise filter (Phase 11J-STATE-"
            "MISMATCH-RETUNE-SHIP). AWBW combatInfo records DISPLAY HP only "
            "(integer 1-10); the engine pins to display×10 via the existing "
            "oracle override; PHP per-day snapshots use sub-display "
            "hit_points decimals — so |Δ| ≤ 9 is rounding remainder, |Δ| ≥ 10 "
            "is genuine signal (e.g. Sonja D2D hidden HP at exactly 10). "
            "Pass --state-mismatch-hp-tolerance 0 to restore the legacy EXACT "
            "comparison. See docs/oracle_exception_audit/"
            "phase11j_state_mismatch_retune_ship.md for the empirical "
            "magnitude distribution justifying this floor."
        ),
    )
    return ap


def main() -> int:
    ap = _build_arg_parser()
    args = ap.parse_args()

    catalog_paths: list[Path] = (
        list(args.catalog) if args.catalog is not None else [CATALOG_DEFAULT]
    )
    for cp in catalog_paths:
        if not cp.is_file():
            print(f"[desync_audit] missing catalog: {cp}", file=sys.stderr)
            return 1
    if not args.zips_dir.is_dir():
        print(f"[desync_audit] missing zips dir: {args.zips_dir}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[desync_audit] missing map pool: {args.map_pool}", file=sys.stderr)
        return 1

    # Single path: use the raw JSON parse so default runs match the historical
    # ``_load_catalog`` behavior byte-for-byte in ``--register`` output.
    if len(catalog_paths) == 1:
        catalog = _load_catalog(catalog_paths[0])
    else:
        catalog = _merge_catalog_files(catalog_paths)
    games_block = catalog.get("games") or {}
    total_games = sum(
        1
        for _k, g in games_block.items()
        if isinstance(g, dict) and "games_id" in g
    )
    paths_display = " ".join(str(p) for p in catalog_paths)
    print(
        f"[desync_audit] catalogs: {paths_display} total_games={total_games}",
        file=sys.stderr,
    )
    std_map_ids = gl_std_map_ids(args.map_pool)
    gid_set = set(args.games_id) if args.games_id else None
    if args.from_bottom and args.max_games is None:
        print(
            "[desync_audit] --from-bottom without --max-games has no effect (auditing all matches)",
            file=sys.stderr,
        )
    filtered_map_pool, filtered_co = _count_zip_filter_stats(
        zips_dir=args.zips_dir,
        catalog=catalog,
        games_ids=gid_set,
        std_map_ids=std_map_ids,
    )
    targets = list(_iter_zip_targets(
        zips_dir=args.zips_dir,
        catalog=catalog,
        games_ids=gid_set,
        max_games=args.max_games,
        from_bottom=args.from_bottom,
        std_map_ids=std_map_ids,
    ))
    print(
        f"[desync_audit] zips_matched={len(targets)} "
        f"filtered_out_by_map_pool={filtered_map_pool} "
        f"filtered_out_by_co={filtered_co}",
        file=sys.stderr,
    )
    if not targets:
        print("[desync_audit] no zips matched (catalog + zips_dir intersection empty)")
        return 0

    ensure_logs_dir()
    args.register.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    rows: list[AuditRow] = []
    with open(args.register, "w", encoding="utf-8") as f:
        for gid, zpath, meta in targets:
            try:
                if not catalog_row_has_both_cos(meta):
                    row = _audit_catalog_incomplete(gid, zpath, meta)
                else:
                    row = _audit_one(
                        games_id=gid,
                        zip_path=zpath,
                        meta=meta,
                        map_pool=args.map_pool,
                        maps_dir=args.maps_dir,
                        seed=args.seed,
                        enable_state_mismatch=args.enable_state_mismatch,
                        state_mismatch_hp_tolerance=args.state_mismatch_hp_tolerance,
                    )
            except Exception as exc:  # safety net — never let one zip stop the batch
                row = AuditRow(
                    games_id=gid, map_id=_meta_int(meta, "map_id"),
                    tier=str(meta.get("tier", "")),
                    co_p0_id=_meta_int(meta, "co_p0_id"),
                    co_p1_id=_meta_int(meta, "co_p1_id"),
                    matchup=str(meta.get("matchup", "")),
                    zip_path=str(zpath), status="first_divergence",
                    cls=CLS_LOADER_ERROR, exception_type=type(exc).__name__,
                    message=f"audit harness exception: {exc}",
                    approx_day=None, approx_action_kind=None,
                    approx_envelope_index=None,
                    envelopes_total=0, envelopes_applied=0,
                    actions_applied=0,
                )
                if args.print_traceback:
                    traceback.print_exc()
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            counts[row.cls] = counts.get(row.cls, 0) + 1
            tail = row.message[:90].replace("\n", " ")
            print(
                f"[{row.games_id}] {row.cls:<28} day~{row.approx_day} "
                f"acts={row.actions_applied} | {tail}"
            )

    print()
    print(f"[desync_audit] register -> {args.register}")
    print(f"[desync_audit] {len(rows)} games audited")
    width = max((len(k) for k in counts), default=8)
    for k in sorted(counts):
        print(f"  {k:<{width}}  {counts[k]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
