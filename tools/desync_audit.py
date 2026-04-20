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

from engine.action import Action, ActionType  # noqa: E402
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from rl.paths import LOGS_DIR, ensure_logs_dir  # noqa: E402

from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402

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
CLS_STATE_MISMATCH_INVESTIGATE = "state_mismatch_investigate"  # reserved (snapshot diff, not implemented)
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


def _run_replay_instrumented(
    state: GameState,
    envelopes: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
    progress: _ReplayProgress,
) -> Optional[Exception]:
    """
    Step the engine through ``envelopes``. On the first exception, populate
    ``progress`` with the divergence location and return the exception. Return
    ``None`` if the entire stream replayed cleanly (including resign / terminal).
    """
    progress.envelopes_total = len(envelopes)
    for env_i, (_pid, day, actions) in enumerate(envelopes):
        for obj in actions:
            if state.done:
                return None
            progress.last_day = day
            progress.last_action_kind = str(obj.get("action") or "?")
            progress.last_envelope_index = env_i
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=_pid
                )
            except Exception as exc:  # noqa: BLE001 — we classify upstream
                return exc
            progress.actions_applied += 1
            if state.done:
                progress.envelopes_applied = env_i + 1
                return None
        progress.envelopes_applied = env_i + 1
    return None


def _classify(exc: Optional[Exception]) -> tuple[str, str, str]:
    """Return (class, exception_type, message) for the register row."""
    if exc is None:
        return CLS_OK, "", ""
    et = type(exc).__name__
    msg = str(exc)
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

    def to_json(self) -> dict[str, Any]:
        return {
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
    exc = _run_replay_instrumented(state, envelopes, awbw_to_engine, progress)
    base.envelopes_total = progress.envelopes_total
    base.envelopes_applied = progress.envelopes_applied
    base.actions_applied = progress.actions_applied

    if exc is None:
        base.status = "ok"
        base.cls = CLS_OK
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
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
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
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[desync_audit] missing catalog: {args.catalog}", file=sys.stderr)
        return 1
    if not args.zips_dir.is_dir():
        print(f"[desync_audit] missing zips dir: {args.zips_dir}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[desync_audit] missing map pool: {args.map_pool}", file=sys.stderr)
        return 1

    catalog = _load_catalog(args.catalog)
    std_map_ids = gl_std_map_ids(args.map_pool)
    gid_set = set(args.games_id) if args.games_id else None
    if args.from_bottom and args.max_games is None:
        print(
            "[desync_audit] --from-bottom without --max-games has no effect (auditing all matches)",
            file=sys.stderr,
        )
    targets = list(_iter_zip_targets(
        zips_dir=args.zips_dir,
        catalog=catalog,
        games_ids=gid_set,
        max_games=args.max_games,
        from_bottom=args.from_bottom,
        std_map_ids=std_map_ids,
    ))
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
