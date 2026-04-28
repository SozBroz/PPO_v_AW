#!/usr/bin/env python3
"""
Automated **Replay Player snapshot vs engine** validation for AWBW ``.zip`` replays.

**Contrast target:** serialized ``awbwGame`` state embedded in the zip (gzipped
PHP lines). That is the same on-disk contract the **C# AWBW Replay Player**
deserializes and displays — not our Flask ``/replay`` UI, which reads
``game_log.jsonl`` from this engine only.

**Catalog filter:** games whose ``map_id`` is not in the Global League **std**
rotation (``type == \"std\"`` in ``--map-pool``) are skipped — delete those zips
or re-scrape; they are not valid GL-Std replay fixtures for this harness.

**Procedure:** For each zip we:

1. Load PHP game lines and ``p:`` envelopes (see ``tools.replay_snapshot_compare``
   for **trailing** vs **tight** line counts).
2. Build the engine from ``map_id`` + CO ids (catalog or CLI) and
   ``make_initial_state``.
3. Compare ``frame[0]`` to the fresh engine (**initial state**).
4. For each envelope, apply JSON actions, then compare the engine to ``frame[i+1]``
   when that line exists (tight exports omit a line after the final half-turn).

We do not shell out to the desktop viewer; mismatches are printed / JSONL rows.
Unsupported frame/envelope counts (neither trailing nor tight) fail with a short error.

Examples::

  python tools/replay_state_diff.py --games-id 1623065
  python tools/replay_state_diff.py --zips-dir replays/amarriner_gl --register logs/replay_state_diff.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402

from tools.amarriner_catalog_cos import catalog_row_has_both_cos, pair_catalog_cos_ids  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.human_demo_rows import infer_training_meta_from_awbw_zip  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.oracle_state_sync import sync_state_to_snapshot  # noqa: E402
from tools.replay_snapshot_compare import (  # noqa: E402
    compare_snapshot_to_engine,
    replay_snapshot_pairing,
)

CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"


@dataclass
class ZipDiffResult:
    games_id: int
    ok: bool
    n_frames: int
    n_envelopes: int
    aligned: bool
    """True when gzip line count matches a known trailing or tight layout."""
    pairing: Optional[str]
    """``trailing`` | ``tight`` — see ``tools.replay_snapshot_compare``."""
    initial_mismatches: list[str]
    first_step_mismatch: Optional[int]
    step_mismatches: list[str]
    oracle_error: Optional[str]
    catalog_incomplete: bool
    sync_snapped_units: int = 0
    sync_out_of_range_units: int = 0
    sync_php_only_units: int = 0
    sync_engine_only_units: int = 0
    sync_funds_snapped: int = 0
    sync_oor_examples: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _meta(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def run_zip(
    *,
    zip_path: Path,
    map_pool: Path,
    maps_dir: Path,
    map_id: int,
    co0: int,
    co1: int,
    tier_name: str,
    sync_to_php: bool = False,
) -> ZipDiffResult:
    gid = int(zip_path.stem)
    out = ZipDiffResult(
        games_id=gid,
        ok=True,
        n_frames=0,
        n_envelopes=0,
        aligned=False,
        pairing=None,
        initial_mismatches=[],
        first_step_mismatch=None,
        step_mismatches=[],
        oracle_error=None,
        catalog_incomplete=False,
    )
    try:
        frames = load_replay(zip_path)
    except Exception as e:
        out.ok = False
        out.oracle_error = f"load_replay: {type(e).__name__}: {e}"
        return out
    envs = parse_p_envelopes_from_zip(zip_path)
    out.n_frames = len(frames)
    out.n_envelopes = len(envs)
    if not envs:
        out.ok = False
        out.oracle_error = (
            "no a<game_id> action stream (ReplayVersion 1 snapshot-only zip); "
            "cannot diff oracle steps"
        )
        return out
    pairing = replay_snapshot_pairing(len(frames), len(envs))
    out.pairing = pairing
    out.aligned = pairing is not None

    if not frames:
        out.ok = False
        out.oracle_error = "empty replay"
        return out

    try:
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    except Exception as e:
        out.ok = False
        out.oracle_error = f"map_snapshot_player_ids_to_engine: {e}"
        return out

    map_data = load_map(map_id, map_pool, maps_dir)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    state = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=tier_name or "T2",
        replay_first_mover=first_mover,
    )

    out.initial_mismatches = compare_snapshot_to_engine(frames[0], state, awbw_to_engine)
    if out.initial_mismatches:
        out.ok = False
        out.first_step_mismatch = -1
        out.step_mismatches = list(out.initial_mismatches)
        return out

    if pairing is None:
        out.ok = False
        out.oracle_error = (
            f"unsupported snapshot layout: {len(frames)} gzip lines vs {len(envs)} p: envelopes"
        )
        return out

    for step_i, (_pid, _day, actions) in enumerate(envs):
        for obj in actions:
            try:
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=_pid
                )
            except UnsupportedOracleAction as e:
                out.ok = False
                out.oracle_error = f"step {step_i} UnsupportedOracleAction: {e}"
                return out
            except Exception as e:
                out.ok = False
                out.oracle_error = f"step {step_i} {type(e).__name__}: {e}"
                return out
            if state.done:
                out.oracle_error = (
                    "Game ended before zip exhausted (e.g. Resign) — snapshot compare truncated"
                )
                return out

        snap_i = step_i + 1
        if snap_i >= len(frames):
            # Tight export: no PHP line after the last half-turn.
            continue

        if sync_to_php:
            # Validate-then-snap: keep replay continuing even when individual
            # attacks differ from PHP by luck noise. The post-envelope sync
            # snaps every unit within the cap and resurrects engine units
            # the random luck wrongly killed (see ``oracle_state_sync``).
            #
            # In sync mode ``ok`` means "no out-of-range HP delta and no hard
            # oracle abort"; per-envelope structural divergence (php_only /
            # engine_only) is *expected* — that is what sync is for. Engine
            # state is reconciled to PHP after every envelope, so the next
            # envelope's actions execute against the correct board.
            rep = sync_state_to_snapshot(state, frames[snap_i], awbw_to_engine)
            out.sync_snapped_units += rep.snapped_units
            out.sync_out_of_range_units += rep.out_of_range_units
            out.sync_php_only_units += len(rep.php_only_units)
            out.sync_engine_only_units += len(rep.engine_only_units)
            out.sync_funds_snapped += len(rep.funds_snapped)
            for d in rep.deltas:
                if d.out_of_range and len(out.sync_oor_examples) < 8:
                    out.sync_oor_examples.append(
                        f"step {step_i} P{d.seat} {d.pos} {d.unit_type} "
                        f"engine_hp={d.engine_hp} php_hp={d.php_hp}"
                    )
            if rep.out_of_range_units > 0 and out.first_step_mismatch is None:
                # OOR is the only sync-mode failure: HP delta exceeded the
                # plausibility cap, meaning the engine's combat outcome was
                # outside the entire luck range — a real engine/oracle bug.
                out.first_step_mismatch = step_i
                out.ok = False
            continue

        mm = compare_snapshot_to_engine(frames[snap_i], state, awbw_to_engine)
        if mm:
            out.ok = False
            out.first_step_mismatch = step_i
            out.step_mismatches = mm
            return out

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--register", type=Path, default=None, help="Append one JSON object per zip")
    ap.add_argument("--games-id", type=int, action="append", default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument(
        "--sync",
        action="store_true",
        help=(
            "After each envelope, snap engine HPs / funds to the PHP snapshot "
            "frame and continue (instead of aborting at the first mismatch). "
            "Per-unit deltas within MAX_PLAUSIBLE_HP_SWING (~60 internal HP) "
            "are treated as luck noise and snapped silently; larger deltas are "
            "recorded as ``sync_oor_examples`` for triage. Use to measure how "
            "much drift survives once luck noise is removed."
        ),
    )
    ap.add_argument(
        "--infer-meta-from-zip",
        action="store_true",
        help=(
            "If a zip's games_id is not in --catalog, infer map_id / CO ids / tier "
            "from the zip's first PHP snapshot (``infer_training_meta_from_awbw_zip``). "
            "Non-std maps are still skipped per --map-pool. Requires a readable zip."
        ),
    )
    args = ap.parse_args()

    by_id: dict[int, dict[str, Any]] = {}
    if args.catalog.is_file():
        cat = _load_catalog(args.catalog)
        games = cat.get("games") or {}
        for _k, g in games.items():
            if isinstance(g, dict) and "games_id" in g:
                by_id[int(g["games_id"])] = g
    elif not args.infer_meta_from_zip:
        print(f"[replay_state_diff] missing catalog {args.catalog}", file=sys.stderr)
        return 1

    std_maps = gl_std_map_ids(args.map_pool)
    gid_filter = set(args.games_id) if args.games_id else None
    targets: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(args.zips_dir.glob("*.zip")):
        if not p.stem.isdigit():
            continue
        gid = int(p.stem)
        if gid_filter is not None and gid not in gid_filter:
            continue
        meta = by_id.get(gid)
        if meta is None and args.infer_meta_from_zip:
            try:
                inf = infer_training_meta_from_awbw_zip(p, map_pool=args.map_pool)
            except Exception as e:
                print(
                    f"[replay_state_diff] skip games_id={gid}: infer meta: {e}",
                    file=sys.stderr,
                )
                continue
            mid_inf = int(inf.get("map_id", 0) or 0)
            if mid_inf not in std_maps:
                if gid_filter is not None:
                    print(
                        f"[replay_state_diff] skip games_id={gid}: map_id={mid_inf} "
                        f"not in GL std pool",
                        file=sys.stderr,
                    )
                continue
            meta = {
                "games_id": gid,
                "map_id": mid_inf,
                "co_p0_id": int(inf["co0"]),
                "co_p1_id": int(inf["co1"]),
                "tier": str(inf.get("tier") or "T2"),
            }
        if meta is None:
            continue
        mid = _meta(meta, "map_id")
        if mid not in std_maps:
            if gid_filter is not None:
                print(
                    f"[replay_state_diff] skip games_id={gid}: map_id={mid} "
                    f"not in GL std pool ({args.map_pool})",
                    file=sys.stderr,
                )
            continue
        targets.append((p, meta))
    targets.sort(key=lambda t: int(t[0].stem))
    if args.max_games is not None:
        targets = targets[: max(0, args.max_games)]

    if not targets:
        print("[replay_state_diff] no zips to process")
        return 0

    reg = None
    if args.register:
        args.register.parent.mkdir(parents=True, exist_ok=True)
        reg = open(args.register, "w", encoding="utf-8")

    n_ok = 0
    try:
        for zpath, meta in targets:
            gid = int(meta["games_id"])
            if not catalog_row_has_both_cos(meta):
                r = ZipDiffResult(
                    games_id=gid,
                    ok=False,
                    n_frames=0,
                    n_envelopes=0,
                    aligned=False,
                    pairing=None,
                    initial_mismatches=[],
                    first_step_mismatch=None,
                    step_mismatches=[],
                    oracle_error="catalog missing co_p0_id/co_p1_id",
                    catalog_incomplete=True,
                )
            else:
                co0, co1 = pair_catalog_cos_ids(meta)
                r = run_zip(
                    zip_path=zpath,
                    map_pool=args.map_pool,
                    maps_dir=args.maps_dir,
                    map_id=_meta(meta, "map_id"),
                    co0=co0,
                    co1=co1,
                    tier_name=str(meta.get("tier") or "T2"),
                    sync_to_php=args.sync,
                )
            if r.ok:
                n_ok += 1
            line = json.dumps(r.to_json(), ensure_ascii=False)
            detail = r.oracle_error or ""
            if not r.ok and r.step_mismatches:
                detail = "; ".join(r.step_mismatches[:4])
            elif not r.ok and r.initial_mismatches:
                detail = "; ".join(r.initial_mismatches[:4])
            pair = r.pairing or "-"
            if args.sync:
                detail = (
                    f"snapped={r.sync_snapped_units} oor={r.sync_out_of_range_units} "
                    f"php_only={r.sync_php_only_units} engine_only={r.sync_engine_only_units} "
                    f"funds={r.sync_funds_snapped}"
                    + (f" | first_oor: {r.sync_oor_examples[0]}" if r.sync_oor_examples else "")
                )
            print(f"[{gid}] ok={r.ok} pairing={pair} {detail[:240]}")
            if reg:
                reg.write(line + "\n")
                reg.flush()
    finally:
        if reg:
            reg.close()

    print(f"[replay_state_diff] done {n_ok}/{len(targets)} ok")
    return 0 if n_ok == len(targets) else 3


if __name__ == "__main__":
    raise SystemExit(main())
