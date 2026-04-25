"""
Build a corpus of mid-replay GameState snapshots for the PROPERTY-EQUIV
equivalence test (`tests/test_engine_legal_actions_equivalence.py`).

This is Phase 4 Thread PROPERTY-EQUIV of the
`desync_purge_engine_harden` campaign:
the equivalence test asserts, on every snapshot in the corpus, that

    {a for a in candidate_actions(state) if step(state, a) succeeds}
        == set(get_legal_actions(state))

So the corpus must be diverse, deterministic, and pickle-stable.

Pipeline (mirrors `tools/desync_audit.py::_audit_one`):

1. Stratify ~50 GL games from `data/amarriner_gl_std_catalog.json`
   by (tier, first-CO seat). Skip games with incomplete CO ids
   or missing zips.
2. For each game, replay through `tools.oracle_zip_replay` envelope
   by envelope, the same code path the audit uses.
3. Snapshot the engine `GameState` at three checkpoints per game
   (the first envelope of day 3, day 7, and day 15) via
   `copy.deepcopy` + `pickle.dump(HIGHEST_PROTOCOL)`. If a game ends
   before a checkpoint, the missed checkpoints are simply skipped.
4. Tolerate `oracle_gap` / `engine_bug` failures mid-stream: emit
   whatever snapshots have been captured up to the point of failure
   and move on.

Idempotency: with the default `--seed 1`, this script produces the
same set of pickle files on repeated runs (modulo file mtimes).
The seed is mixed into Python's process-wide RNG via
`tools.desync_audit._seed_for_game`, so combat luck rolls match the
audit pipeline.

Usage::

    python tools/build_legal_actions_equivalence_corpus.py \\
        --catalog data/amarriner_gl_std_catalog.json \\
        --out tests/data/legal_actions_equivalence_corpus \\
        --target 150 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402

from tools.desync_audit import _seed_for_game  # noqa: E402
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
OUT_DEFAULT = ROOT / "tests" / "data" / "legal_actions_equivalence_corpus"

# Per-game checkpoints: snapshot at the FIRST envelope whose ``day`` >= each
# value below. Three snapshots per game gives a nice mid-game spread; if a
# game ends before reaching one of these days, that checkpoint is silently
# skipped (the corpus skews short — that is fine).
CHECKPOINT_DAYS: tuple[tuple[str, int], ...] = (
    ("d03", 3),
    ("d07", 7),
    ("d15", 15),
)


# ---------------------------------------------------------------------------
# Catalog selection (stratified)
# ---------------------------------------------------------------------------
@dataclass
class CatalogTarget:
    games_id: int
    zip_path: Path
    meta: dict[str, Any]
    tier: str
    co_p0: int
    co_p1: int


def _load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible_targets(
    catalog: dict[str, Any],
    zips_dir: Path,
    std_map_ids: set[int],
) -> list[CatalogTarget]:
    games = catalog.get("games") or {}
    out: list[CatalogTarget] = []
    for _k, g in games.items():
        if not isinstance(g, dict) or "games_id" not in g:
            continue
        gid = int(g["games_id"])
        zpath = zips_dir / f"{gid}.zip"
        if not zpath.is_file():
            continue
        if not catalog_row_has_both_cos(g):
            continue
        mid = g.get("map_id")
        if mid is None or int(mid) not in std_map_ids:
            continue
        co_p0, co_p1 = pair_catalog_cos_ids(g)
        tier = str(g.get("tier", "T2") or "T2")
        out.append(CatalogTarget(
            games_id=gid,
            zip_path=zpath,
            meta=g,
            tier=tier,
            co_p0=int(co_p0),
            co_p1=int(co_p1),
        ))
    return out


def _stratified_pick(
    targets: list[CatalogTarget],
    n_games: int,
    seed: int,
) -> list[CatalogTarget]:
    """Stratify by (tier, first-CO seat) and pick ``n_games`` deterministically.

    Strata: (tier, co_p0). We round-robin across strata sorted by key so the
    selection is stable under the same seed and target list.
    """
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, int], list[CatalogTarget]] = defaultdict(list)
    for t in targets:
        by_stratum[(t.tier, t.co_p0)].append(t)

    for k in by_stratum:
        by_stratum[k].sort(key=lambda x: x.games_id)
        rng.shuffle(by_stratum[k])

    keys = sorted(by_stratum.keys())
    chosen: list[CatalogTarget] = []
    cursors = {k: 0 for k in keys}
    while len(chosen) < n_games:
        progressed = False
        for k in keys:
            if len(chosen) >= n_games:
                break
            i = cursors[k]
            if i < len(by_stratum[k]):
                chosen.append(by_stratum[k][i])
                cursors[k] = i + 1
                progressed = True
        if not progressed:
            break
    return chosen


# ---------------------------------------------------------------------------
# Per-game replay + checkpoint capture
# ---------------------------------------------------------------------------
@dataclass
class GameCaptureResult:
    games_id: int
    snapshots: int
    error: Optional[str] = None
    error_envelope: Optional[int] = None


def _safe_pickle(state: GameState, dest: Path) -> bool:
    """Deep-copy + pickle ``state`` to ``dest``. Returns True on success.

    deepcopy first, so any in-place mutation by subsequent envelopes does not
    contaminate prior snapshots.
    """
    try:
        snap = copy.deepcopy(state)
    except Exception as exc:  # noqa: BLE001
        print(f"[corpus]   deepcopy failed for {dest.name}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False
    try:
        with open(dest, "wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001
        print(f"[corpus]   pickle.dump failed for {dest.name}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        # Clean up partial file if any
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    # Round-trip sanity check on the first few snapshots is expensive; skip
    # by default. The test harness will fail loudly if any pickle is bad.
    return True


def _capture_game(
    target: CatalogTarget,
    map_pool: Path,
    maps_dir: Path,
    out_dir: Path,
    seed: int,
) -> GameCaptureResult:
    gid = target.games_id
    # Same RNG mixing the audit pipeline uses, so combat luck rolls in oracle
    # replay are deterministic across runs of this script.
    random.seed(_seed_for_game(seed, gid))

    try:
        frames = load_replay(target.zip_path)
        if not frames:
            return GameCaptureResult(gid, 0, error="empty_replay")
        awbw_to_engine = map_snapshot_player_ids_to_engine(
            frames[0], target.co_p0, target.co_p1
        )
        map_data = load_map(int(target.meta.get("map_id", -1)), map_pool, maps_dir)
        envelopes = parse_p_envelopes_from_zip(target.zip_path)
        if not envelopes:
            return GameCaptureResult(gid, 0, error="no_action_stream")
        first_mover = resolve_replay_first_mover(
            envelopes, frames[0], awbw_to_engine
        )
        state = make_initial_state(
            map_data,
            target.co_p0,
            target.co_p1,
            starting_funds=0,
            tier_name=target.tier or "T2",
            replay_first_mover=first_mover,
        )
    except Exception as exc:  # noqa: BLE001
        return GameCaptureResult(
            gid, 0, error=f"setup_{type(exc).__name__}:{exc}",
        )

    snapshots_taken = 0
    pending_checkpoints = list(CHECKPOINT_DAYS)  # list of (label, day)
    error: Optional[str] = None
    error_env: Optional[int] = None

    for env_i, (pid, day, actions) in enumerate(envelopes):
        if state.done:
            break

        # Take a snapshot at the FIRST envelope whose day >= the next
        # outstanding checkpoint day. We snapshot BEFORE applying this
        # envelope so the state is "envelope-aligned mid-turn" — exactly
        # the kind of state mid-turn engine code consumes.
        while pending_checkpoints and day >= pending_checkpoints[0][1]:
            label, _ = pending_checkpoints.pop(0)
            phase = "select" if state.action_stage.name == "SELECT" else state.action_stage.name.lower()
            dest = out_dir / f"{gid}_e{env_i:04d}_{label}_{phase}.pkl"
            if _safe_pickle(state, dest):
                snapshots_taken += 1

        try:
            for obj in actions:
                if state.done:
                    break
                apply_oracle_action_json(
                    state, obj, awbw_to_engine, envelope_awbw_player_id=pid
                )
        except UnsupportedOracleAction as exc:
            error = f"oracle_gap:{exc}"
            error_env = env_i
            break
        except Exception as exc:  # noqa: BLE001
            error = f"engine_bug:{type(exc).__name__}:{exc}"
            error_env = env_i
            break

    return GameCaptureResult(
        games_id=gid,
        snapshots=snapshots_taken,
        error=error,
        error_envelope=error_env,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--target", type=int, default=150,
        help="Target snapshot count. Roughly 3 snapshots per game; the "
             "script picks ceil(target/3) games and tries to capture all "
             "three checkpoints from each. Effective count <= target.",
    )
    ap.add_argument("--max-games", type=int, default=None,
                    help="Hard cap on games sampled (overrides --target / 3).")
    ap.add_argument("--seed", type=int, default=1,
                    help="Deterministic stratified pick + RNG seed for "
                         "oracle combat. Pin to 1 for reproducibility.")
    ap.add_argument("--clean", action="store_true",
                    help="Delete all .pkl files in --out before generating.")
    args = ap.parse_args()

    if not args.catalog.is_file():
        print(f"[corpus] missing catalog: {args.catalog}", file=sys.stderr)
        return 1
    if not args.zips_dir.is_dir():
        print(f"[corpus] missing zips dir: {args.zips_dir}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[corpus] missing map pool: {args.map_pool}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in args.out.glob("*.pkl"):
            p.unlink()

    catalog = _load_catalog(args.catalog)
    std_map_ids = gl_std_map_ids(args.map_pool)
    eligible = _eligible_targets(catalog, args.zips_dir, std_map_ids)
    print(f"[corpus] eligible games: {len(eligible)}", file=sys.stderr)
    if not eligible:
        print("[corpus] no eligible games — bail", file=sys.stderr)
        return 1

    n_games = args.max_games if args.max_games is not None else max(
        1, (args.target + len(CHECKPOINT_DAYS) - 1) // len(CHECKPOINT_DAYS)
    )
    chosen = _stratified_pick(eligible, n_games, args.seed)
    print(f"[corpus] games picked: {len(chosen)} (target snapshots ~{args.target})",
          file=sys.stderr)

    results: list[GameCaptureResult] = []
    total = 0
    for i, t in enumerate(chosen, 1):
        try:
            res = _capture_game(t, args.map_pool, args.maps_dir, args.out, args.seed)
        except Exception as exc:  # noqa: BLE001 — never let one game stop the batch
            traceback.print_exc()
            res = GameCaptureResult(t.games_id, 0, error=f"harness:{exc}")
        results.append(res)
        total += res.snapshots
        err = res.error or ""
        if err:
            err = f" err={err[:100]}"
        print(
            f"[{i:>3}/{len(chosen)}] gid={t.games_id} tier={t.tier} "
            f"co={t.co_p0}/{t.co_p1} snaps={res.snapshots}{err}",
            flush=True,
        )
        if total >= args.target:
            print(f"[corpus] reached target ({total} >= {args.target}) — stopping early",
                  file=sys.stderr)
            break

    print()
    print(f"[corpus] snapshots written: {total}")
    print(f"[corpus] games attempted:   {len(results)}")
    print(f"[corpus] games with err:    {sum(1 for r in results if r.error)}")
    print(f"[corpus] out_dir:           {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
