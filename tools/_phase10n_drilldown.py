#!/usr/bin/env python3
"""
Phase 10N — funds drift drilldown.

Monkey-patches ``GameState._grant_income`` at runtime (no engine source edits),
replays selected zips with the same per-game RNG seed as ``tools.desync_audit``
/ ``tools._phase10f_recon``, and emits per-step engine vs PHP funds plus income
instrumentation.

Usage::

  python tools/_phase10n_drilldown.py --games-id 1628546 --games-id 1620188 --games-id 1628609
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.desync_audit import CANONICAL_SEED, _seed_for_game  # noqa: E402
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import (  # noqa: E402
    compare_snapshot_to_engine,
    replay_snapshot_pairing,
)

CATALOG_DEFAULT = ROOT / "data" / "amarriner_gl_std_catalog.json"
ZIPS_DEFAULT = ROOT / "replays" / "amarriner_gl"
MAP_POOL_DEFAULT = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR_DEFAULT = ROOT / "data" / "maps"
OUT_DEFAULT = ROOT / "logs" / "phase10n_funds_drilldown.json"


def _meta_int(meta: dict[str, Any], key: str, default: int = -1) -> int:
    v = meta.get(key, default)
    if v is None:
        return default
    return int(v)


def _load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_city_income_tiles(state: GameState, player: int) -> int:
    """
    AWBW \"City\" tiles: income property that is not HQ/base/airport/port/lab/tower.
    Used for Kindle (co_id 23) +50%% city income recon.
    """
    return sum(
        1
        for p in state.properties
        if p.owner == player
        and not p.is_comm_tower
        and not p.is_lab
        and not p.is_hq
        and not p.is_base
        and not p.is_airport
        and not p.is_port
    )


def _php_funds_by_seat(
    php_frame: dict[str, Any], awbw_to_engine: dict[int, int]
) -> dict[int, int]:
    out = {0: 0, 1: 0}
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        pid = int(pl["id"])
        eng = awbw_to_engine[pid]
        out[eng] = int(pl.get("funds", 0) or 0)
    return out


def _envelope_action_summary(actions: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for obj in actions:
        a = str(obj.get("action") or "?")
        out.append(a)
    return out[-6:]


def drill_one_game(
    *,
    games_id: int,
    zip_path: Path,
    meta: dict[str, Any],
    map_pool: Path,
    maps_dir: Path,
) -> dict[str, Any]:
    from engine.game import make_initial_state  # local import after patch setup

    random.seed(_seed_for_game(CANONICAL_SEED, games_id))

    income_events: list[dict[str, Any]] = []
    orig_grant: Callable[..., None] = GameState._grant_income

    def _patched_grant(self: GameState, player: int) -> None:
        n_inc = self.count_income_properties(player)
        n_city = _count_city_income_tiles(self, player)
        fb0, fb1 = int(self.funds[0]), int(self.funds[1])
        co_id = int(self.co_states[player].co_id)
        income_events.append({
            "turn": int(self.turn),
            "active_player": int(self.active_player),
            "income_recipient": int(player),
            "co_id": co_id,
            "weather": str(self.weather),
            "n_income_properties": n_inc,
            "n_city_tiles_kindle_relevant": n_city,
            "funds_before_grant": [fb0, fb1],
        })
        orig_grant(self, player)
        income_events[-1]["funds_after_grant"] = [int(self.funds[0]), int(self.funds[1])]
        income_events[-1]["grant_delta"] = [
            income_events[-1]["funds_after_grant"][0] - fb0,
            income_events[-1]["funds_after_grant"][1] - fb1,
        ]

    GameState._grant_income = _patched_grant  # type: ignore[assignment]
    try:
        frames = load_replay(zip_path)
        envs = parse_p_envelopes_from_zip(zip_path)
        pairing = replay_snapshot_pairing(len(frames), len(envs))
        co0, co1 = pair_catalog_cos_ids(meta)
        mid = _meta_int(meta, "map_id")
        map_data = load_map(mid, map_pool, maps_dir)
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
        first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
        state = make_initial_state(
            map_data,
            co0,
            co1,
            starting_funds=0,
            tier_name=str(meta.get("tier") or "T2"),
            replay_first_mover=first_mover,
        )

        steps: list[dict[str, Any]] = []
        initial_php = _php_funds_by_seat(frames[0], awbw_to_engine)
        initial_eng = {0: int(state.funds[0]), 1: int(state.funds[1])}
        steps.append({
            "label": "frame0_initial",
            "step_i": -1,
            "php_funds": initial_php,
            "engine_funds": initial_eng,
            "delta": {s: initial_eng[s] - initial_php[s] for s in (0, 1)},
        })

        oracle_error: Optional[str] = None
        first_nonzero: Optional[int] = None

        if pairing is None:
            oracle_error = (
                f"unsupported snapshot layout: {len(frames)} frames vs {len(envs)} envs"
            )
        elif not envs:
            oracle_error = "no p: envelopes"
        else:
            for step_i, (pid, day, actions) in enumerate(envs):
                for obj in actions:
                    try:
                        apply_oracle_action_json(
                            state,
                            obj,
                            awbw_to_engine,
                            envelope_awbw_player_id=pid,
                        )
                    except UnsupportedOracleAction as e:
                        oracle_error = f"step {step_i} UnsupportedOracleAction: {e}"
                        break
                    except Exception as e:
                        oracle_error = f"step {step_i} {type(e).__name__}: {e}"
                        break
                    if state.done:
                        oracle_error = (
                            "Game ended before zip exhausted — snapshot compare truncated"
                        )
                        break
                if oracle_error:
                    break

                snap_i = step_i + 1
                if snap_i >= len(frames):
                    continue

                php_f = _php_funds_by_seat(frames[snap_i], awbw_to_engine)
                eng_f = {0: int(state.funds[0]), 1: int(state.funds[1])}
                delta = {s: eng_f[s] - php_f[s] for s in (0, 1)}
                mm = compare_snapshot_to_engine(frames[snap_i], state, awbw_to_engine)
                row = {
                    "step_i": step_i,
                    "envelope_awbw_player_id": int(pid),
                    "envelope_day_field": int(day),
                    "engine_turn": int(state.turn),
                    "engine_active_player": int(state.active_player),
                    "engine_weather": str(state.weather),
                    "action_tail": _envelope_action_summary(actions),
                    "php_funds": php_f,
                    "engine_funds": eng_f,
                    "delta_engine_minus_php": delta,
                    "snapshot_mismatches": mm[:8],
                }
                steps.append(row)
                if first_nonzero is None and (delta[0] != 0 or delta[1] != 0):
                    first_nonzero = step_i

                if mm:
                    break

        return {
            "games_id": games_id,
            "map_id": mid,
            "tier": str(meta.get("tier") or ""),
            "co_p0_id": co0,
            "co_p1_id": co1,
            "pairing": pairing,
            "n_frames": len(frames),
            "n_envelopes": len(envs),
            "income_grant_events": income_events,
            "steps": steps,
            "first_step_i_with_nonzero_funds_delta": first_nonzero,
            "oracle_error": oracle_error,
        }
    finally:
        GameState._grant_income = orig_grant  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT)
    ap.add_argument("--zips-dir", type=Path, default=ZIPS_DEFAULT)
    ap.add_argument("--map-pool", type=Path, default=MAP_POOL_DEFAULT)
    ap.add_argument("--maps-dir", type=Path, default=MAPS_DIR_DEFAULT)
    ap.add_argument("--games-id", type=int, action="append", required=True)
    ap.add_argument("--out-json", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    cat = _load_catalog(args.catalog)
    games = cat.get("games") or {}
    by_id: dict[int, dict[str, Any]] = {}
    for _k, g in games.items():
        if isinstance(g, dict) and "games_id" in g:
            by_id[int(g["games_id"])] = g

    std_maps = gl_std_map_ids(args.map_pool)
    results: list[dict[str, Any]] = []

    for gid in args.games_id:
        meta = by_id.get(int(gid))
        zpath = args.zips_dir / f"{gid}.zip"
        if meta is None:
            results.append({"games_id": gid, "error": "missing catalog row"})
            continue
        if not catalog_row_has_both_cos(meta):
            results.append({"games_id": gid, "error": "catalog incomplete cos"})
            continue
        mid = _meta_int(meta, "map_id")
        if mid not in std_maps:
            results.append({"games_id": gid, "error": f"map_id {mid} not in std pool"})
            continue
        if not zpath.is_file():
            results.append({"games_id": gid, "error": f"missing zip {zpath}"})
            continue
        results.append(
            drill_one_game(
                games_id=int(gid),
                zip_path=zpath,
                meta=meta,
                map_pool=args.map_pool,
                maps_dir=args.maps_dir,
            )
        )

    payload = {
        "_phase10n_meta": {
            "rng": "random.seed(_seed_for_game(CANONICAL_SEED, games_id)) per game",
            "income_patch": "GameState._grant_income monkey-patch logs grants only",
        },
        "cases": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_json), "n_cases": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
