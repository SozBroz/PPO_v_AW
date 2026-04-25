#!/usr/bin/env python3
"""Phase 11Y-CO-WAVE-2 one-off: inventory + drills for Sasha/Colin/Rachel (read-only report)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import load_map  # noqa: E402
from tools.amarriner_catalog_cos import (  # noqa: E402
    catalog_row_has_both_cos,
    pair_catalog_cos_ids,
)
from tools.desync_audit import (  # noqa: E402
    CANONICAL_SEED,
    _merge_catalog_files,
    _iter_zip_targets,
    _seed_for_game,
)
from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.gl_std_maps import gl_std_map_ids  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)
from tools.replay_snapshot_compare import replay_snapshot_pairing  # noqa: E402

MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
ZIPS_DIR = ROOT / "replays" / "amarriner_gl"
CATALOGS = [
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
]

CO_SASHA = 19
CO_COLIN = 15
CO_RACHEL = 28


def _php_funds(frame: dict[str, Any], awbw_pid: int) -> int:
    for pl in (frame.get("players") or {}).values():
        if not isinstance(pl, dict):
            continue
        if int(pl["id"]) == int(awbw_pid):
            return int(pl.get("funds", 0) or 0)
    raise KeyError(f"player {awbw_pid} not in frame")


def _co_for_awbw_pid(meta: dict[str, Any], awbw_pid: int, frames0: dict[str, Any]) -> int:
    co0, co1 = pair_catalog_cos_ids(meta)
    m = map_snapshot_player_ids_to_engine(frames0, co0, co1)
    eng = m[int(awbw_pid)]
    return co0 if eng == 0 else co1


def _power_events(envs: list[tuple[int, int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, (pid, day, acts) in enumerate(envs):
        for ai, a in enumerate(acts):
            if a.get("action") != "Power":
                continue
            out.append(
                {
                    "env_idx": idx,
                    "action_idx": ai,
                    "awbw_pid": int(a.get("playerID") or 0),
                    "day": day,
                    "co_name": str(a.get("coName") or ""),
                    "co_power": str(a.get("coPower") or ""),
                    "power_name": str(a.get("powerName") or ""),
                }
            )
    return out


def _step_engine_through(
    state: GameState,
    envs: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
    *,
    stop_before_env: int,
) -> None:
    for ei, (_pid, _day, actions) in enumerate(envs):
        if ei >= stop_before_env:
            break
        for obj in actions:
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=_pid)


def _sum_sasha_war_bonds_damage(
    state: GameState,
    envs: list[tuple[int, int, list[dict[str, Any]]]],
    awbw_to_engine: dict[int, int],
    *,
    sasha_engine: int,
    start_env: int,
    start_action_idx: int = 0,
) -> tuple[int, int, Optional[int]]:
    """Return (sum_primary_damage_internal, n_fires, env_idx_where_sasha_end_applied).

    Engine clears ``scop_active`` at Sasha's END_TURN. Start stepping at
    envelope ``start_env``, action offset ``start_action_idx`` (use the action
    *after* the ``Power`` row when SCOP is mid-envelope). Sum primary ``dmg``
    from ``game_log`` on Sasha-owned ``Fire`` rows while ``scop_active``.
    """
    total = 0
    n_fire = 0
    for ei, (pid, _day, actions) in enumerate(envs):
        if ei < start_env:
            continue
        eng = awbw_to_engine[int(pid)]
        start_j = start_action_idx if ei == start_env else 0
        for obj in actions[start_j:]:
            kind = obj.get("action")
            if kind == "End" and eng == sasha_engine:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
                return total, n_fire, ei
            apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            if kind == "Fire" and eng == sasha_engine:
                co = state.co_states[sasha_engine]
                if co.co_id == CO_SASHA and co.scop_active and state.game_log:
                    last = state.game_log[-1]
                    if last.get("type") == "attack":
                        d = int(last.get("dmg") or 0)
                        total += d
                        n_fire += 1
    return total, n_fire, None


def main() -> int:
    catalog = _merge_catalog_files(CATALOGS)
    games = catalog.get("games") or {}
    n_catalog = sum(
        1 for _k, g in games.items() if isinstance(g, dict) and "games_id" in g
    )
    std_ids = gl_std_map_ids(MAP_POOL)

    def count_co(co_id: int) -> tuple[int, list[int]]:
        gids: list[int] = []
        for _k, g in games.items():
            if not isinstance(g, dict) or "games_id" not in g:
                continue
            p0 = int(g.get("co_p0_id") or -1)
            p1 = int(g.get("co_p1_id") or -1)
            if p0 == co_id or p1 == co_id:
                gids.append(int(g["games_id"]))
        return len(gids), sorted(gids)

    c_sasha = count_co(CO_SASHA)
    c_colin = count_co(CO_COLIN)
    c_rachel = count_co(CO_RACHEL)

    targets = list(
        _iter_zip_targets(
            zips_dir=ZIPS_DIR,
            catalog=catalog,
            games_ids=None,
            max_games=None,
            from_bottom=False,
            std_map_ids=std_ids,
        )
    )

    def zips_with_co(co_id: int) -> list[tuple[int, Path, dict[str, Any]]]:
        return [t for t in targets if int(t[2].get("co_p0_id", -1)) == co_id or int(t[2].get("co_p1_id", -1)) == co_id]

    z_sasha = zips_with_co(CO_SASHA)
    z_colin = zips_with_co(CO_COLIN)
    z_rachel = zips_with_co(CO_RACHEL)

    # Power scans (envelope-only)
    sasha_scop_games: list[dict[str, Any]] = []
    colin_cop_games: list[dict[str, Any]] = []
    for gid, zpath, meta in targets:
        if not catalog_row_has_both_cos(meta):
            continue
        try:
            envs = parse_p_envelopes_from_zip(zpath)
        except Exception:
            continue
        if not envs:
            continue
        co0, co1 = pair_catalog_cos_ids(meta)
        try:
            fr0 = load_replay(zpath)[0]
            awbw_map = map_snapshot_player_ids_to_engine(fr0, co0, co1)
        except Exception:
            continue
        for ev in _power_events(envs):
            pid = ev["awbw_pid"]
            if pid not in awbw_map:
                continue
            eng = awbw_map[pid]
            cid = co0 if eng == 0 else co1
            if cid == CO_SASHA and ev["co_power"] == "S":
                sasha_scop_games.append({"games_id": gid, **ev})
            if cid == CO_COLIN and ev["co_power"] == "Y":
                colin_cop_games.append({"games_id": gid, **ev})

    report: dict[str, Any] = {
        "catalog_unique_games": n_catalog,
        "gl_std_zips_matched": len(targets),
        "co_counts_catalog_p0_or_p1": {
            "Sasha_19": {"n": c_sasha[0]},
            "Colin_15": {"n": c_colin[0]},
            "Rachel_28": {"n": c_rachel[0]},
        },
        "co_zips_in_gl_std_pool": {
            "Sasha_19": len(z_sasha),
            "Colin_15": len(z_colin),
            "Rachel_28": len(z_rachel),
        },
        "sasha_scop_envelope_hits": sasha_scop_games[:80],
        "sasha_scop_n_distinct_games": len({x["games_id"] for x in sasha_scop_games}),
        "colin_cop_envelope_hits": colin_cop_games[:80],
        "colin_cop_n_distinct_games": len({x["games_id"] for x in colin_cop_games}),
    }

    # --- Colin COP drill (up to 5 games) ---
    colin_drill: list[dict[str, Any]] = []
    seen_colin: set[int] = set()
    for ev in colin_cop_games:
        gid = int(ev["games_id"])
        if gid in seen_colin:
            continue
        seen_colin.add(gid)
        if len(colin_drill) >= 5:
            break
        zpath = ZIPS_DIR / f"{gid}.zip"
        meta = games[str(gid)]
        co0, co1 = pair_catalog_cos_ids(meta)
        frames = load_replay(zpath)
        envs = parse_p_envelopes_from_zip(zpath)
        pair = replay_snapshot_pairing(len(frames), len(envs))
        if pair is None:
            colin_drill.append({"games_id": gid, "error": "frame/envelope pairing unsupported"})
            continue
        k = int(ev["env_idx"])
        if k + 1 >= len(frames):
            colin_drill.append({"games_id": gid, "error": "no post-envelope frame"})
            continue
        awbw_pid = int(ev["awbw_pid"])
        try:
            pre = _php_funds(frames[k], awbw_pid)
            post = _php_funds(frames[k + 1], awbw_pid)
        except Exception as e:
            colin_drill.append({"games_id": gid, "error": f"php funds: {e}"})
            continue
        exp = int(pre * 1.5)
        colin_drill.append(
            {
                "games_id": gid,
                "env_idx": k,
                "day": ev["day"],
                "php_funds_pre": pre,
                "php_funds_post": post,
                "php_delta": post - pre,
                "expected_post_int_pre_x1_5": exp,
                "post_matches_x1_5": post == exp,
                "ratio_post_over_pre": round(post / pre, 6) if pre else None,
            }
        )

    report["colin_cop_drill_5"] = colin_drill

    # --- Sasha SCOP drill (up to 5 games with SCOP in stream) ---
    sasha_drill: list[dict[str, Any]] = []
    seen_sasha: set[int] = set()
    for ev in sasha_scop_games:
        gid = int(ev["games_id"])
        if gid in seen_sasha:
            continue
        seen_sasha.add(gid)
        if len(sasha_drill) >= 5:
            break
        zpath = ZIPS_DIR / f"{gid}.zip"
        meta = games[str(gid)]
        co0, co1 = pair_catalog_cos_ids(meta)
        map_id = int(meta.get("map_id") or 0)
        tier = str(meta.get("tier") or "T2")
        random.seed(_seed_for_game(CANONICAL_SEED, gid))
        frames = load_replay(zpath)
        envs = parse_p_envelopes_from_zip(zpath)
        pair = replay_snapshot_pairing(len(frames), len(envs))
        if pair is None:
            sasha_drill.append({"games_id": gid, "error": "pairing unsupported"})
            continue
        k = int(ev["env_idx"])
        sub_power = int(ev["action_idx"])
        awbw_map = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
        sasha_eng = awbw_map[int(ev["awbw_pid"])]
        try:
            md = load_map(map_id, MAP_POOL, MAPS_DIR)
            fm = resolve_replay_first_mover(envs, frames[0], awbw_map)
            st = make_initial_state(md, co0, co1, starting_funds=0, tier_name=tier, replay_first_mover=fm)
            _step_engine_through(st, envs, awbw_map, stop_before_env=k)
            for obj in envs[k][2][:sub_power]:
                apply_oracle_action_json(st, obj, awbw_map, envelope_awbw_player_id=envs[k][0])
            php_pre_power = _php_funds(frames[k], ev["awbw_pid"])
            eng_pre_power = int(st.funds[sasha_eng])
            apply_oracle_action_json(
                st, envs[k][2][sub_power], awbw_map, envelope_awbw_player_id=envs[k][0]
            )
            php_post_power = _php_funds(frames[k + 1], ev["awbw_pid"]) if k + 1 < len(frames) else None
            eng_post_power = int(st.funds[sasha_eng])
            st2 = make_initial_state(md, co0, co1, starting_funds=0, tier_name=tier, replay_first_mover=fm)
            _step_engine_through(st2, envs, awbw_map, stop_before_env=k)
            for obj in envs[k][2][: sub_power + 1]:
                apply_oracle_action_json(st2, obj, awbw_map, envelope_awbw_player_id=envs[k][0])
            # Remainder of env k (post-SCOP fires) + later envs until Sasha End.
            dmg_sum, n_fires, end_ei = _sum_sasha_war_bonds_damage(
                st2,
                envs,
                awbw_map,
                sasha_engine=sasha_eng,
                start_env=k,
                start_action_idx=sub_power + 1,
            )
            expected_bonus = int(dmg_sum * 0.5)
            php_after_window: Optional[int] = None
            eng_after_window = int(st2.funds[sasha_eng])
            if end_ei is not None and end_ei + 1 < len(frames):
                php_after_window = _php_funds(frames[end_ei + 1], ev["awbw_pid"])
            sasha_drill.append(
                {
                    "games_id": gid,
                    "scop_env_idx": k,
                    "day": ev["day"],
                    "php_funds_pre_power_envelope": php_pre_power,
                    "engine_funds_pre_power": eng_pre_power,
                    "php_funds_post_power_envelope": php_post_power,
                    "engine_funds_post_power": eng_post_power,
                    "sum_primary_damage_internal_scop_window": dmg_sum,
                    "n_fires_sasha_scop_window": n_fires,
                    "expected_war_bonds_bonus_gold": expected_bonus,
                    "engine_funds_after_sasha_end_turn": eng_after_window,
                    "php_funds_after_sasha_end_turn_frame": php_after_window,
                    "engine_minus_php_after_end": (
                        eng_after_window - php_after_window if php_after_window is not None else None
                    ),
                }
            )
        except (UnsupportedOracleAction, ValueError, KeyError) as e:
            sasha_drill.append({"games_id": gid, "error": f"{type(e).__name__}: {e}"})
        except Exception as e:
            sasha_drill.append({"games_id": gid, "error": f"{type(e).__name__}: {e}"})

    report["sasha_scop_drill_5"] = sasha_drill

    # --- Rachel: replay_state_diff style first funds mismatch (10 games) ---
    from tools.replay_snapshot_compare import compare_snapshot_to_engine  # noqa: E402

    rachel_drill: list[dict[str, Any]] = []
    for gid, zpath, meta in z_rachel[:25]:
        if len(rachel_drill) >= 10:
            break
        if not catalog_row_has_both_cos(meta):
            continue
        co0, co1 = pair_catalog_cos_ids(meta)
        map_id = int(meta.get("map_id") or 0)
        tier = str(meta.get("tier") or "T2")
        random.seed(_seed_for_game(CANONICAL_SEED, gid))
        try:
            frames = load_replay(zpath)
            envs = parse_p_envelopes_from_zip(zpath)
            pair = replay_snapshot_pairing(len(frames), len(envs))
            if pair is None:
                rachel_drill.append({"games_id": gid, "error": "pairing unsupported"})
                continue
            awbw_map = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
            md = load_map(map_id, MAP_POOL, MAPS_DIR)
            fm = resolve_replay_first_mover(envs, frames[0], awbw_map)
            st = make_initial_state(md, co0, co1, starting_funds=0, tier_name=tier, replay_first_mover=fm)
            init_m = compare_snapshot_to_engine(frames[0], st, awbw_map)
            if init_m:
                rachel_drill.append({"games_id": gid, "error": f"initial mismatch: {init_m[:2]}"})
                continue
            first_funds_step: Optional[int] = None
            funds_mismatch_lines: list[str] = []
            for step_i, (_pid, _day, actions) in enumerate(envs):
                for obj in actions:
                    apply_oracle_action_json(st, obj, awbw_map, envelope_awbw_player_id=_pid)
                snap_i = step_i + 1
                if snap_i >= len(frames):
                    continue
                mism = compare_snapshot_to_engine(frames[snap_i], st, awbw_map)
                funds_only = [m for m in mism if "funds" in m]
                if funds_only and first_funds_step is None:
                    first_funds_step = step_i
                    funds_mismatch_lines = funds_only[:4]
                if mism:
                    pass
            rachel_eng = 0 if co0 == CO_RACHEL else 1
            rachel_drill.append(
                {
                    "games_id": gid,
                    "matchup": meta.get("matchup"),
                    "first_funds_mismatch_step": first_funds_step,
                    "funds_mismatch_sample": funds_mismatch_lines,
                    "final_engine_funds_p_rachel": int(st.funds[rachel_eng]),
                }
            )
        except Exception as e:
            rachel_drill.append({"games_id": gid, "error": f"{type(e).__name__}: {e}"})

    report["rachel_drill_10"] = rachel_drill

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
