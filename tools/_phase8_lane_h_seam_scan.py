"""
Phase 8 Lane H — scan local replay zips for AttackSeam with Battleship / Piperunner.

Reads the same action stream as the oracle: gzipped ``a<games_id>`` members with
``p:`` lines (see :func:`tools.oracle_zip_replay.parse_p_envelopes_from_zip`).
There is no separate ``actions.json`` in standard AWBW export zips.

Outputs ``logs/phase8_lane_h_seam_scan.json`` and prints a one-line headline.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diff_replay_zips import load_replay  # noqa: E402
from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402


def _snapshot_id_to_name(frame0: dict[str, Any]) -> dict[int, str]:
    """Map AWBW ``units_id`` (PHP ``id``) → ``name`` from turn-0 snapshot."""
    out: dict[int, str] = {}
    raw = frame0.get("units") or {}
    if not isinstance(raw, dict):
        return out
    for u in raw.values():
        if not isinstance(u, dict):
            continue
        uid = u.get("id")
        name = u.get("name")
        if uid is None or not name:
            continue
        try:
            out[int(uid)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def _walk_collect_unit_ids(obj: Any, acc: dict[int, str]) -> None:
    """Collect ``units_id`` → ``units_name`` pairs from any nested JSON (built units, etc.)."""
    if isinstance(obj, dict):
        uid = obj.get("units_id")
        uname = obj.get("units_name")
        if uid is not None and uname:
            try:
                acc[int(uid)] = str(uname)
            except (TypeError, ValueError):
                pass
        for v in obj.values():
            _walk_collect_unit_ids(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_collect_unit_ids(v, acc)


def _build_id_to_name_for_zip(zp: Path, envelopes: list[tuple[int, int, list[dict[str, Any]]]]) -> dict[int, str]:
    acc: dict[int, str] = {}
    try:
        frames = load_replay(zp)
        if frames:
            acc.update(_snapshot_id_to_name(frames[0]))
    except Exception:  # noqa: BLE001 — best-effort; seam scan must not abort
        pass
    for _pid, _day, actions in envelopes:
        for obj in actions:
            _walk_collect_unit_ids(obj, acc)
    return acc


def _zip_has_attackseam(envelopes: list[tuple[int, int, list[dict[str, Any]]]]) -> bool:
    for _pid, _day, actions in envelopes:
        for obj in actions:
            if isinstance(obj, dict) and obj.get("action") == "AttackSeam":
                return True
    return False


def _discover_pools(replays_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return [(pool_name, sorted_zip_paths), ...]."""
    out: list[tuple[str, list[Path]]] = []
    for sub_name in ("amarriner_gl", "amarriner_gl_current"):
        d = replays_dir / sub_name
        if not d.is_dir():
            continue
        zips = sorted(d.glob("*.zip"))
        out.append((sub_name, zips))
    loose = sorted(replays_dir.glob("*.zip"))
    if loose:
        out.append(("replays_root", loose))
    return out


def _path_end_rc(path: list[dict[str, Any]]) -> tuple[int, int]:
    last = path[-1]
    return int(last["y"]), int(last["x"])


def _global_from_attackseam_unit(uwrap: Any) -> dict[str, Any]:
    if not isinstance(uwrap, dict):
        return {}
    if isinstance(uwrap.get("global"), dict):
        return uwrap["global"]
    gu: dict[str, Any] = {}
    for v in uwrap.values():
        if isinstance(v, dict) and isinstance(v.get("combatInfo"), dict) and v["combatInfo"]:
            gu = v
            break
    return gu


def _extract_attackseam_fields(
    obj: dict[str, Any],
    *,
    games_id: int,
    pool: str,
    envelope_awbw_player_id: int,
    day: int,
) -> Optional[dict[str, Any]]:
    if obj.get("action") != "AttackSeam":
        return None
    aseam = obj.get("AttackSeam") or {}
    if not isinstance(aseam, dict):
        return None
    seam_r = int(aseam["seamY"])
    seam_c = int(aseam["seamX"])
    move_raw = obj.get("Move")
    move = move_raw if isinstance(move_raw, dict) else {}

    uwrap = aseam.get("unit") or {}
    gu = _global_from_attackseam_unit(uwrap)
    ci = gu.get("combatInfo") if isinstance(gu, dict) else {}
    if not isinstance(ci, dict):
        ci = {}

    units_name = ci.get("units_name")
    units_id = ci.get("units_id")
    units_players_id = ci.get("units_players_id")

    mu = move.get("unit") if isinstance(move, dict) else None
    if isinstance(mu, dict):
        mg = mu.get("global") if isinstance(mu.get("global"), dict) else {}
        if units_name is None and isinstance(mg, dict):
            units_name = mg.get("units_name")
        if units_id is None and isinstance(mg, dict):
            units_id = mg.get("units_id")
        if units_players_id is None and isinstance(mg, dict):
            units_players_id = mg.get("units_players_id")

    paths_g = (move.get("paths") or {}).get("global") or []
    if isinstance(paths_g, list) and len(paths_g) > 0:
        ar, ac = _path_end_rc(paths_g)
    else:
        try:
            ar = int(ci["units_y"])
            ac = int(ci["units_x"])
        except (KeyError, TypeError, ValueError):
            ar, ac = -1, -1

    uid_i: Optional[int] = None
    if units_id is not None:
        try:
            uid_i = int(units_id)
        except (TypeError, ValueError):
            uid_i = None
    upid_i: Optional[int] = None
    if units_players_id is not None:
        try:
            upid_i = int(units_players_id)
        except (TypeError, ValueError):
            upid_i = None

    return {
        "games_id": games_id,
        "pool": pool,
        "day": day,
        "envelope_players_id": envelope_awbw_player_id,
        "units_name": str(units_name) if units_name is not None else None,
        "units_id": uid_i,
        "units_players_id": upid_i,
        "attacker_pos": [ar, ac],
        "seam_pos": [seam_r, seam_c],
    }


def _is_battleship(name: Optional[str]) -> bool:
    return name is not None and name.strip() == "Battleship"


def _is_piperunner(name: Optional[str]) -> bool:
    if name is None:
        return False
    n = name.strip().lower().replace(" ", "")
    return n == "piperunner"


def main() -> int:
    replays = ROOT / "replays"
    pools_spec = _discover_pools(replays)
    attacker_type_counts: Counter[str] = Counter()
    all_attackseam = 0
    scanned_zips = 0
    battleship_hits: list[dict[str, Any]] = []
    piperunner_hits: list[dict[str, Any]] = []

    scanned_pools_meta: list[dict[str, Any]] = []
    for pool_name, zips in pools_spec:
        scanned_pools_meta.append({"name": pool_name, "zip_count": len(zips)})
        for zp in zips:
            scanned_zips += 1
            stem = zp.stem
            try:
                gid = int(stem)
            except ValueError:
                gid = -1
            try:
                envelopes = parse_p_envelopes_from_zip(zp)
            except Exception:  # noqa: BLE001
                continue
            id_to_name: dict[int, str] = {}
            if _zip_has_attackseam(envelopes):
                id_to_name = _build_id_to_name_for_zip(zp, envelopes)
            for envelope_pid, day, actions in envelopes:
                for obj in actions:
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("action") != "AttackSeam":
                        continue
                    all_attackseam += 1
                    rec = _extract_attackseam_fields(
                        obj,
                        games_id=gid,
                        pool=pool_name,
                        envelope_awbw_player_id=envelope_pid,
                        day=day,
                    )
                    if rec is None:
                        continue
                    uname = rec.get("units_name")
                    uid_i = rec.get("units_id")
                    if (not uname) and isinstance(uid_i, int) and uid_i in id_to_name:
                        uname = id_to_name[uid_i]
                        rec["units_name_resolved"] = uname
                    if uname:
                        attacker_type_counts[uname] += 1
                    else:
                        attacker_type_counts["<unresolved>"] += 1
                    if _is_battleship(uname):
                        battleship_hits.append(rec)
                    if _is_piperunner(uname):
                        piperunner_hits.append(rec)

    top5 = attacker_type_counts.most_common(5)
    payload = {
        "battleship_hits": battleship_hits,
        "piperunner_hits": piperunner_hits,
        "all_attackseam_count": all_attackseam,
        "scanned_zips": scanned_zips,
        "scanned_pools": scanned_pools_meta,
        "attacker_type_top5": [{"units_name": k, "count": v} for k, v in top5],
        "attacker_resolution": (
            "JSON units_name when present; else units_id → name from turn-0 PHP snapshot "
            "plus any units_id/units_name pairs seen in the action stream (matches "
            "phase3_seam_canon sweep)."
        ),
    }
    out_path = ROOT / "logs" / "phase8_lane_h_seam_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"BATTLESHIP={len(battleship_hits)} PIPERUNNER={len(piperunner_hits)} "
        f"across {scanned_zips} zips in {len(pools_spec)} pools"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
