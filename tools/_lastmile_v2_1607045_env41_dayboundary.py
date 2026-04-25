"""Drill the Rachel day-21 day-start tick (income+repair) at env 41 boundary.

Instruments _resupply_on_properties for P1 to dump:
  * eligible units (id, type, pos, hp_internal, display_hp, on_property)
  * per-unit step / cost / treasury before & after
  * total repair charge
  * income credited

Then prints engine vs PHP at envelope 40 end and envelope 41 entry to verify.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    parse_p_envelopes_from_zip,
    apply_oracle_action_json,
    resolve_replay_first_mover,
    map_snapshot_player_ids_to_engine,
    UnsupportedOracleAction,
)
from engine.map_loader import load_map
from engine.game import make_initial_state
from engine.unit import UNIT_STATS


def main() -> int:
    gid = 1607045
    co0, co1 = 5, 28
    map_id = 77060
    target_env = 40  # P0 Drake's End triggers P1 Rachel day-start

    zip_path = REPO / f"replays/amarriner_gl/{gid}.zip"
    map_pool = REPO / "data/gl_map_pool.json"
    maps_dir = REPO / "data/maps"

    frames = load_replay(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)
    awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
    first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
    map_data = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(map_data, co0, co1, starting_funds=0,
                               tier_name="T1", replay_first_mover=first_mover)

    def _php_player_summary(frame: dict, eng_seat: int) -> dict:
        """Return PHP funds and per-unit hp_int for engine seat eng_seat."""
        out = {"funds": None, "units": {}}
        for k, p in (frame.get("players") or {}).items() if isinstance(frame.get("players"), dict) else []:
            try:
                pid = int(p.get("id"))
                if awbw_to_engine.get(pid) == eng_seat:
                    out["funds"] = int(p.get("funds", 0))
            except Exception:
                pass
        for u in (frame.get("units") or {}).values():
            try:
                pid = int(u.get("players_id"))
                if awbw_to_engine.get(pid) != eng_seat:
                    continue
                uid = int(u.get("id"))
                hp = float(u.get("hit_points", 0))
                out["units"][uid] = (
                    int(round(hp * 10)),
                    str(u.get("name") or u.get("unit_name") or "?"),
                    int(u.get("x", -1)),
                    int(u.get("y", -1)),
                )
            except Exception:
                pass
        return out

    def _setup_pin(env_i: int, actions: list) -> None:
        post_frame = frames[env_i + 1] if (env_i + 1) < len(frames) else None
        if post_frame is None:
            state._oracle_post_envelope_units_by_id = None
            state._oracle_post_envelope_multi_hit_defenders = None
            return
        pin: dict[int, int] = {}
        for u in (post_frame.get("units") or {}).values():
            try:
                uid = int(u["id"])
                hp = float(u["hit_points"])
            except (TypeError, ValueError, KeyError):
                continue
            pin[uid] = max(0, min(100, int(round(hp * 10))))
        end_rep: set[int] = set()
        for obj in actions:
            if isinstance(obj, dict) and obj.get("action") == "End":
                ui = obj.get("updatedInfo") or {}
                rep = ui.get("repaired") if isinstance(ui, dict) else None
                if isinstance(rep, dict):
                    rep = rep.get("global")
                if isinstance(rep, list):
                    for r in rep:
                        if isinstance(r, dict):
                            try:
                                end_rep.add(int(r.get("units_id")))
                            except Exception:
                                pass
        for uid in end_rep:
            pin.pop(uid, None)
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
            except Exception:
                continue
            def_hits[d_uid] = def_hits.get(d_uid, 0) + 1
        multi = {uid for uid, c in def_hits.items() if c > 1}
        state._oracle_post_envelope_units_by_id = pin
        state._oracle_post_envelope_multi_hit_defenders = multi

    # Replay envs 0..target_env-1 with production pin logic.
    for i, (pid, day, actions) in enumerate(envs):
        if i >= target_env:
            break
        _setup_pin(i, actions)
        for obj in actions:
            if state.done:
                break
            try:
                apply_oracle_action_json(state, obj, awbw_to_engine, envelope_awbw_player_id=pid)
            except UnsupportedOracleAction as e:
                print(f"  pre env={i} ABORT {type(e).__name__}: {e}")
                return 1

    # Now at start of env target_env. Setup pin.
    pid, day, actions = envs[target_env]
    print(f"\n=== Engine state at start of env {target_env} (Drake's day {day}) ===")
    print(f"  funds: P0={state.funds[0]}  P1={state.funds[1]}  active={state.active_player}")
    print(f"  P1 unit count: {len(state.units[1])}")

    php_pre = _php_player_summary(frames[target_env], 1)
    print(f"  PHP P1 funds at frame {target_env}: {php_pre['funds']}")

    # Find Rachel properties owned for income calc.
    p1_props = [p for p in state.properties if p.owner == 1]
    p1_income_props = [p for p in p1_props if not p.is_lab and not p.is_comm_tower]
    print(f"  P1 owned properties: total={len(p1_props)} income-eligible={len(p1_income_props)}")
    print(f"    income tiles: {[(p.col, p.row, ('hq' if p.is_hq else 'base' if p.is_base else 'air' if p.is_airport else 'port' if p.is_port else 'city')) for p in p1_income_props]}")

    # Rachel D2D heal value
    print(f"\n=== Predicted Rachel _resupply_on_properties at env {target_env} End ===")
    eligible: list = []
    for unit in state.units[1]:
        prop = state.get_property_at(*unit.pos)
        if prop is None or prop.owner != 1:
            continue
        eligible.append((unit, prop))
    eligible.sort(key=lambda up: (up[1].col, up[1].row))
    print(f"  {'col,row':>8} {'uid':>10} {'type':>10} {'hp_int':>6} {'disp':>4} {'cost':>5} {'step':>4} {'on':>5}")
    total_repair_cost = 0
    for unit, prop in eligible:
        stats = UNIT_STATS[unit.unit_type]
        cls = stats.unit_class
        is_city = not (
            prop.is_hq or prop.is_lab or prop.is_comm_tower
            or prop.is_base or prop.is_airport or prop.is_port
        )
        qualifies = False
        if cls in ("infantry", "mech", "vehicle", "pipe"):
            qualifies = prop.is_hq or prop.is_base or is_city
        elif cls in ("air", "copter"):
            qualifies = prop.is_airport
        elif cls == "naval":
            qualifies = prop.is_port
        listed = stats.cost
        display_hp = (unit.hp + 9) // 10
        if not qualifies or unit.hp >= 100 or prop.is_lab or prop.is_comm_tower:
            cost = 0; step = 0
        elif display_hp >= 10:
            cost = 0; step = 0
        else:
            display_step = min(3, 10 - display_hp)
            cost = max(1, (display_step * 10 * listed) // 100) if listed > 0 else 0
            step = min(display_step * 10, 100 - unit.hp)
        prop_kind = ('hq' if prop.is_hq else 'base' if prop.is_base else 'air' if prop.is_airport else 'port' if prop.is_port else 'city')
        print(f"  {prop.col:>3},{prop.row:<3} {unit.unit_id:>10} {unit.unit_type:>10} {unit.hp:>6} {display_hp:>4} {cost:>5} {step:>4} {prop_kind:>5}")
        total_repair_cost += cost
    print(f"  TOTAL repair cost predicted: {total_repair_cost}")
    print(f"  Income predicted: {len(p1_income_props)} * 1000 = {len(p1_income_props)*1000}")
    print(f"  Net delta to P1 funds: +{len(p1_income_props)*1000 - total_repair_cost}")

    # Now compare each unit's hp to PHP frame target_env (= pre-Rachel-tick from PHP perspective)
    # Actually frames[target_env] is "before env target_env" snapshot. For Drake's turn this is also
    # PHP's state right after Rachel's previous-turn End. So Rachel units in frames[target_env]
    # should already include Rachel's income/repair from prior boundary -- not what we want.
    # frames[target_env+1] is post-env-target_env = post-Rachel-day-start in PHP.

    php_post = _php_player_summary(frames[target_env + 1], 1)
    print(f"\n=== PHP state at frame {target_env+1} (after Drake env {target_env} End -> Rachel day start) ===")
    print(f"  PHP P1 funds: {php_post['funds']}")

    # Per-unit HP comparison (PHP after vs engine pre-tick) shows PHP's healed deltas.
    print(f"\n=== Per-Rachel-unit HP: engine_pre vs PHP_post (frame {target_env+1}) ===")
    print(f"  {'uid':>10} {'eng_pos':>10} {'eng_hp':>6} {'php_pos':>10} {'php_hp':>6} {'delta':>6} {'name':>14}")
    eng_units = {u.unit_id: u for u in state.units[1]}
    php_units = php_post["units"]
    all_uids = set(eng_units.keys()) | set(php_units.keys())
    for uid in sorted(all_uids):
        eu = eng_units.get(uid)
        pu = php_units.get(uid)
        eng_pos = f"{eu.pos[0]},{eu.pos[1]}" if eu else "-"
        eng_hp = eu.hp if eu else 0
        php_hp, php_name, php_x, php_y = (pu if pu else (0, "-", -1, -1))
        php_pos = f"{php_x},{php_y}" if pu else "-"
        delta = php_hp - eng_hp
        flag = "" if delta == 0 else "  <<"
        print(f"  {uid:>10} {eng_pos:>10} {eng_hp:>6} {php_pos:>10} {php_hp:>6} {delta:>+6} {php_name:>14}{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
