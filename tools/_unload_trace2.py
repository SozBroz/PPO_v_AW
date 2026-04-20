"""Run oracle replay; on Unload failure, dump rich engine state."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.unit import UNIT_STATS  # noqa: E402
from engine.action import get_loadable_into  # noqa: E402
from engine.weather import effective_move_cost  # noqa: E402
from engine.terrain import INF_PASSABLE  # noqa: E402

# Reuse the helper from the simpler trace script.
from tools._unload_trace import _build_initial_state_for_zip  # noqa: E402
from tools.oracle_zip_replay import (  # noqa: E402
    UnsupportedOracleAction,
    apply_oracle_action_json,
    parse_p_envelopes_from_zip,
)


def _carrier_dump(state, eng_seat: int) -> str:
    rows = []
    for u in state.units[eng_seat]:
        if not u.is_alive:
            continue
        if UNIT_STATS[u.unit_type].carry_capacity == 0:
            continue
        cargo = ",".join(
            f"{c.unit_type.name}#{c.unit_id}" for c in u.loaded_units
        ) or "-"
        rows.append(f"{u.unit_type.name}#{u.unit_id}@{u.pos} cargo=[{cargo}]")
    return "; ".join(rows) if rows else "<no carriers>"


def _all_units_near(state, target, eng_seat, radius=3):
    rows = []
    tr, tc = target
    for seat in range(2):
        for u in state.units[seat]:
            if not u.is_alive:
                continue
            d = abs(u.pos[0] - tr) + abs(u.pos[1] - tc)
            if d <= radius:
                tag = "ENG" if seat == eng_seat else "OPP"
                cargo = ",".join(c.unit_type.name for c in u.loaded_units) or "-"
                rows.append(
                    f"  {tag} {u.unit_type.name}#{u.unit_id}@{u.pos} d={d} cargo=[{cargo}]"
                )
    return rows


def main() -> None:
    zip_path = Path(sys.argv[1])
    state, awbw_to_engine, _meta = _build_initial_state_for_zip(zip_path)
    envs = parse_p_envelopes_from_zip(zip_path)

    cumulative = 0
    last_action = None
    for i, (pid, day, acts) in enumerate(envs):
        for j, a in enumerate(acts):
            cumulative += 1
            kind = a.get("action")
            try:
                apply_oracle_action_json(
                    state, a, awbw_to_engine, envelope_awbw_player_id=int(pid)
                )
                last_action = (i, j, kind, pid, day, cumulative, a)
            except UnsupportedOracleAction as exc:
                msg = str(exc)
                if not msg.startswith("Unload"):
                    print(f"NON-UNLOAD failure at env#{i} a#{j} kind={kind}: {exc}")
                    return
                print("=" * 80)
                print(
                    f"UNLOAD FAILURE: {zip_path.name} env#{i} day={day} pid={pid} "
                    f"a#{j}/{len(acts)} cum#{cumulative}"
                )
                print(f"  msg: {exc}")
                eng = awbw_to_engine.get(int(pid), -1)
                gu = (a.get("unit") or {}).get("global") if isinstance(a.get("unit"), dict) else None
                if gu:
                    target = (int(gu["units_y"]), int(gu["units_x"]))
                    cargo_name = gu.get("units_name")
                    cargo_id = gu.get("units_id")
                    print(
                        f"  cargo: {cargo_name}#{cargo_id} target={target} "
                        f"transportID={a.get('transportID')}"
                    )
                    # Terrain at target
                    h, w = state.map_data.height, state.map_data.width
                    tr, tc = target
                    if 0 <= tr < h and 0 <= tc < w:
                        tid = state.map_data.terrain[tr][tc]
                        occ = state.get_unit_at(tr, tc)
                        print(f"  target tile terrain_id={tid} occupant={occ}")
                    # Adjacent tiles + units
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = tr + dr, tc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        adj = state.get_unit_at(nr, nc)
                        adj_t = state.map_data.terrain[nr][nc]
                        adj_str = (
                            f"{adj.unit_type.name}#{adj.unit_id} P{adj.player} "
                            f"loaded=[{','.join(c.unit_type.name for c in adj.loaded_units) or '-'}]"
                            if adj is not None else "empty"
                        )
                        print(f"  ADJ ({nr},{nc}) terrain_id={adj_t} : {adj_str}")
                    # Last engine state
                    print(f"  stage={state.action_stage.name} active={state.active_player} eng_seat={eng}")
                    if eng in (0, 1):
                        print(f"  carriers eng: {_carrier_dump(state, eng)}")
                        print(f"  carriers opp: {_carrier_dump(state, 1-eng)}")
                    print("  units near target (r<=4):")
                    for line in _all_units_near(state, target, eng, radius=4):
                        print(line)
                    if last_action is not None:
                        li, lj, lk, lp, ld, lc, _la = last_action
                        print(
                            f"  prev: env#{li} a#{lj} {lk} pid={lp} day={ld} cum#{lc}"
                        )
                return

    print("Replay finished cleanly.")


if __name__ == "__main__":
    main()
