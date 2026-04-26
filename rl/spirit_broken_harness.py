"""
In-memory std map copy (123858 Misery) with test-only map_id, P0 gross advantage,
and multi-day ``END_TURN`` simulation (engine spirit hook on each turn end).

**Do not** add ``TEST_HARNESS_MAP_ID`` to ``data/gl_map_pool.json``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from engine.action import Action, ActionType, get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map, MapData
from engine.unit import UNIT_STATS, Unit, UnitType
from engine.terrain import MOVE_TREAD, get_terrain
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
from rl.heuristic_termination import SpiritConfig

# Reserved — never add to gl_map_pool.json
TEST_HARNESS_MAP_ID = 999_100_001
JESS_CO_ID = 14

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def harness_map_data() -> MapData:
    base = load_map(123858, _DATA_ROOT / "gl_map_pool.json", _DATA_ROOT / "maps")
    if str(base.map_type).lower() != "std":
        raise AssertionError("harness source map must be std")
    md = copy.deepcopy(base)
    md.map_id = TEST_HARNESS_MAP_ID
    md.name = "SPIRIT_BROKEN_HARNESS_IN_MEMORY_NOT_IN_POOL"
    return md


def _place_neotanks_p0(st: GameState, n: int) -> None:
    st_neo = UNIT_STATS[UnitType.NEO_TANK]
    need = n
    for r in range(st.map_data.height):
        for c in range(st.map_data.width):
            if need <= 0:
                return
            if st.get_unit_at(r, c) is not None:
                continue
            tinfo = get_terrain(st.map_data.terrain[r][c])
            if tinfo.is_property:
                continue
            if tinfo.move_costs.get(MOVE_TREAD, 0) <= 0:
                continue
            u = Unit(
                unit_type=UnitType.NEO_TANK,
                player=0,
                hp=100,
                ammo=st_neo.max_ammo if st_neo.max_ammo else 0,
                fuel=st_neo.max_fuel,
                pos=(r, c),
                moved=True,
                loaded_units=[],
                is_submerged=False,
                capture_progress=20,
            )
            u.unit_id = st._allocate_unit_id()
            st.units[0].append(u)
            need -= 1
    if need > 0:
        raise RuntimeError("could not place neotanks on open tread tiles")


def give_p0_ten_cities_from_p1_or_neutral(st: GameState) -> int:
    n = 0
    for p in st.properties:
        if n >= 10:
            break
        if p.is_hq or p.is_comm_tower or p.is_lab:
            continue
        if p.owner == 1:
            p.owner = 0
            p.capture_points = 0
            n += 1
    if n < 10:
        for p in st.properties:
            if n >= 10:
                break
            if p.is_hq or p.is_comm_tower or p.is_lab:
                continue
            if p.owner is None:
                p.owner = 0
                p.capture_points = 0
                n += 1
    st._refresh_comm_towers()
    return n


def build_snowball_harness_state(
    *,
    tier_name: str = "T4",
    max_turns: int = 30,
) -> GameState:
    """Jess mirror, handicapped P0, same as spirit snowball unit test."""
    st = make_initial_state(
        harness_map_data(), JESS_CO_ID, JESS_CO_ID, tier_name=tier_name, max_turns=max_turns
    )
    give_p0_ten_cities_from_p1_or_neutral(st)
    _place_neotanks_p0(st, 10)
    return st


def _encode_state(st: object, o: int) -> dict[str, Any]:
    from rl.encoder import encode_state

    sp, sc = encode_state(st, observer=o, belief=None)  # type: ignore[arg-type]
    return {"spatial": sp, "scalars": sc}


class HighValuePolicyStub:
    """Value head that always favors strong positive raw values (P(win) high after sigmoid)."""

    def obs_to_tensor(self, obs: dict) -> tuple:
        o = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
        return o, None

    def predict_values(self, _obs_t: object) -> torch.Tensor:
        return torch.tensor([[6.0]], dtype=torch.float32)


def stub_spirit_model() -> Any:
    m = type("M", (), {"device": "cpu", "policy": HighValuePolicyStub()})()
    return m


def play_one_full_calendar_day(
    st: GameState, *, end_turn: ActionType = ActionType.END_TURN
) -> None:
    """
    P0 and P1 each end their turn in order; calendar ``turn`` increments when P1 ends.
    All units are marked moved so END_TURN is available from SELECT.
    """
    if st.done:
        return
    turn0 = st.turn
    for _ in range(5000):
        for pl in (0, 1):
            for u in st.units[pl]:
                u.moved = True
        if st.done:
            return
        leg = get_legal_actions(st)
        ends = [a for a in leg if a.action_type == end_turn]
        if not ends:
            raise RuntimeError(
                f"no END_TURN in legal (stage={st.action_stage} ap={st.active_player}); {leg[:3]!r}..."
            )
        st.step(ends[0])
        if st.done:
            return
        if st.active_player == 0 and st.turn > turn0:
            return
    raise RuntimeError("infinite day loop — turn never advanced")


def run_train_style_spirit_loop(
    st: GameState,
    *,
    model: object,
    cfg: SpiritConfig,
    max_calendar_days: int = 10,
    encode_fn: Optional[Callable[[Any, int], dict[str, Any]]] = None,
    is_std_map: bool = True,
    map_tier_ok: bool = True,
    learner_seat: int = 0,
) -> Tuple[Optional[str], int, int]:
    """
    For each in-game day: P0 and P1 pass (END_TURN with units marked moved).
    Spirit termination is evaluated in the engine after each ``END_TURN`` when
    ``AWBW_SPIRIT_BROKEN`` is on. ``model`` / ``cfg`` / ``encode_fn`` are kept for
    API compatibility; value-head diag is not run here (use ``run_calendar_day`` in RL).

    Returns (spirit_broken_kind, final_turn, days_simulated) where kind is
    ``snowball`` / ``resign`` / ``None``.
    """
    _ = (model, cfg, encode_fn, map_tier_ok, learner_seat)
    st.spirit_map_is_std = bool(is_std_map)
    for i in range(1, max_calendar_days + 1):
        if st.done:
            return st.spirit.spirit_broken_kind, st.turn, i - 1
        play_one_full_calendar_day(st)
        if st.done:
            return st.spirit.spirit_broken_kind, st.turn, i
    return None, st.turn, max_calendar_days
