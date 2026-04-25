"""
Regression: byte-stable ``encode_state`` output vs frozen pre-restart baseline.

See tests/fixtures/encoder_equivalence_README.md and MASTERPLAN.md §12.2.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.map_loader import MapData, PropertyState  # noqa: E402
from engine.game import make_initial_state  # noqa: E402
from engine.terrain import get_terrain, get_country  # noqa: E402
from engine.unit import Unit, UnitType, UNIT_STATS  # noqa: E402
from rl.encoder import (  # noqa: E402
    GRID_SIZE,
    N_SCALARS,
    N_SPATIAL_CHANNELS,
    encode_state,
)

REGEN_ENV = "AWBW_REGEN_ENCODER_BASELINE"
BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "encoder_equivalence_pre_restart.npz"
N_STATES = 8

_META = {
    "n_states": N_STATES,
    "shapes": {
        "spatial": [GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS],
        "scalars": [N_SCALARS],
    },
    "corpus": [
        "s0: 5x5 all plain, no properties/units, opening P0, luck_seed=10001",
        "s1: 12x10 land+sea, mixed terrain, 8+ unit types (inf–naval)",
        "s2: neutral city capture in progress (property capture_points<20, inf on tile)",
        "s3: multiple units with HP 100, 64, 31",
        "s4: turn=8, active_player=1, non-zero funds",
        "s5: weather snow + co_weather_segments_remaining",
        "s6: tier T4, distinct CO ids, non-trivial power bars",
        "s7: two faction cities (P0/P1) for p0_income_share=0.5",
    ],
}


def _seed_all() -> None:
    random.seed(12345)
    np.random.seed(12345)


def _property_states_for_terrain(
    terrain: list[list[int]],
    *,
    country_to_player: dict[int, int] | None = None,
    capture_at: dict[tuple[int, int], int] | None = None,
) -> list[PropertyState]:
    country_to_player = country_to_player or {}
    capture_at = capture_at or {}
    out: list[PropertyState] = []
    for r, row in enumerate(terrain):
        for c, tid in enumerate(row):
            info = get_terrain(tid)
            if not info.is_property:
                continue
            cid = get_country(tid)
            if cid is not None and cid in country_to_player:
                owner: int | None = country_to_player[cid]
            else:
                owner = None
            cap = capture_at.get((r, c), 20)
            out.append(
                PropertyState(
                    terrain_id=tid,
                    row=r,
                    col=c,
                    owner=owner,
                    capture_points=cap,
                    is_hq=info.is_hq,
                    is_lab=info.is_lab,
                    is_comm_tower=info.is_comm_tower,
                    is_base=info.is_base,
                    is_airport=info.is_airport,
                    is_port=info.is_port,
                )
            )
    return out


def _map(
    name: str,
    map_id: int,
    terrain: list[list[int]],
    **kwargs,
) -> MapData:
    h = len(terrain)
    w = len(terrain[0]) if terrain else 0
    ctp = kwargs.pop("country_to_player", None)
    cap_at = kwargs.pop("capture_at", None)
    props = _property_states_for_terrain(terrain, country_to_player=ctp, capture_at=cap_at)
    return MapData(
        map_id=map_id,
        name=name,
        map_type="std",
        terrain=terrain,
        height=h,
        width=w,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=props,
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player=ctp or {},
        predeployed_specs=[],
    )


def _place_units(state, placements: list[tuple[UnitType, int, tuple[int, int], int]]) -> None:
    """
    ``placements``: (utype, player, (r,c), hp). Fresh ids via ``_allocate_unit_id``.
    """
    for ut, player, pos, hp in placements:
        st = UNIT_STATS[ut]
        u = Unit(
            ut,
            player,
            hp,
            st.max_ammo,
            st.max_fuel,
            pos,
            False,
            [],
            False,
            20,
        )
        u.unit_id = state._allocate_unit_id()
        state.units[player].append(u)


def _s0() -> object:
    m = _map("enc_eq_s0", 9_000_001, [[1, 1, 1, 1, 1] for _ in range(5)])
    return make_initial_state(
        m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_001
    )


def _s1() -> object:
    h, w = 12, 10
    t = [[1 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        t[0][c] = [1, 1, 3, 2, 4, 15, 34, 1, 35, 1][c]
    t[1][0] = 26
    t[2][0] = 33
    t[2][1] = 29
    t[1][1] = 1
    for r in range(7, h):
        for c in range(w):
            t[r][c] = 28
    m = _map("enc_eq_s1", 9_000_002, t)
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_002)
    s.units[0].clear()
    s.units[1].clear()
    _place_units(
        s,
        [
            (UnitType.INFANTRY, 0, (0, 0), 100),
            (UnitType.MECH, 0, (0, 7), 100),
            (UnitType.TANK, 0, (1, 1), 100),
            (UnitType.RECON, 1, (0, 1), 100),
            (UnitType.ARTILLERY, 0, (3, 0), 100),
            (UnitType.B_COPTER, 0, (4, 0), 100),
            (UnitType.BATTLESHIP, 1, (11, 0), 100),
            (UnitType.LANDER, 1, (10, 2), 100),
        ],
    )
    return s


def _s2_capture() -> object:
    t = [[1 for _ in range(8)] for _ in range(8)]
    t[4][4] = 34
    m = _map("enc_eq_s2", 9_000_003, t, capture_at={(4, 4): 6})
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_003)
    s.units[0].clear()
    s.units[1].clear()
    u = Unit(
        UnitType.INFANTRY,
        0,
        100,
        0,
        99,
        (4, 4),
        False,
        [],
        False,
        20 - 6,
    )
    u.unit_id = s._allocate_unit_id()
    s.units[0].append(u)
    return s


def _s3_hp() -> object:
    m = _map("enc_eq_s3", 9_000_004, [[1, 1, 1, 1, 1] for _ in range(5)])
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_004)
    s.units[0].clear()
    s.units[1].clear()
    _place_units(
        s,
        [
            (UnitType.INFANTRY, 0, (0, 0), 100),
            (UnitType.MED_TANK, 1, (0, 1), 64),
            (UnitType.ROCKET, 0, (0, 2), 31),
        ],
    )
    return s


def _s4_p1() -> object:
    m = _map("enc_eq_s4", 9_000_005, [[1, 1, 1, 1, 1, 1] for _ in range(5)])
    s = make_initial_state(
        m, 2, 5, starting_funds=5000, tier_name="T3", replay_first_mover=0, luck_seed=10_005, max_turns=99
    )
    s.funds[0] = 24_200
    s.funds[1] = 18_900
    s.turn = 8
    s.active_player = 1
    s.units[0].clear()
    s.units[1].clear()
    _place_units(s, [(UnitType.TANK, 0, (0, 0), 100), (UnitType.ANTI_AIR, 1, (0, 1), 100)])
    return s


def _s5_snow() -> object:
    m = _map("enc_eq_s5", 9_000_006, [[1, 1, 1] for _ in range(4)])
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_006)
    s.weather = "snow"
    s.co_weather_segments_remaining = 4
    s.units[0].clear()
    s.units[1].clear()
    _place_units(s, [(UnitType.MECH, 0, (0, 0), 100), (UnitType.MECH, 1, (0, 2), 80)])
    return s


def _s6_tier_co() -> object:
    m = _map("enc_eq_s6", 9_000_007, [[1, 1, 1, 1, 1] for _ in range(4)])
    s = make_initial_state(
        m, 3, 7, starting_funds=0, tier_name="T4", replay_first_mover=0, luck_seed=10_007
    )
    s.co_states[0].power_bar = 8500
    s.co_states[1].power_bar = 12_200
    s.units[0].clear()
    s.units[1].clear()
    _place_units(s, [(UnitType.NEO_TANK, 0, (0, 0), 100), (UnitType.ROCKET, 1, (0, 1), 100)])
    return s


def _s7_income() -> object:
    t = [[1, 1, 1] for _ in range(3)]
    t[0][0] = 38
    t[0][1] = 43
    m = _map(
        "enc_eq_s7",
        9_000_008,
        t,
        country_to_player={1: 0, 2: 1},
    )
    s = make_initial_state(
        m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0, luck_seed=10_008
    )
    s.units[0].clear()
    s.units[1].clear()
    return s


def _build_corpus():
    return (_s0(), _s1(), _s2_capture(), _s3_hp(), _s4_p1(), _s5_snow(), _s6_tier_co(), _s7_income())


def _stack_encodings(
    states: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sp0 = np.zeros((N_STATES, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
    sc0 = np.zeros((N_STATES, N_SCALARS), dtype=np.float32)
    sp1 = np.zeros((N_STATES, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
    sc1 = np.zeros((N_STATES, N_SCALARS), dtype=np.float32)
    for i, st in enumerate(states):
        a0, b0 = encode_state(st, observer=0, belief=None)
        a1, b1 = encode_state(st, observer=1, belief=None)
        sp0[i] = a0
        sc0[i] = b0
        sp1[i] = a1
        sc1[i] = b1
    return sp0, sc0, sp1, sc1


def _write_baseline() -> None:
    _seed_all()
    states = _build_corpus()
    sp0, sc0, sp1, sc1 = _stack_encodings(states)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(_META, indent=2)
    np.savez_compressed(
        BASELINE_PATH,
        spatial_o0=sp0,
        scalars_o0=sc0,
        spatial_o1=sp1,
        scalars_o1=sc1,
        meta_json=np.array(meta),
    )


def _assert_array_byte_equal(name: str, got: np.ndarray, exp: np.ndarray) -> None:
    if got.shape != exp.shape:
        raise AssertionError(f"{name}: shape mismatch got {got.shape} expected {exp.shape}")
    if got.dtype != exp.dtype or exp.dtype != np.float32:
        raise AssertionError(f"{name}: expected float32, got {got.dtype} / {exp.dtype}")
    if np.array_equal(got, exp):
        return
    d = np.abs(got.astype(np.float64) - exp.astype(np.float64))
    flat = d.ravel()
    j = int(np.argmax(flat))
    unr = np.unravel_index(j, d.shape)
    raise AssertionError(
        f"{name}: byte mismatch vs baseline (encoder layout may have changed).\n"
        f"  max_abs_diff={float(flat[j])!r} at index {unr!r} "
        f"got={float(got.ravel()[j])!r} expected={float(exp.ravel()[j])!r}\n"
        "  Confirm with the lead before regenerating the baseline; see encoder_equivalence_README.md"
    )


def test_encoder_output_matches_frozen_baseline() -> None:
    _seed_all()
    if not BASELINE_PATH.exists() and os.environ.get(REGEN_ENV) != "1":
        pytest.fail(
            f"Missing encoder baseline: {BASELINE_PATH}\n"
            f"Set {REGEN_ENV}=1 once to generate tests/fixtures/encoder_equivalence_pre_restart.npz"
        )

    states = _build_corpus()
    sp0, sc0, sp1, sc1 = _stack_encodings(states)

    if os.environ.get(REGEN_ENV) == "1":
        _write_baseline()

    with np.load(BASELINE_PATH) as z:
        bsp0 = z["spatial_o0"].astype(np.float32, copy=False)
        bsc0 = z["scalars_o0"].astype(np.float32, copy=False)
        bsp1 = z["spatial_o1"].astype(np.float32, copy=False)
        bsc1 = z["scalars_o1"].astype(np.float32, copy=False)

    _assert_array_byte_equal("spatial_o0", sp0, bsp0)
    _assert_array_byte_equal("scalars_o0", sc0, bsc0)
    _assert_array_byte_equal("spatial_o1", sp1, bsp1)
    _assert_array_byte_equal("scalars_o1", sc1, bsc1)
