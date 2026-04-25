"""
Encodes a GameState into a numpy observation tensor.

Output shape: (H, W, N_CHANNELS) where H=W=30 (padded), N_CHANNELS is:
  - 28 unit channels (14 unit types × **me / enemy** relative to ``observer``)
  - 2 HP channels (hp_lo_ch, hp_hi_ch) — observer-aware belief interval
  - 15 terrain channels (one-hot, main terrain categories)
  - 15 property channels (5 property types × 3 states: neutral / me / enemy)
  - 2 capture-progress channels (capturing unit is **me** vs **enemy**)
  - 1 neutral contestable income-property mask (owner None, not lab/comm)
  - 6 influence channels (threat/reach/capture-ETA planes; ``engine/threat.py``)
  - 1 defense-stars channel (``TerrainInfo.defense / 4``, map-static; cached on ``MapData``)
  Total: 28 + 2 + 15 + 15 + 3 + 6 + 1 = 70 spatial channels

Plus scalar features (**ego-centric**, ``observer`` = engine seat of “me”):
  - funds[me], funds[enemy] (normalized by 50000)
  - co_power_bar[me], co_power_bar[enemy] (normalized by max)
  - cop_active[me], scop_active[me], cop_active[enemy], scop_active[enemy] (binary)
  - turn (normalized by MAX_TURNS)
  - **my_turn**: 1.0 iff ``state.active_player == observer`` (not raw seat id)
  - co_id[me], co_id[enemy] (normalized by 30)
  - tier (normalized: T1=0.25, T2=0.5, T3=0.75, T4=1.0)
  - weather_rain / weather_snow / weather_turns (same as pre-bundle)
  - **me** income share (contestable income tiles owned by ``observer`` / total)
  Total scalars: 17

HP belief (see ``docs/hp_belief.md``)
-------------------------------------
Humans see enemy HP as a single bar (display bucket 1..10 = ceil(hp/10)) and
narrow the plausible interval via the damage formula after each combat. The
engine owns the exact 0..100 integer; the encoder never reads it directly
for enemy units when a belief overlay is provided.

``encode_state(state, *, observer=0, belief=None)``:

- ``belief`` is an optional ``engine.belief.BeliefState`` keyed to the
  observer. When supplied, enemy-unit HP is read from the belief interval
  ``(hp_min, hp_max) / 100``. Observer's own units are read from the
  engine (lo == hi == unit.hp / 100).
- When ``belief`` is ``None`` (tools, debug, legacy callers), both HP
  channels collapse to ``unit.hp / 100`` — identical values in both
  channels, equivalent to the pre-belief single-HP layout.

**Checkpoint compatibility:** legacy 62-channel zips (single HP) load via
``rl.ckpt_compat.load_maskable_ppo_compat`` — stem weights and Adam moments
are expanded to 63 channels by duplicating the old HP channel into
``hp_lo`` / ``hp_hi``. The 63→70 channel bump (influence placeholders +
defense stars) is a **restart-bundle** contract; do not load pre-restart
policy weights without a stem transplant. See ``docs/hp_belief.md``.
"""
from __future__ import annotations

import os
import numpy as np

from engine.belief import BeliefState
from engine.game import GameState, MAX_TURNS
from engine.map_loader import MapData
from engine.terrain import get_terrain
from engine.threat import compute_influence_planes
from engine.unit import UnitType

# Flag to enable/disable Cython optimizations
USE_CYTHON = True
try:
    from . import _encoder_cython
except ImportError:
    USE_CYTHON = False

GRID_SIZE = 30
N_UNIT_CHANNELS = 28       # 14 unit types × 2 players
N_HP_CHANNELS = 2          # hp_lo, hp_hi (belief interval)
N_TERRAIN_CHANNELS = 15
N_PROPERTY_CHANNELS = 15   # 5 property types × 3 ownership states
N_CAPTURE_EXTRA_CHANNELS = 3  # p0 progress, p1 progress, neutral-income mask
# Influence planes (MASTER_SPEC / influence_channels_spec); see ``compute_influence_planes``.
N_INFLUENCE_CHANNELS = 6
N_DEFENSE_STARS_CHANNELS = 1
N_SPATIAL_CHANNELS = (
    N_UNIT_CHANNELS
    + N_HP_CHANNELS
    + N_TERRAIN_CHANNELS
    + N_PROPERTY_CHANNELS
    + N_CAPTURE_EXTRA_CHANNELS
    + N_INFLUENCE_CHANNELS
    + N_DEFENSE_STARS_CHANNELS
)  # 70

# First channel index for the 6-plane influence block (63..68); last channel 69 = defense stars.
N_INFLUENCE_CHANNEL_BASE = (
    N_UNIT_CHANNELS
    + N_HP_CHANNELS
    + N_TERRAIN_CHANNELS
    + N_PROPERTY_CHANNELS
    + N_CAPTURE_EXTRA_CHANNELS
)
N_DEFENSE_STARS_CHANNEL = N_SPATIAL_CHANNELS - 1
N_SCALARS = 17

# Terrain one-hot categories
TERRAIN_CATEGORIES: dict[str, int] = {
    "plain": 0,
    "mountain": 1,
    "wood": 2,
    "road": 3,
    "river": 4,
    "bridge": 5,
    "sea": 6,
    "shoal": 7,
    "reef": 8,
    "city": 9,
    "base": 10,
    "airport": 11,
    "port": 12,
    "hq": 13,
    "lab": 14,
}

# Property type index (0-4) for the 5 ownership slots
_PROP_TYPE_HQ_LAB = 4
_PROP_TYPE_BASE = 1
_PROP_TYPE_AIRPORT = 2
_PROP_TYPE_PORT = 3
_PROP_TYPE_CITY = 0

# UnitType → channel 0-13 (clamped so unknown types fall into last bucket)
_N_UNIT_TYPES = 14
UNIT_TO_CHANNEL: dict[UnitType, int] = {ut: min(int(ut), _N_UNIT_TYPES - 1) for ut in UnitType}

# Tier name → normalized scalar
_TIER_MAP: dict[str, float] = {
    "T0": 0.0,
    "TL": 0.1,
    "T1": 0.25,
    "T2": 0.5,
    "T3": 0.75,
    "T4": 1.0,
}


def _compute_terrain_category_uncached(terrain_id: int) -> int:
    """Map a raw terrain tile id to a TERRAIN_CATEGORIES index (ground truth for table build)."""
    info = get_terrain(terrain_id)
    # Property sub-types first (most specific)
    if info.is_hq:
        return TERRAIN_CATEGORIES["hq"]
    if info.is_lab:
        return TERRAIN_CATEGORIES["lab"]
    if info.is_airport:
        return TERRAIN_CATEGORIES["airport"]
    if info.is_port:
        return TERRAIN_CATEGORIES["port"]
    if info.is_base:
        return TERRAIN_CATEGORIES["base"]
    if info.is_property:
        return TERRAIN_CATEGORIES["city"]
    # Non-property: match by name substring
    name = info.name.lower() if hasattr(info, "name") else ""
    for cat, idx in TERRAIN_CATEGORIES.items():
        if cat in name:
            return idx
    return TERRAIN_CATEGORIES["plain"]  # default fallback


# Precomputed terrain_id -> category index. Built once at module import from
# the static TERRAIN_TABLE in engine/terrain.py. Phase 2a hot-path optimization:
# encode_state calls _get_terrain_category up to GRID_SIZE*GRID_SIZE = 900 times
# per observation; substring scan + str.lower() per call dominated cProfile.
def _build_terrain_category_table() -> dict[int, int]:
    from engine.terrain import TERRAIN_TABLE  # local import to avoid cycles

    table: dict[int, int] = {}
    for tid in TERRAIN_TABLE.keys():
        table[tid] = _compute_terrain_category_uncached(tid)
    return table


_TERRAIN_CATEGORY_TABLE = _build_terrain_category_table()


def _build_defense_norm_table() -> dict[int, float]:
    """terrain_id → defense_stars / 4 (matches ``TerrainInfo.defense`` scale 0–4)."""
    from engine.terrain import TERRAIN_TABLE

    return {tid: float(info.defense) / 4.0 for tid, info in TERRAIN_TABLE.items()}


_DEFENSE_NORM_TABLE = _build_defense_norm_table()


def _defense_norm_for_tid(tid: int) -> float:
    v = _DEFENSE_NORM_TABLE.get(tid)
    if v is not None:
        return v
    return float(get_terrain(tid).defense) / 4.0


def _refresh_map_static_caches_if_needed(md: MapData, *, force: bool = False) -> None:
    """
    Ensure ``md`` has Phase-3b terrain one-hot plus defense-stars grid, sharing one
    terrain-id snapshot for invalidation (pipe breaks / terrain mutations).
    """
    H_md = min(md.height, GRID_SIZE)
    W_md = min(md.width, GRID_SIZE)
    tids_live = np.empty((H_md, W_md), dtype=np.int32)
    for r in range(H_md):
        tids_live[r, :] = md.terrain[r][:W_md]

    terrain_block = getattr(md, "_encoded_terrain_channels", None)
    defense_block = getattr(md, "_encoded_defense_stars", None)
    terrain_snap = getattr(md, "_encoded_terrain_tid_snapshot", None)
    cache_ok = (
        not force
        and terrain_block is not None
        and defense_block is not None
        and terrain_snap is not None
        and terrain_block.shape == (GRID_SIZE, GRID_SIZE, N_TERRAIN_CHANNELS)
        and defense_block.shape == (GRID_SIZE, GRID_SIZE)
        and terrain_snap.shape == (H_md, W_md)
        and np.array_equal(terrain_snap, tids_live)
    )
    if cache_ok:
        return

    terrain_block = np.zeros(
        (GRID_SIZE, GRID_SIZE, N_TERRAIN_CHANNELS), dtype=np.float32
    )
    defense_block = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    cython_terrain_filled = False
    try:
        from ._encoder_cython import fill_terrain_channels

        fill_terrain_channels(
            terrain_block,
            tids_live,
            _TERRAIN_CATEGORY_TABLE,
            _DEFENSE_NORM_TABLE,
        )
        cython_terrain_filled = True
    except ImportError:
        for r in range(H_md):
            for c in range(W_md):
                tid = int(tids_live[r, c])
                cat = _get_terrain_category(tid)
                terrain_block[r, c, cat] = 1.0
                defense_block[r, c] = _defense_norm_for_tid(tid)

    # Cython one-hot only touches terrain_block; keep defense in lockstep with Python.
    if cython_terrain_filled:
        for r in range(H_md):
            for c in range(W_md):
                tid = int(tids_live[r, c])
                defense_block[r, c] = _defense_norm_for_tid(tid)
    
    try:
        md._encoded_terrain_channels = terrain_block
        md._encoded_defense_stars = defense_block
        md._encoded_terrain_tid_snapshot = tids_live.copy()
    except (AttributeError, TypeError):
        pass


def warm_map_static_encoder_cache(map_data: MapData) -> None:
    """Pre-fill terrain + defense-star caches on ``MapData`` (no ``GameState`` required).

    Call after ``load_map`` so the first ``encode_state`` skips cache-miss work.
    Idempotent per terrain grid; use ``_refresh_map_static_caches_if_needed(..., force=True)``
    after in-place terrain mutation on the same ``MapData`` instance.
    """
    _refresh_map_static_caches_if_needed(map_data, force=False)


def _get_terrain_category(terrain_id: int) -> int:
    """Map a raw terrain tile id to a TERRAIN_CATEGORIES index."""
    cached = _TERRAIN_CATEGORY_TABLE.get(terrain_id)
    if cached is not None:
        return cached
    return _compute_terrain_category_uncached(terrain_id)


# Cython bridge setup
CYTHON_AVAILABLE = False
try:
    from rl import _encoder_cython
    CYTHON_AVAILABLE = True
except ImportError:
    # Fallback to pure Python implementation
    pass


def encode_state(
    state: GameState,
    *,
    observer: int = 0,
    belief: "BeliefState | None" = None,
    out_spatial: np.ndarray | None = None,
    out_scalars: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a GameState into spatial and scalar tensors.

    Args:
        state:    GameState to encode.
        observer: Seat whose perspective this observation serves (0 or 1).
                  Own units are encoded with exact HP (``hp_lo == hp_hi``);
                  enemy units read from ``belief`` when supplied.
        belief:   Optional ``BeliefState`` keyed to ``observer``. When None,
                  both HP channels collapse to ``unit.hp / 100`` for all
                  units (legacy / debug path; leaks exact enemy HP — do not
                  use for bot runtime).
        out_spatial, out_scalars:
                  Optional pre-allocated float32 arrays with shapes
                  (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS) and (N_SCALARS,).
                  When provided, results are written in place (spatial is fully
                  zeroed before encoding). When omitted, fresh arrays are allocated.

    Returns:
        spatial: (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS) float32
        scalars: (N_SCALARS,) float32
    """
    H = min(state.map_data.height, GRID_SIZE)
    W = min(state.map_data.width, GRID_SIZE)
    
    # Feature flag for Cython
    USE_CYTHON = os.environ.get("AWBW_USE_CYTHON_ENCODER", "1") == "1" and CYTHON_AVAILABLE
    
    # Memory-efficient channel encoding
    # Use float16 for spatial data to reduce memory footprint
    spatial_dtype = np.float16 if os.environ.get("AWBW_USE_FLOAT16", "0") == "1" else np.float32
    
    # Use more compact representation for unit channels
    UNIT_CHANNEL_COMPRESSION = int(os.environ.get("AWBW_UNIT_CHANNEL_COMPRESSION", "1"))

    if out_spatial is not None:
        spatial = out_spatial
        spatial.fill(0.0)
    else:
        spatial = np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    hp_lo_ch = N_UNIT_CHANNELS
    hp_hi_ch = N_UNIT_CHANNELS + 1

    # ── Terrain channels ─────────────────────────────────────────────────────
    # Phase 3b: one-hot terrain block is a pure function of map_data.terrain.
    # Cache per MapData; invalidate when the terrain grid changes (capture and
    # pipe-seam breaks mutate terrain in place — see engine/game.py).
    terrain_ch_offset = N_UNIT_CHANNELS + N_HP_CHANNELS
    md = state.map_data
    _refresh_map_static_caches_if_needed(md, force=False)
    terrain_block = getattr(md, "_encoded_terrain_channels", None)
    defense_block = getattr(md, "_encoded_defense_stars", None)
    if terrain_block is None:
        terrain_block = np.zeros(
            (GRID_SIZE, GRID_SIZE, N_TERRAIN_CHANNELS), dtype=np.float32
        )
    if defense_block is None:
        defense_block = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    np.copyto(
        spatial[:, :, terrain_ch_offset : terrain_ch_offset + N_TERRAIN_CHANNELS],
        terrain_block,
    )

    # ── Property ownership + capture progress + neutral income (single pass) ─
    prop_ch_offset = N_UNIT_CHANNELS + N_HP_CHANNELS + N_TERRAIN_CHANNELS
    cap_ch0 = N_UNIT_CHANNELS + N_HP_CHANNELS + N_TERRAIN_CHANNELS + N_PROPERTY_CHANNELS
    cap_ch1 = cap_ch0 + 1
    neutral_inc_ch = cap_ch1 + 1
    
    # Use Cython for property encoding if available
    if USE_CYTHON:
        from rl import _encoder_cython
        _encoder_cython.encode_properties(
            spatial,
            state,
            state.properties,
            observer,
            prop_ch_offset,
            cap_ch0,
            cap_ch1,
            neutral_inc_ch,
        )
    else:
        for prop in state.properties:
            r, c = prop.row, prop.col
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
                continue

            if prop.is_hq or prop.is_lab:
                ptype = _PROP_TYPE_HQ_LAB
            elif prop.is_base:
                ptype = _PROP_TYPE_BASE
            elif prop.is_airport:
                ptype = _PROP_TYPE_AIRPORT
            elif prop.is_port:
                ptype = _PROP_TYPE_PORT
            else:
                ptype = _PROP_TYPE_CITY

            if prop.owner is None:
                ownership = 0
            elif prop.owner == observer:
                ownership = 1  # me
            else:
                ownership = 2  # enemy

            spatial[r, c, prop_ch_offset + ptype * 3 + ownership] = 1.0

            if prop.owner is None and not prop.is_comm_tower and not prop.is_lab:
                spatial[r, c, neutral_inc_ch] = 1.0

            if prop.capture_points < 20:
                occ = state.get_unit_at(r, c)
                if occ is not None:
                    prog = (20 - prop.capture_points) / 20.0
                    if occ.player == observer:
                        spatial[r, c, cap_ch0] = max(spatial[r, c, cap_ch0], prog)
                    elif occ.player != observer:
                        spatial[r, c, cap_ch1] = max(spatial[r, c, cap_ch1], prog)

    # ── Unit presence + HP belief channels (me then enemy) ──────────────────
    hp_lo_ch = N_UNIT_CHANNELS
    hp_hi_ch = N_UNIT_CHANNELS + 1
    
    # Use Cython for unit encoding if available
    if USE_CYTHON:
        from . import _encoder_cython
        # BeliefState stores UnitBelief in _beliefs; expose as id -> belief for Cython.
        belief_dict = (
            {b.unit_id: b for b in belief.all()} if belief is not None else {}
        )
        
        # Encode observer units
        _encoder_cython.encode_units(
            spatial,
            state.units[observer],
            observer,
            belief_dict,
            hp_lo_ch,
            hp_hi_ch,
            _N_UNIT_TYPES,
            0,
        )
        # Encode enemy units
        _encoder_cython.encode_units(
            spatial,
            state.units[1 - observer],
            observer,
            belief_dict,
            hp_lo_ch,
            hp_hi_ch,
            _N_UNIT_TYPES,
            _N_UNIT_TYPES,
        )
    else:
        for idx, player in enumerate((observer, 1 - observer)):
            player_ch_offset = _N_UNIT_TYPES * idx  # 0 = me, 14 = enemy
            for unit in state.units[player]:
                r, c = unit.pos
                if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
                    continue
                ch = UNIT_TO_CHANNEL[unit.unit_type] + player_ch_offset
                spatial[r, c, ch] = 1.0

                # HP belief interval (own units collapse to exact).
                if unit.player == observer or belief is None:
                    hp_norm = unit.hp / 100.0
                    hp_lo = hp_hi = hp_norm
                else:
                    b = belief.get(unit.unit_id)
                    if b is None:
                        # Enemy unit with no belief entry — fall back to the full
                        # visible bucket (happens for units spawned by CO powers
                        # that bypass the built event; defensive).
                        bucket = (unit.hp + 9) // 10
                        hp_lo = max(0, bucket * 10 - 9) / 100.0
                        hp_hi = max(0, bucket * 10) / 100.0
                    else:
                        hp_lo = b.hp_min / 100.0
                        hp_hi = b.hp_max / 100.0
                # Latest unit written wins on stacked tiles — acceptable for
                # non-transport boards; transports stack own+loaded which is a
                # known rendering ambiguity, unchanged from the pre-belief layout.
                spatial[r, c, hp_lo_ch] = hp_lo
                spatial[r, c, hp_hi_ch] = hp_hi
                
    # ── Terrain cache building optimization ─────────────────────────────────
    if USE_CYTHON and terrain_block is None:
        from rl import _encoder_cython
        H_md = min(md.height, GRID_SIZE)
        W_md = min(md.width, GRID_SIZE)
        tids_live = np.empty((H_md, W_md), dtype=np.int32)
        for r in range(H_md):
            tids_live[r, :] = md.terrain[r][:W_md]
        
        terrain_block = np.zeros(
            (GRID_SIZE, GRID_SIZE, N_TERRAIN_CHANNELS), dtype=np.float32
        )
        defense_block = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        _encoder_cython.build_terrain_cache(
            tids_live,
            terrain_block,
            defense_block,
            H_md,
            W_md
        )
        
        try:
            md._encoded_terrain_channels = terrain_block
            md._encoded_defense_stars = defense_block
            md._encoded_terrain_tid_snapshot = tids_live.copy()
        except (AttributeError, TypeError):
            pass

    # ── Influence planes (63..68) ────────────────────────────────────────────
    infl_base = N_INFLUENCE_CHANNEL_BASE
    t_me, t_en, r_me, r_en, c_me, c_en = compute_influence_planes(
        state, me=observer, grid=GRID_SIZE
    )
    spatial[:, :, infl_base + 0] = t_me
    spatial[:, :, infl_base + 1] = t_en
    spatial[:, :, infl_base + 2] = r_me
    spatial[:, :, infl_base + 3] = r_en
    spatial[:, :, infl_base + 4] = c_me
    spatial[:, :, infl_base + 5] = c_en

    # ── Map-static defense stars (channel 69) ────────────────────────────────
    np.copyto(spatial[:, :, N_DEFENSE_STARS_CHANNEL], defense_block)

    # ── Scalar features (ego-centric) ───────────────────────────────────────
    enemy = 1 - int(observer)
    co_me = state.co_states[observer]
    co_en = state.co_states[enemy]

    # Normalise power bar against the current SCOP threshold (grows with power_uses)
    def _norm_power(co_state) -> float:
        denom = co_state._scop_threshold
        if denom <= 0 or denom >= 10**11:
            return 0.0
        return min(1.0, float(co_state.power_bar) / denom)

    weather = getattr(state, "weather", "clear")
    my_turn = 1.0 if int(state.active_player) == int(observer) else 0.0
    if out_scalars is not None:
        scalars = out_scalars
        scalars[0] = state.funds[observer] / 50_000.0
        scalars[1] = state.funds[enemy] / 50_000.0
        scalars[2] = _norm_power(co_me)
        scalars[3] = _norm_power(co_en)
        scalars[4] = float(co_me.cop_active)
        scalars[5] = float(co_me.scop_active)
        scalars[6] = float(co_en.cop_active)
        scalars[7] = float(co_en.scop_active)
        scalars[8] = state.turn / max(1, int(getattr(state, "max_turns", MAX_TURNS)))
        scalars[9] = my_turn
        scalars[10] = co_me.co_id / 30.0
        scalars[11] = co_en.co_id / 30.0
        scalars[12] = _TIER_MAP.get(state.tier_name, 0.5)
        scalars[13] = 1.0 if weather == "rain" else 0.0
        scalars[14] = 1.0 if weather == "snow" else 0.0
        scalars[15] = getattr(state, "co_weather_segments_remaining", 0) / 2.0
        scalars[16] = _income_share_for(state, observer)
    else:
        scalars = np.array(
            [
                state.funds[observer] / 50_000.0,
                state.funds[enemy] / 50_000.0,
                _norm_power(co_me),
                _norm_power(co_en),
                float(co_me.cop_active),
                float(co_me.scop_active),
                float(co_en.cop_active),
                float(co_en.scop_active),
                state.turn / max(1, int(getattr(state, "max_turns", MAX_TURNS))),
                my_turn,
                co_me.co_id / 30.0,
                co_en.co_id / 30.0,
                _TIER_MAP.get(state.tier_name, 0.5),
                1.0 if weather == "rain" else 0.0,
                1.0 if weather == "snow" else 0.0,
                getattr(state, "co_weather_segments_remaining", 0) / 2.0,
                _income_share_for(state, observer),
            ],
            dtype=np.float32,
        )

    return spatial, scalars


def _income_share_for(state: GameState, observer: int) -> float:
    n_income = sum(
        1 for p in state.properties if not p.is_comm_tower and not p.is_lab
    )
    if n_income <= 0:
        return 0.0
    return float(state.count_income_properties(observer)) / float(n_income)
