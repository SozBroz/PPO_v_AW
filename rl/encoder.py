"""
Encodes a GameState into a numpy observation tensor.

Output shape: (H, W, N_CHANNELS) where H=W=30 (padded), N_CHANNELS is:
  - 28 unit channels (14 unit types × 2 players)
  - 2 HP channels (hp_lo_ch, hp_hi_ch) — observer-aware belief interval
  - 15 terrain channels (one-hot, main terrain categories)
  - 15 property channels (5 property types × 3 states: neutral/p0/p1)
  - 2 capture-progress channels (P0 / P1 unit reducing ``capture_points`` on tile)
  - 1 neutral contestable income-property mask (owner None, not lab/comm)
  Total: 28 + 2 + 15 + 15 + 3 = 63 spatial channels

Plus scalar features (appended after CNN flatten):
  - funds[0], funds[1] (normalized by 50000)
  - co_power_bar[0], co_power_bar[1] (normalized by max)
  - cop_active[0], cop_active[1] (binary)
  - scop_active[0], scop_active[1] (binary)
  - turn (normalized by MAX_TURNS)
  - active_player (0 or 1)
  - p0_co_id, p1_co_id (normalized by 30)
  - tier (normalized: T1=0.25, T2=0.5, T3=0.75, T4=1.0)
  - weather_rain  (binary: 1.0 if rain active, else 0.0)
  - weather_snow  (binary: 1.0 if snow active, else 0.0)
  - weather_turns (co_weather_segments_remaining / 2.0; 0 = clear/default)
  - p0_income_share (income properties owned by P0 / total income tiles on map)
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
``hp_lo`` / ``hp_hi``. See ``docs/hp_belief.md`` for the observation change.
"""
import numpy as np
from engine.game import GameState, MAX_TURNS
from engine.unit import UnitType
from engine.terrain import get_terrain
from engine.belief import BeliefState

GRID_SIZE = 30
N_UNIT_CHANNELS = 28       # 14 unit types × 2 players
N_HP_CHANNELS = 2          # hp_lo, hp_hi (belief interval)
N_TERRAIN_CHANNELS = 15
N_PROPERTY_CHANNELS = 15   # 5 property types × 3 ownership states
N_CAPTURE_EXTRA_CHANNELS = 3  # p0 progress, p1 progress, neutral-income mask
N_SPATIAL_CHANNELS = (
    N_UNIT_CHANNELS + N_HP_CHANNELS + N_TERRAIN_CHANNELS
    + N_PROPERTY_CHANNELS + N_CAPTURE_EXTRA_CHANNELS
)  # 63
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


def _get_terrain_category(terrain_id: int) -> int:
    """Map a raw terrain tile id to a TERRAIN_CATEGORIES index."""
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


def encode_state(
    state: GameState,
    *,
    observer: int = 0,
    belief: "BeliefState | None" = None,
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

    Returns:
        spatial: (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS) float32
        scalars: (N_SCALARS,) float32
    """
    H = min(state.map_data.height, GRID_SIZE)
    W = min(state.map_data.width, GRID_SIZE)

    spatial = np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)

    hp_lo_ch = N_UNIT_CHANNELS
    hp_hi_ch = N_UNIT_CHANNELS + 1

    # ── Terrain channels ─────────────────────────────────────────────────────
    terrain_ch_offset = N_UNIT_CHANNELS + N_HP_CHANNELS
    for r in range(H):
        for c in range(W):
            tid = state.map_data.terrain[r][c]
            cat = _get_terrain_category(tid)
            spatial[r, c, terrain_ch_offset + cat] = 1.0

    # ── Property ownership channels ──────────────────────────────────────────
    prop_ch_offset = N_UNIT_CHANNELS + N_HP_CHANNELS + N_TERRAIN_CHANNELS
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

        # ownership: 0 = neutral, 1 = player 0, 2 = player 1
        if prop.owner is None:
            ownership = 0
        elif prop.owner == 0:
            ownership = 1
        else:
            ownership = 2

        spatial[r, c, prop_ch_offset + ptype * 3 + ownership] = 1.0

    # ── Capture progress + neutral income mask (after property ownership) ─────
    cap_ch0 = N_UNIT_CHANNELS + N_HP_CHANNELS + N_TERRAIN_CHANNELS + N_PROPERTY_CHANNELS
    cap_ch1 = cap_ch0 + 1
    neutral_inc_ch = cap_ch1 + 1
    for prop in state.properties:
        r, c = prop.row, prop.col
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            continue
        if prop.owner is None and not prop.is_comm_tower and not prop.is_lab:
            spatial[r, c, neutral_inc_ch] = 1.0
        if prop.capture_points < 20:
            occ = state.get_unit_at(r, c)
            if occ is not None:
                prog = (20 - prop.capture_points) / 20.0
                if occ.player == 0:
                    spatial[r, c, cap_ch0] = max(spatial[r, c, cap_ch0], prog)
                elif occ.player == 1:
                    spatial[r, c, cap_ch1] = max(spatial[r, c, cap_ch1], prog)

    # ── Unit presence + HP belief channels ───────────────────────────────────
    for player in (0, 1):
        player_ch_offset = _N_UNIT_TYPES * player  # 0 for p0, 14 for p1
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

    # ── Scalar features ───────────────────────────────────────────────────────
    co0 = state.co_states[0]
    co1 = state.co_states[1]

    # Normalise power bar against the current SCOP threshold (grows with power_uses)
    def _norm_power(co_state) -> float:
        denom = co_state._scop_threshold
        if denom <= 0 or denom >= 10**11:
            return 0.0
        return min(1.0, float(co_state.power_bar) / denom)

    weather = getattr(state, "weather", "clear")
    scalars = np.array(
        [
            state.funds[0] / 50_000.0,
            state.funds[1] / 50_000.0,
            _norm_power(co0),
            _norm_power(co1),
            float(co0.cop_active),
            float(co0.scop_active),
            float(co1.cop_active),
            float(co1.scop_active),
            state.turn / MAX_TURNS,
            float(state.active_player),
            co0.co_id / 30.0,
            co1.co_id / 30.0,
            _TIER_MAP.get(state.tier_name, 0.5),
            # Weather scalars (indices 13-15)
            1.0 if weather == "rain" else 0.0,
            1.0 if weather == "snow" else 0.0,
            getattr(state, "co_weather_segments_remaining", 0) / 2.0,
            # Share of contestable income tiles owned by P0 (labs/comm excluded).
            _p0_income_share(state),
        ],
        dtype=np.float32,
    )

    return spatial, scalars


def _p0_income_share(state: GameState) -> float:
    n_income = sum(
        1 for p in state.properties if not p.is_comm_tower and not p.is_lab
    )
    if n_income <= 0:
        return 0.0
    return float(state.count_income_properties(0)) / float(n_income)
