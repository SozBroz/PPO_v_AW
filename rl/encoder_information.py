"""
Encoder round-trip / information-loss audit for ``rl.encoder.encode_state``.

``encode_state`` is a feature map for the policy, not a lossless codec of
:class:`engine.game.GameState`.  This module decodes the spatial + scalar
tensors to a :class:`RecoveredObservation` and compares to the true state
via :class:`InformationLossReport` (categorized loss, not full equality).

See ``rl.encoder`` module docstring for channel layout and scalar semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.game import GameState, MAX_TURNS
from engine.threat import compute_influence_planes
from engine.unit import Unit, UnitType

from rl import encoder as _enc

GRID_SIZE = _enc.GRID_SIZE
N_UNIT_CHANNELS = _enc.N_UNIT_CHANNELS
N_HP_CHANNELS = _enc.N_HP_CHANNELS
N_TERRAIN_CHANNELS = _enc.N_TERRAIN_CHANNELS
N_PROPERTY_CHANNELS = _enc.N_PROPERTY_CHANNELS
N_SPATIAL_CHANNELS = _enc.N_SPATIAL_CHANNELS
N_INFLUENCE_CHANNEL_BASE = _enc.N_INFLUENCE_CHANNEL_BASE
N_DEFENSE_STARS_CHANNEL = _enc.N_DEFENSE_STARS_CHANNEL
N_UNIT_MODIFIER_CHANNEL_BASE = _enc.N_UNIT_MODIFIER_CHANNEL_BASE
N_UNIT_MODIFIER_CHANNELS = _enc.N_UNIT_MODIFIER_CHANNELS
N_SCALARS = _enc.N_SCALARS
N_INFLUENCE_CHANNELS = 6

# GameState (and friends) fields not represented in the observation tensor.
NOT_IN_OBSERVATION: frozenset[str] = frozenset(
    {
        "map_data",  # partial: only categories + static defense, not full tid
        "units",  # only one unit's type+side per cell in spatial
        "properties",  # per-cell subset only
        "action_stage",
        "selected_unit",
        "selected_move_pos",
        "done",
        "winner",
        "win_reason",
        "game_log",
        "full_trace",
        "luck_rng",
        "gold_spent",
        "losses_hp",
        "losses_units",
        "next_unit_id",
        "seam_hp",
    }
)

SCALAR_NAMES: tuple[str, ...] = (
    "funds_me",
    "funds_enemy",
    "power_bar_me",
    "cop_stars_me",
    "scop_stars_me",
    "cop_active_me",
    "scop_active_me",
    "power_bar_enemy",
    "cop_stars_enemy",
    "scop_stars_enemy",
    "cop_active_enemy",
    "scop_active_enemy",
    "turn_norm",
    "my_turn",
    "co_id_me",
    "co_id_enemy",
    "weather_rain",
    "weather_snow",
    "co_weather_segments_norm",
    "income_share_me",
)


@dataclass
class RecoveredScalars:
    """Inferred values from the 20-d scalar vector (``encode_state`` order)."""

    raw: np.ndarray
    funds_me: float
    funds_enemy: float
    power_bar_me: float
    cop_stars_me: int
    scop_stars_me: int
    cop_active_me: float
    scop_active_me: float
    power_bar_enemy: float
    cop_stars_enemy: int
    scop_stars_enemy: int
    cop_active_enemy: float
    scop_active_enemy: float
    turn_index: float
    my_turn: float
    co_id_me: int
    co_id_enemy: int
    weather_rain: float
    weather_snow: float
    co_weather_segments: float


def _as_float32(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float16 or a.dtype == np.float64:
        return np.asarray(a, dtype=np.float32)
    return a


def decode_scalars(scalars: np.ndarray) -> RecoveredScalars:
    s = np.asarray(scalars).reshape(-1)
    if s.shape[0] != N_SCALARS:
        raise ValueError(f"expected {N_SCALARS} scalars, got {s.shape[0]}")
    return RecoveredScalars(
        raw=s.astype(np.float64),
        funds_me=float(s[0]) * 50_000.0,
        funds_enemy=float(s[1]) * 50_000.0,
        power_bar_me=float(s[2]) * 50_000.0,
        cop_stars_me=int(round(float(s[3]) * 10.0)),
        scop_stars_me=int(round(float(s[4]) * 10.0)),
        cop_active_me=float(s[5]),
        scop_active_me=float(s[6]),
        power_bar_enemy=float(s[7]) * 50_000.0,
        cop_stars_enemy=int(round(float(s[8]) * 10.0)),
        scop_stars_enemy=int(round(float(s[9]) * 10.0)),
        cop_active_enemy=float(s[10]),
        scop_active_enemy=float(s[11]),
        turn_index=float(s[12]),
        my_turn=float(s[13]),
        co_id_me=int(round(float(s[14]) * 30.0)),
        co_id_enemy=int(round(float(s[15]) * 30.0)),
        weather_rain=float(s[16]),
        weather_snow=float(s[17]),
        co_weather_segments=float(s[18]) * 2.0,
    )


def decode_scalars_with_max_turns(scalars: np.ndarray, max_turns: int) -> RecoveredScalars:
    rs = decode_scalars(scalars)
    s = np.asarray(scalars).reshape(-1)
    mt = max(1, int(max_turns))
    return RecoveredScalars(
        raw=rs.raw,
        funds_me=rs.funds_me,
        funds_enemy=rs.funds_enemy,
        power_bar_me=rs.power_bar_me,
        cop_stars_me=rs.cop_stars_me,
        scop_stars_me=rs.scop_stars_me,
        cop_active_me=rs.cop_active_me,
        scop_active_me=rs.scop_active_me,
        power_bar_enemy=rs.power_bar_enemy,
        cop_stars_enemy=rs.cop_stars_enemy,
        scop_stars_enemy=rs.scop_stars_enemy,
        cop_active_enemy=rs.cop_active_enemy,
        scop_active_enemy=rs.scop_active_enemy,
        turn_index=float(s[12]) * mt,
        my_turn=rs.my_turn,
        co_id_me=rs.co_id_me,
        co_id_enemy=rs.co_id_enemy,
        weather_rain=rs.weather_rain,
        weather_snow=rs.weather_snow,
        co_weather_segments=rs.co_weather_segments,
    )


def decode_observation_maximal(
    spatial: np.ndarray,
    scalars: np.ndarray,
    *,
    observer: int,
    height: int,
    width: int,
    max_turns: int = MAX_TURNS,
) -> RecoveredObservation:
    sp = _as_float32(spatial)
    if sp.shape != (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS):
        raise ValueError(
            f"spatial shape {sp.shape} != "
            f"({GRID_SIZE}, {GRID_SIZE}, {N_SPATIAL_CHANNELS})"
        )
    H = min(int(height), GRID_SIZE)
    W = min(int(width), GRID_SIZE)
    terrain = np.full((H, W), -1, dtype=np.int32)
    defense = np.zeros((H, W), dtype=np.float64)
    unit_ch = np.full((H, W), -1, dtype=np.int32)
    hp_lo = np.zeros((H, W), dtype=np.float64)
    hp_hi = np.zeros((H, W), dtype=np.float64)
    prop = np.zeros((H, W, N_PROPERTY_CHANNELS), dtype=np.float64)
    cap0 = np.zeros((H, W), dtype=np.float64)
    cap1 = np.zeros((H, W), dtype=np.float64)
    neutral = np.zeros((H, W), dtype=np.float64)
    infl = np.zeros((H, W, N_INFLUENCE_CHANNELS), dtype=np.float64)
    unit_mods = np.zeros((H, W, N_UNIT_MODIFIER_CHANNELS), dtype=np.float64)

    t_off = N_UNIT_CHANNELS + N_HP_CHANNELS
    p_off = t_off + N_TERRAIN_CHANNELS
    cap0_i = p_off + N_PROPERTY_CHANNELS
    cap1_i = cap0_i + 1
    ne_i = cap1_i + 1
    infl0 = N_INFLUENCE_CHANNEL_BASE

    for r in range(H):
        for c in range(W):
            ter = sp[r, c, t_off : t_off + N_TERRAIN_CHANNELS]
            ssum = float(ter.sum())
            if ssum > 0.1:
                terrain[r, c] = int(np.argmax(ter))
            else:
                terrain[r, c] = -1
            defense[r, c] = float(sp[r, c, N_DEFENSE_STARS_CHANNEL])
            uvec = sp[r, c, :N_UNIT_CHANNELS]
            mask = uvec > 0.5
            nact = int(mask.sum())
            if nact == 0:
                unit_ch[r, c] = -1
            elif nact == 1:
                unit_ch[r, c] = int(np.argmax(uvec))
            else:
                unit_ch[r, c] = -2
            hp_lo[r, c] = float(sp[r, c, N_UNIT_CHANNELS])
            hp_hi[r, c] = float(sp[r, c, N_UNIT_CHANNELS + 1])
            prop[r, c, :] = sp[r, c, p_off : p_off + N_PROPERTY_CHANNELS]
            cap0[r, c] = float(sp[r, c, cap0_i])
            cap1[r, c] = float(sp[r, c, cap1_i])
            neutral[r, c] = float(sp[r, c, ne_i])
            infl[r, c, :] = sp[r, c, infl0 : infl0 + N_INFLUENCE_CHANNELS]
            unit_mods[r, c, :] = sp[
                r,
                c,
                N_UNIT_MODIFIER_CHANNEL_BASE : N_UNIT_MODIFIER_CHANNEL_BASE
                + N_UNIT_MODIFIER_CHANNELS,
            ]

    rs = decode_scalars_with_max_turns(scalars, max_turns)
    return RecoveredObservation(
        height=H,
        width=W,
        terrain_category=terrain,
        defense_norm=defense.astype(np.float32),
        unit_channel=unit_ch,
        hp_lo=hp_lo.astype(np.float32),
        hp_hi=hp_hi.astype(np.float32),
        property_hot=prop.astype(np.float32),
        cap_me=cap0.astype(np.float32),
        cap_enemy=cap1.astype(np.float32),
        neutral_income=neutral.astype(np.float32),
        influence=infl.astype(np.float32),
        unit_modifiers=unit_mods.astype(np.float32),
        scalars=rs,
    )


def _unit_at_cell(state: GameState, r: int, c: int) -> list[Unit]:
    out: list[Unit] = []
    for p in (0, 1):
        for u in state.units[p]:
            if u.pos == (r, c):
                out.append(u)
    return out


def _iter_true_stack_excess(state: GameState, h: int, w: int) -> int:
    acc = 0
    for r in range(h):
        for c in range(w):
            n = len(_unit_at_cell(state, r, c))
            if n > 1:
                acc += n - 1
    return acc


@dataclass
class InformationLossReport:
    not_represented_gamestate_fields: frozenset[str] = field(
        default_factory=lambda: NOT_IN_OBSERVATION
    )
    stacked_units_unrepresented: int = 0
    """Sum over cells of max(0, true_units_on_cell - 1)."""
    terrain_category_mismatch_count: int = 0
    terrain_mismatch_samples: list[tuple[int, int, int, int]] = field(default_factory=list)
    """(r, c, true_cat, decoded_cat) up to 8 samples."""
    scalar_abs_errors: dict[str, float] = field(default_factory=dict)
    max_scalar_abs_error: float = 0.0
    influence_mae: float = 0.0
    influence_max_abs: float = 0.0
    belief_enemy_non_point_hp_cells: int = 0
    """Decoded hp_lo/hp_hi differ for cells with enemy in belief mode (if belief used)."""
    enemy_hp_interval_width_gt_zero: int = 0
    """Count enemy-occupied cells where decoded interval width > eps (belief hides exact)."""

    def human_readable_summary(self) -> str:
        lines = [
            f"not_in_observation (static, {len(self.not_represented_gamestate_fields)} fields): "
            f"see NOT_IN_OBSERVATION in rl/encoder_information.py",
            f"stacked_units_unrepresented: {self.stacked_units_unrepresented}",
            f"terrain_category_mismatch_count: {self.terrain_category_mismatch_count}",
            f"influence_mae: {self.influence_mae:.6g}  influence_max_abs: {self.influence_max_abs:.6g}",
            f"max_scalar_abs_error: {self.max_scalar_abs_error:.6g}",
        ]
        if self.scalar_abs_errors:
            se = self.scalar_abs_errors
            lines.append(
                "scalar_abs_errors: "
                + ", ".join(f"{k}={v:.4g}" for k, v in sorted(se.items())[:12])
            )
        if self.terrain_mismatch_samples:
            lines.append(f"terrain_mismatch_samples: {self.terrain_mismatch_samples[:4]}")
        if self.belief_enemy_non_point_hp_cells or self.enemy_hp_interval_width_gt_zero:
            lines.append(
                f"belief: enemy non-point HP cells (decoded)={self.enemy_hp_interval_width_gt_zero} "
                f"(heuristic non_point={self.belief_enemy_non_point_hp_cells})"
            )
        return "\n".join(lines)


def _scalar_errors(true_state: GameState, observer: int, scalars: np.ndarray) -> dict[str, float]:
    s = np.asarray(scalars, dtype=np.float64).reshape(-1)
    enemy = 1 - int(observer)
    co_me = true_state.co_states[observer]
    co_en = true_state.co_states[enemy]
    max_t = max(1, int(getattr(true_state, "max_turns", MAX_TURNS)))

    weather = getattr(true_state, "weather", "clear")
    n_income = sum(
        1 for p in true_state.properties if not p.is_comm_tower and not p.is_lab
    )
    if n_income <= 0:
        share_t = 0.0
    else:
        share_t = float(true_state.count_income_properties(observer)) / float(
            n_income
        )

    # Expected encoded scalars (new 20-scalar layout)
    def _stars_norm(stars) -> float:
        if stars is None:
            return 0.0
        return min(10.0, float(stars)) / 10.0

    exp = [
        true_state.funds[observer] / 50_000.0,
        true_state.funds[enemy] / 50_000.0,
        co_me.power_bar / 50_000.0,           # raw power bar
        _stars_norm(co_me.cop_stars),            # COP stars (0 for Von Bolt)
        _stars_norm(co_me.scop_stars),           # SCOP stars
        float(co_me.cop_active),
        float(co_me.scop_active),
        co_en.power_bar / 50_000.0,
        _stars_norm(co_en.cop_stars),
        _stars_norm(co_en.scop_stars),
        float(co_en.cop_active),
        float(co_en.scop_active),
        true_state.turn / max_t,
        1.0 if int(true_state.active_player) == int(observer) else 0.0,
        co_me.co_id / 30.0,
        co_en.co_id / 30.0,
        1.0 if weather == "rain" else 0.0,
        1.0 if weather == "snow" else 0.0,
        getattr(true_state, "co_weather_segments_remaining", 0) / 2.0,
        share_t,
    ]
    out: dict[str, float] = {}
    for i, name in enumerate(SCALAR_NAMES):
        e = exp[i] if i < len(exp) else 0.0
        v = s[i] if i < s.shape[0] else 0.0
        out[name] = abs(float(e) - float(v))
    return out


def information_loss(
    true_state: GameState,
    spatial: np.ndarray,
    scalars: np.ndarray,
    observer: int,
    *,
    belief_was_used: bool = False,
) -> InformationLossReport:
    """
    Compare encoded tensors to *true* ``true_state`` and tabulate information loss.

    ``belief_was_used`` should be True iff ``encode_state(..., belief=...)`` was
    called with a non-None belief so HP-interval interpretation is expected for
    enemies.
    """
    md = true_state.map_data
    h = min(md.height, GRID_SIZE)
    w = min(md.width, GRID_SIZE)
    max_t = int(getattr(true_state, "max_turns", MAX_TURNS))
    rec = decode_observation_maximal(
        spatial, scalars, observer=observer, height=h, width=w, max_turns=max_t
    )

    stack_loss = _iter_true_stack_excess(true_state, h, w)

    t_mis = 0
    t_samples: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            tid = int(md.terrain[r][c])
            tc = _enc._get_terrain_category(tid)
            dc = int(rec.terrain_category[r, c])
            if dc < 0:
                t_mis += 1
                if len(t_samples) < 8:
                    t_samples.append((r, c, tc, dc))
            elif tc != dc:
                t_mis += 1
                if len(t_samples) < 8:
                    t_samples.append((r, c, tc, dc))

    infl = rec.influence
    t_me, t_en, r_me, r_en, c_me, c_en = compute_influence_planes(
        true_state, me=observer, grid=GRID_SIZE
    )[:6]
    planes = [t_me, t_en, r_me, r_en, c_me, c_en]
    all_diffs: list[np.ndarray] = []
    for i, p in enumerate(planes):
        a = _as_float32(p)[:h, :w]
        b = _as_float32(infl[:, :, i])
        b = b[: a.shape[0], : a.shape[1]]
        all_diffs.append(np.abs(a - b))
    if all_diffs:
        cat = np.stack(all_diffs, axis=-1)
        mae = float(np.mean(cat))
        mx = float(np.max(cat))
    else:
        mae = 0.0
        mx = 0.0

    s_err = _scalar_errors(true_state, observer, scalars)
    max_se = max(s_err.values()) if s_err else 0.0

    # HP width on enemy cells (decoder path)
    enemy_hp_w = 0
    non_point = 0
    for r in range(h):
        for c in range(w):
            lo = float(rec.hp_lo[r, c])
            hi = float(rec.hp_hi[r, c])
            wdt = abs(hi - lo)
            if wdt > 0.01:
                enemy_hp_w += 1
            ulist = _unit_at_cell(true_state, r, c)
            for u in ulist:
                if u.player != observer and wdt > 0.01:
                    non_point += 1
                    break

    return InformationLossReport(
        stacked_units_unrepresented=stack_loss,
        terrain_category_mismatch_count=t_mis,
        terrain_mismatch_samples=t_samples,
        scalar_abs_errors=s_err,
        max_scalar_abs_error=float(max_se),
        influence_mae=mae,
        influence_max_abs=mx,
        belief_enemy_non_point_hp_cells=non_point,
        enemy_hp_interval_width_gt_zero=enemy_hp_w,
    )
