"""
Own-end-turn spirit_broken heuristic (material + EMA / soft-hold) and resign streak.

Runs from ``GameState._end_turn`` so ``state.step`` / MCTS rollouts match training.
Enabled when ``AWBW_SPIRIT_BROKEN`` is truthy; tier/std gates mirror ``rl.heuristic_termination``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engine.unit import UNIT_STATS

if TYPE_CHECKING:
    from engine.game import GameState

SPIRIT_BROKEN_REASON = "spirit_broken"
_EPS = 1.0


def _spirit_play_enabled() -> bool:
    v = (os.environ.get("AWBW_SPIRIT_BROKEN", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class SpiritPlayConfig:
    """Termination thresholds; env-tunable (AWBW_SPIRIT_*)."""

    min_day: int = 8
    required_own_turn_streak: int = 3
    # Hard (crushing) — ratio is actor_value / opponent_value
    income_property_lead: int = 2
    unit_count_lead: int = 2
    unit_value_ratio: float = 1.10
    # Soft hold
    soft_income_property_lead: int = 1
    soft_unit_count_lead: int = 1
    soft_unit_value_ratio: float = 1.06
    # EMA
    ema_alpha: float = 0.35
    ema_threshold: float = 1.0
    # Pressure score weights (redesign defaults)
    property_weight: float = 0.35
    income_property_weight: float = 0.65
    unit_count_weight: float = 0.35
    unit_value_ratio_weight: float = 2.0
    funds_weight: float = 0.00002
    # Gates (shared with RL diag naming)
    value_margin: float = 0.10  # for resign material only (ratio lead uses this margin)
    allowed_tiers: frozenset[str] = field(default_factory=frozenset)
    require_std_map: bool = True


def spirit_play_config_from_env() -> SpiritPlayConfig:
    c = SpiritPlayConfig(
        min_day=max(1, _read_int("AWBW_SPIRIT_MIN_DAY", 8)),
        required_own_turn_streak=max(1, _read_int("AWBW_SPIRIT_REQUIRED_STREAK", 3)),
        income_property_lead=max(0, _read_int("AWBW_SPIRIT_INCOME_LEAD", 2)),
        unit_count_lead=max(0, _read_int("AWBW_SPIRIT_UNIT_COUNT_LEAD", 2)),
        unit_value_ratio=max(1.0, _read_float("AWBW_SPIRIT_UNIT_VALUE_RATIO", 1.10)),
        soft_income_property_lead=max(0, _read_int("AWBW_SPIRIT_SOFT_INCOME_LEAD", 1)),
        soft_unit_count_lead=max(0, _read_int("AWBW_SPIRIT_SOFT_UNIT_COUNT_LEAD", 1)),
        soft_unit_value_ratio=max(1.0, _read_float("AWBW_SPIRIT_SOFT_UNIT_VALUE_RATIO", 1.06)),
        ema_alpha=min(1.0, max(0.0, _read_float("AWBW_SPIRIT_EMA_ALPHA", 0.35))),
        ema_threshold=_read_float("AWBW_SPIRIT_EMA_THRESHOLD", 1.0),
        property_weight=_read_float("AWBW_SPIRIT_W_PROP", 0.35),
        income_property_weight=_read_float("AWBW_SPIRIT_W_INCOME_PROP", 0.65),
        unit_count_weight=_read_float("AWBW_SPIRIT_W_COUNT", 0.35),
        unit_value_ratio_weight=_read_float("AWBW_SPIRIT_W_RATIO", 2.0),
        funds_weight=_read_float("AWBW_SPIRIT_W_FUNDS", 0.00002),
        value_margin=_read_float("AWBW_SPIRIT_VALUE_MARGIN", 0.10),
    )
    raw = (os.environ.get("AWBW_SPIRIT_TIERS", "") or "").strip()
    if raw:
        c.allowed_tiers = frozenset(x.strip() for x in raw.split(",") if x.strip())
    v = (os.environ.get("AWBW_SPIRIT_REQUIRE_STD", "1") or "1").strip().lower()
    c.require_std_map = v in ("1", "true", "yes", "on", "")
    return c


@dataclass
class SpiritState:
    """Persistent spirit bookkeeping on ``GameState`` (deepcopy-safe)."""

    pressure_streak: list[int] = field(default_factory=lambda: [0, 0])
    pressure_ema: list[float] = field(default_factory=lambda: [0.0, 0.0])
    resign_streak: list[int] = field(default_factory=lambda: [0, 0])
    spirit_broken_kind: str | None = None


def _army_value(state: Any, player: int) -> float:
    return float(
        sum(
            UNIT_STATS[u.unit_type].cost * u.hp / 100.0
            for u in state.units.get(player, [])
            if u.is_alive
        )
    )


def _income_snapshot(state: Any) -> dict[str, Any]:
    p0 = sum(1 for u in state.units[0] if u.is_alive)
    p1 = sum(1 for u in state.units[1] if u.is_alive)
    return {
        "income_p0": int(state.count_income_properties(0)),
        "income_p1": int(state.count_income_properties(1)),
        "count_p0": p0,
        "count_p1": p1,
        "value_p0": _army_value(state, 0),
        "value_p1": _army_value(state, 1),
    }


def _material_margins(
    m: dict[str, Any], value_margin: float
) -> tuple[int, int, bool, bool]:
    d_prop = int(m["income_p0"]) - int(m["income_p1"])
    d_count = int(m["count_p0"]) - int(m["count_p1"])
    v0, v1 = float(m["value_p0"]), float(m["value_p1"])
    vm = 1.0 + float(value_margin)
    r01 = v0 / max(_EPS, v1) if v1 > 0 else (vm + 1.0 if v0 > 0 else 1.0)
    r10 = v1 / max(_EPS, v0) if v0 > 0 else (vm + 1.0 if v1 > 0 else 1.0)
    p0_value_lead = r01 >= vm
    p1_value_lead = r10 >= vm
    return d_prop, d_count, p0_value_lead, p1_value_lead


def _resign_crush_material(m: dict[str, Any], seat: int, value_margin: float) -> bool:
    d_prop, d_count, p0v, p1v = _material_margins(m, value_margin)
    if seat == 0:
        return d_prop <= -2 and d_count <= -2 and p1v
    return d_prop >= 2 and d_count >= 2 and p0v


def _gate_applies(state: Any, cfg: SpiritPlayConfig) -> bool:
    if state.done:
        return False
    if cfg.require_std_map:
        sm = getattr(state, "spirit_map_is_std", None)
        if sm is not True:
            return False
    tier = str(getattr(state, "tier_name", "") or "")
    if cfg.allowed_tiers and tier not in cfg.allowed_tiers:
        return False
    return True


def _pressure_score(
    income_lead: int,
    count_lead: int,
    ratio: float,
    captured_lead: int,
    funds_lead: int,
    cfg: SpiritPlayConfig,
) -> float:
    unit_ratio_excess = ratio - 1.0
    s = 0.0
    s += cfg.property_weight * captured_lead
    s += cfg.income_property_weight * income_lead
    s += cfg.unit_count_weight * count_lead
    s += cfg.unit_value_ratio_weight * unit_ratio_excess
    s += cfg.funds_weight * funds_lead
    return float(s)


def _captured_income_lead(state: Any, actor: int, opp: int) -> int:
    """Income-producing tiles (same rule as ``count_income_properties``)."""

    def _inc(p: Any) -> bool:
        return (
            p.owner is not None
            and not p.is_comm_tower
            and not p.is_lab
        )

    a = sum(1 for p in state.properties if p.owner == actor and _inc(p))
    b = sum(1 for p in state.properties if p.owner == opp and _inc(p))
    return a - b


def maybe_spirit_after_end_turn(state: GameState, ended_player: int) -> None:
    """Call at end of ``_end_turn`` with the seat that just ended (0 or 1)."""
    if getattr(state, "spirit", None) is None:
        state.spirit = SpiritState()
    if not _spirit_play_enabled() or state.done:
        return
    cfg = spirit_play_config_from_env()
    if not _gate_applies(state, cfg):
        return
    if int(state.turn) < int(cfg.min_day):
        return

    actor = int(ended_player)
    if actor not in (0, 1):
        return
    opp = 1 - actor

    m = _income_snapshot(state)
    # Optional instant crush: empty army / zero value (same spirit as old material_crush fast path)
    if (
        int(m["count_p0"]) == 0
        or int(m["count_p1"]) == 0
        or float(m["value_p0"]) == 0.0
        or float(m["value_p1"]) == 0.0
    ):
        # Let normal win conditions handle elimination; do not spirit here.
        pass

    inc0, inc1 = int(m["income_p0"]), int(m["income_p1"])
    c0, c1 = int(m["count_p0"]), int(m["count_p1"])
    v0, v1 = float(m["value_p0"]), float(m["value_p1"])
    if actor == 0:
        income_lead = inc0 - inc1
        count_lead = c0 - c1
        va, vo = v0, v1
    else:
        income_lead = inc1 - inc0
        count_lead = c1 - c0
        va, vo = v1, v0
    ratio = va / max(_EPS, vo) if vo > 0 else (cfg.unit_value_ratio + 1.0 if va > 0 else 1.0)
    captured_lead = _captured_income_lead(state, actor, opp)
    funds_lead = int(state.funds[actor]) - int(state.funds[opp])

    hard = (
        income_lead >= cfg.income_property_lead
        and count_lead >= cfg.unit_count_lead
        and ratio >= cfg.unit_value_ratio
    )
    soft = (
        income_lead >= cfg.soft_income_property_lead
        and count_lead >= cfg.soft_unit_count_lead
        and ratio >= cfg.soft_unit_value_ratio
    )
    score = _pressure_score(
        income_lead, count_lead, ratio, captured_lead, funds_lead, cfg
    )
    sp = state.spirit
    old_ema = float(sp.pressure_ema[actor])
    alpha = float(cfg.ema_alpha)
    new_ema = (1.0 - alpha) * old_ema + alpha * score
    sp.pressure_ema[actor] = new_ema

    if hard and new_ema >= float(cfg.ema_threshold):
        sp.pressure_streak[actor] += 1
    elif soft:
        pass
    else:
        sp.pressure_streak[actor] = 0

    # Resign (trailer) streak — material only, own-turn for this actor
    if _resign_crush_material(m, actor, cfg.value_margin):
        sp.resign_streak[actor] += 1
    else:
        sp.resign_streak[actor] = 0

    req = int(cfg.required_own_turn_streak)
    if sp.pressure_streak[actor] >= req:
        state.done = True
        state.winner = actor
        state.win_reason = SPIRIT_BROKEN_REASON
        sp.spirit_broken_kind = "snowball"
        return
    if sp.resign_streak[actor] >= req:
        state.done = True
        state.winner = opp
        state.win_reason = SPIRIT_BROKEN_REASON
        sp.spirit_broken_kind = "resign"
        return
