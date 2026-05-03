"""
Spirit / heuristic helpers: material predicates, value-head diagnostic JSONL.

Spirit **termination** is implemented in ``engine/spirit_pressure`` (own-end-turn).
This module keeps ``SpiritConfig``, ``maybe_log_disagreements``, and legacy name
``run_calendar_day`` for **diagnostics only** when ``AWBW_HEURISTIC_VALUE_DIAG=1``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from engine.game import GameState
from rl.log_timestamp import log_now_iso
from engine.spirit_pressure import SPIRIT_BROKEN_REASON
from engine.unit import UNIT_STATS
EPS_VALUE = 1.0  # for ratio denominator when both armies are empty

ROOT = Path(__file__).parent.parent
DEFAULT_DISAGREEMENT_LOG = ROOT / "logs" / "heuristic_value_disagreement.jsonl"

# region agent log
_AGENT_DEBUG_LOG_PATH = ROOT / "debug-a6d5a1.log"
_AGENT_DEBUG_SESSION_ID = "a6d5a1"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": {**data, "pid": os.getpid()},
            "timestamp": int(time.time() * 1000),
            "timestamp_iso": log_now_iso(),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# endregion


def spirit_enabled_from_env() -> bool:
    v = (os.environ.get("AWBW_SPIRIT_BROKEN", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def diag_enabled_from_env() -> bool:
    v = (os.environ.get("AWBW_HEURISTIC_VALUE_DIAG", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class SpiritConfig:
    """Thresholds; overridable via environment for quick tuning."""

    p_snowball: float = 0.65
    p_trailer_resign_max: float = 0.35
    value_margin: float = 0.15
    # sigmoid(v / T)
    p_win_temperature: float = 1.0
    # Optional comma-separated tier allow list (empty = all)
    allowed_tiers: set[str] = field(default_factory=set)
    require_std_map: bool = True
    # Max JSONL lines per episode for diag
    diag_max_lines_per_episode: int = 32


def _read_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def config_from_env() -> SpiritConfig:
    c = SpiritConfig(
        p_snowball=_read_env_float("AWBW_SPIRIT_P_SNOWBALL", 0.65),
        p_trailer_resign_max=_read_env_float("AWBW_SPIRIT_P_TRAILER_RESIGN", 0.35),
        value_margin=_read_env_float("AWBW_SPIRIT_VALUE_MARGIN", 0.15),
        p_win_temperature=max(1e-6, _read_env_float("AWBW_SPIRIT_P_WIN_TEMP", 1.0)),
    )
    raw = (os.environ.get("AWBW_SPIRIT_TIERS", "") or "").strip()
    if raw:
        c.allowed_tiers = {x.strip() for x in raw.split(",") if x.strip()}
    v = (os.environ.get("AWBW_SPIRIT_REQUIRE_STD", "1") or "1").strip().lower()
    c.require_std_map = v in ("1", "true", "yes", "on", "")
    try:
        c.diag_max_lines_per_episode = max(0, int(os.environ.get("AWBW_HEURISTIC_DIAG_MAX", "32") or 32))
    except ValueError:
        c.diag_max_lines_per_episode = 32
    return c


def raw_value_to_p_win(raw: float, temp: float) -> float:
    t = max(1e-6, float(temp))
    return float(1.0 / (1.0 + math.exp(-float(raw) / t)))


def army_value_for_player(state: GameState, player: int) -> float:
    return float(
        sum(
            UNIT_STATS[u.unit_type].cost * u.hp / 100.0
            for u in state.units.get(player, [])
            if u.is_alive
        )
    )


def income_props_and_counts(state: GameState) -> dict[str, Any]:
    p0 = sum(1 for u in state.units[0] if u.is_alive)
    p1 = sum(1 for u in state.units[1] if u.is_alive)
    v0 = army_value_for_player(state, 0)
    v1 = army_value_for_player(state, 1)
    return {
        "income_p0": int(state.count_income_properties(0)),
        "income_p1": int(state.count_income_properties(1)),
        "count_p0": p0,
        "count_p1": p1,
        "value_p0": v0,
        "value_p1": v1,
    }


def material_margins(
    m: dict[str, Any], value_margin: float
) -> tuple[int, int, bool, bool]:
    """d_prop, d_count, p0_value_lead, p1_value_lead (>= margin vs enemy)."""
    d_prop = int(m["income_p0"]) - int(m["income_p1"])
    d_count = int(m["count_p0"]) - int(m["count_p1"])
    v0, v1 = float(m["value_p0"]), float(m["value_p1"])
    vm = 1.0 + float(value_margin)
    r01 = v0 / max(EPS_VALUE, v1) if v1 > 0 else (vm + 1.0 if v0 > 0 else 1.0)
    r10 = v1 / max(EPS_VALUE, v0) if v0 > 0 else (vm + 1.0 if v1 > 0 else 1.0)
    p0_value_lead = r01 >= vm
    p1_value_lead = r10 >= vm
    return d_prop, d_count, p0_value_lead, p1_value_lead


def snowball_holds(
    m: dict[str, Any], seat: int, p_win_seat: float, cfg: SpiritConfig
) -> bool:
    d_prop, d_count, p0v, p1v = material_margins(m, cfg.value_margin)
    if seat == 0:
        if d_prop < 2 or d_count < 2 or not p0v:
            return False
    else:
        if -d_prop < 2 or -d_count < 2 or not p1v:
            return False
    return p_win_seat >= float(cfg.p_snowball)


def snowball_material_holds(m: dict[str, Any], seat: int, cfg: SpiritConfig) -> bool:
    d_prop, d_count, p0v, p1v = material_margins(m, cfg.value_margin)
    if seat == 0:
        return d_prop >= 2 and d_count >= 2 and p0v
    return -d_prop >= 2 and -d_count >= 2 and p1v


def resign_crush_holds(
    m: dict[str, Any], seat: int, p_win_seat: float, cfg: SpiritConfig
) -> bool:
    """Trailer S is down ≥2 on props+count; enemy has ≥15% army value lead; pS low."""
    d_prop, d_count, p0v, p1v = material_margins(m, cfg.value_margin)
    if seat == 0:
        if d_prop > -2 or d_count > -2 or not p1v:
            return False
    else:
        if d_prop < 2 or d_count < 2 or not p0v:
            return False
    return p_win_seat < float(cfg.p_trailer_resign_max)


def resign_crush_material_holds(m: dict[str, Any], seat: int, cfg: SpiritConfig) -> bool:
    """Trailer S is down materially enough that neutral critic output is uninformative."""
    d_prop, d_count, p0v, p1v = material_margins(m, cfg.value_margin)
    if seat == 0:
        return d_prop <= -2 and d_count <= -2 and p1v
    return d_prop >= 2 and d_count >= 2 and p0v


@dataclass
class SpiritStreaks:
    snowball: list[int] = field(default_factory=lambda: [0, 0])
    resign: list[int] = field(default_factory=lambda: [0, 0])

    def reset(self) -> None:
        self.snowball = [0, 0]
        self.resign = [0, 0]


def gate_applies(
    state: GameState,
    cfg: SpiritConfig,
    *,
    is_std_map: bool,
    allowed_tier: bool,
    enabled: bool,
) -> bool:
    if not enabled or state.done:
        return False
    if cfg.require_std_map and not is_std_map:
        return False
    if not allowed_tier:
        return False
    return True


def _class_buckets(state: GameState) -> tuple[dict[str, int], dict[str, int]]:
    a0: dict[str, int] = {}
    a1: dict[str, int] = {}
    for u in state.units[0]:
        if not u.is_alive:
            continue
        cl = UNIT_STATS[u.unit_type].unit_class
        a0[cl] = a0.get(cl, 0) + 1
    for u in state.units[1]:
        if not u.is_alive:
            continue
        cl = UNIT_STATS[u.unit_type].unit_class
        a1[cl] = a1.get(cl, 0) + 1
    return a0, a1


def _hq_positions(state: GameState) -> list[dict[str, Any]]:
    out = []
    for p in state.properties:
        if getattr(p, "is_hq", False) and p.owner in (0, 1):
            out.append(
                {
                    "row": p.row,
                    "col": p.col,
                    "owner": int(p.owner),
                }
            )
    return out


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _predict_p_win_both(
    state: GameState,
    model,
    encode_fn,
    *,
    cfg: SpiritConfig,
    observer1_model=None,
) -> tuple[float, float, float, float]:
    """Returns p0, p1, v0, v1 raw from critic after sigmoid map.

    When *observer1_model* is set, seat-1 observations use its policy value head;
    otherwise the same *model* evaluates both ego-centric views (legacy).
    """
    import torch

    obs0 = encode_fn(state, 0)
    obs1 = encode_fn(state, 1)
    pol0 = model.policy
    pol1 = pol0 if observer1_model is None else observer1_model.policy
    device0 = model.device
    device1 = device0 if observer1_model is None else observer1_model.device
    b0, _ = pol0.obs_to_tensor(obs0)
    b1, _ = pol1.obs_to_tensor(obs1)
    if isinstance(b0, dict):
        b0 = {k: v.to(device0) for k, v in b0.items()}
    else:
        b0 = b0.to(device0)
    if isinstance(b1, dict):
        b1 = {k: v.to(device1) for k, v in b1.items()}
    else:
        b1 = b1.to(device1)
    with torch.no_grad():
        v_0 = pol0.predict_values(b0)
        v_1 = pol1.predict_values(b1)
    r0 = float(v_0.reshape(-1)[0].detach().cpu().numpy())
    r1 = float(v_1.reshape(-1)[0].detach().cpu().numpy())
    t = float(cfg.p_win_temperature)
    p0 = raw_value_to_p_win(r0, t)
    p1 = raw_value_to_p_win(r1, t)
    return p0, p1, r0, r1


def maybe_log_disagreements(
    state: GameState,
    m: dict[str, Any],
    p0: float,
    p1: float,
    r0: float,
    r1: float,
    cfg: SpiritConfig,
    *,
    episode_id: int,
    map_id: int | None,
    tier_name: str,
    learner_seat: int,
    log_path: Path,
    lines_used: int,
) -> int:
    """Log spirit **material** vs **value head** using the same thresholds as
    ``snowball_holds`` / ``resign_crush_holds`` (``p_snowball`` / ``p_trailer_resign_max``),
    not a 0.5 neutral line.

    * ``spirit_snowball_material_p_below_bar`` — snowball material for the seat, but
      ``p_win`` is below the snowball bar (streak will not tick).
    * ``spirit_resign_material_p_above_bar`` — resign material for the seat, but
      ``p_win`` is at/above the trailer-resign bar (streak will not tick).
    """
    if not diag_enabled_from_env() or lines_used >= cfg.diag_max_lines_per_episode:
        return 0
    d_prop, d_count, p0v, p1v = material_margins(m, cfg.value_margin)
    extra = 0
    p_sb = float(cfg.p_snowball)
    p_tr = float(cfg.p_trailer_resign_max)
    for seat in (0, 1):
        if lines_used + extra >= cfg.diag_max_lines_per_episode:
            break
        p_s = p0 if seat == 0 else p1
        if seat == 0:
            mat_lose = d_prop <= -2 and d_count <= -2 and p1v
            mat_win = d_prop >= 2 and d_count >= 2 and p0v
        else:
            mat_lose = d_prop >= 2 and d_count >= 2 and p0v
            mat_win = d_prop <= -2 and d_count <= -2 and p1v
        case = None
        if mat_win and p_s < p_sb:
            case = "spirit_snowball_material_p_below_bar"
        elif mat_lose and p_s >= p_tr:
            case = "spirit_resign_material_p_above_bar"
        if case is None:
            continue
        a0, a1 = _class_buckets(state)
        rec: dict[str, Any] = {
            "ts": log_now_iso(),
            "case": case,
            "turn": int(state.turn),
            "map_id": map_id,
            "tier_name": tier_name,
            "episode_id": episode_id,
            "learner_seat": int(learner_seat),
            "seat": seat,
            "p0_co": int(state.co_states[0].co_id),
            "p1_co": int(state.co_states[1].co_id),
            "funds": list(state.funds),
            "cop_stars_p0": float((state.co_states[0].cop_stars or 0)),
            "scop_stars_p0": float(state.co_states[0].scop_stars),
            "cop_stars_p1": float((state.co_states[1].cop_stars or 0)),
            "scop_stars_p1": float(state.co_states[1].scop_stars),
        }
        rec["p_win_temp"] = float(cfg.p_win_temperature)
        rec["p0_m"] = p0
        rec["p1_m"] = p1
        rec["v0_raw"] = r0
        rec["v1_raw"] = r1
        rec["material"] = m
        rec["d_prop"] = d_prop
        rec["d_count"] = d_count
        rec["class_b0"] = a0
        rec["class_b1"] = a1
        rec["hq_tiles"] = _hq_positions(state)
        p0i = int(m["income_p0"])
        p1i = int(m["income_p1"])
        rec["winning_if_income_only"] = 0 if p0i > p1i else (1 if p1i > p0i else -1)
        _append_jsonl(log_path, rec)
        extra += 1
    return extra


def run_calendar_day(
    state: GameState,
    model: Any,
    cfg: SpiritConfig,
    encode_fn: Any,
    *,
    is_std_map: bool,
    map_tier_ok: bool,
    episode_id: int,
    map_id: int | None,
    learner_seat: int,
    log_path: Path = DEFAULT_DISAGREEMENT_LOG,
    diag_line_budget: int = 0,
    observer1_model: Any | None = None,
) -> tuple[Optional[str], int]:
    """
    Value-head diagnostic only (``AWBW_HEURISTIC_VALUE_DIAG``). Spirit **termination**
    runs in ``engine/spirit_pressure.maybe_spirit_after_end_turn`` on each ``END_TURN``.
    Returns ``(None, n_diag_lines)`` — the first value is always ``None`` (legacy API).
    """
    if not diag_enabled_from_env():
        return None, 0
    if not gate_applies(
        state, cfg, is_std_map=is_std_map, allowed_tier=map_tier_ok, enabled=True
    ):
        return None, 0
    m = income_props_and_counts(state)
    n_written = 0
    p0 = p1 = 0.0
    r0 = r1 = 0.0
    if model is not None:
        try:
            p0, p1, r0, r1 = _predict_p_win_both(
                state,
                model,
                encode_fn,
                cfg=cfg,
                observer1_model=observer1_model,
            )
        except Exception:
            p0, p1, r0, r1 = 0.0, 0.0, 0.0, 0.0
        n_written = maybe_log_disagreements(
            state,
            m,
            p0,
            p1,
            r0,
            r1,
            cfg,
            episode_id=episode_id,
            map_id=map_id,
            tier_name=str(state.tier_name),
            learner_seat=learner_seat,
            log_path=log_path,
            lines_used=diag_line_budget,
        )
    return None, n_written
