"""
MASTERPLAN §14 — MCTS rollout stage presets (MCTS-0 … MCTS-4).

Stages are **advisory** defaults: they map to a full :class:`rl.mcts.MCTSConfig`
and JSON/CLI fields override individual knobs when present in the eval payload.
Not all stages are “production ready”; this module encodes the ladder in code
so eval / fleet can select ``mcts_rollout_stage`` without hand-copying sim counts.

Stage IDs (``str``):
  * ``mcts_0`` — plumbing / debug (low sims, some exploration)
  * ``mcts_1`` — symmetric eval / gating (mid sims, reproducible: low Dirichlet)
  * ``mcts_2`` — selective training assist (higher sims, ``p0_mcts_invocation_fraction`` < 1)
  * ``mcts_3`` — distillation / offline-teacher**-shaped** defaults (sims like eval; no distillation training pipeline here yet)
  * ``mcts_4`` — production-style: high sim cap + **optional** wall-time budget (``max_wall_time_s``)
"""

from __future__ import annotations

from dataclasses import fields, replace
from enum import StrEnum
from typing import Any

from rl.mcts import MCTSConfig


class MCTSRolloutStage(StrEnum):
    MCTS_0 = "mcts_0"
    MCTS_1 = "mcts_1"
    MCTS_2 = "mcts_2"
    MCTS_3 = "mcts_3"
    MCTS_4 = "mcts_4"


# Payload / CLI keys -> MCTSConfig field names
_PAYLOAD_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("mcts_sims", "num_sims"),
    ("mcts_c_puct", "c_puct"),
    ("mcts_dirichlet_alpha", "dirichlet_alpha"),
    ("mcts_dirichlet_epsilon", "dirichlet_epsilon"),
    ("mcts_temperature", "temperature"),
    ("mcts_min_depth", "min_depth"),
    ("mcts_root_plans", "root_plans"),
    ("mcts_max_plan_actions", "max_plan_actions"),
    ("mcts_luck_resamples", "luck_resamples"),
    ("mcts_luck_resample_critical_only", "luck_resample_critical_only"),
    ("mcts_risk_mode", "risk_mode"),
    ("mcts_risk_lambda", "risk_lambda"),
    ("mcts_catastrophe_value", "catastrophe_value"),
    ("mcts_max_catastrophe_prob", "max_catastrophe_prob"),
    ("mcts_root_decision_log", "root_decision_log_path"),
    ("mcts_max_wall_time_s", "max_wall_time_s"),
    ("mcts_p0_mcts_invocation_fraction", "p0_mcts_invocation_fraction"),
)

_STAGE_PRESETS: dict[str, MCTSConfig] = {
    "mcts_0": MCTSConfig(
        num_sims=16,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.35,
        temperature=0.0,
        min_depth=0,
        root_plans=4,
        max_plan_actions=256,
        luck_resamples=0,
        luck_resample_critical_only=True,
        risk_mode="visit",
        rollout_stage="mcts_0",
        max_wall_time_s=None,
        p0_mcts_invocation_fraction=1.0,
    ),
    "mcts_1": MCTSConfig(
        num_sims=128,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        min_depth=4,
        root_plans=8,
        max_plan_actions=256,
        luck_resamples=0,
        luck_resample_critical_only=True,
        risk_mode="visit",
        rollout_stage="mcts_1",
        max_wall_time_s=None,
        p0_mcts_invocation_fraction=1.0,
    ),
    "mcts_2": MCTSConfig(
        num_sims=256,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        min_depth=4,
        root_plans=12,
        max_plan_actions=256,
        luck_resamples=0,
        luck_resample_critical_only=True,
        risk_mode="visit",
        rollout_stage="mcts_2",
        max_wall_time_s=None,
        p0_mcts_invocation_fraction=0.5,
    ),
    "mcts_3": MCTSConfig(
        num_sims=256,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        min_depth=4,
        root_plans=8,
        max_plan_actions=256,
        luck_resamples=0,
        luck_resample_critical_only=True,
        risk_mode="visit",
        rollout_stage="mcts_3",
        max_wall_time_s=None,
        p0_mcts_invocation_fraction=1.0,
    ),
    "mcts_4": MCTSConfig(
        num_sims=10_000,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
        temperature=0.0,
        min_depth=4,
        root_plans=16,
        max_plan_actions=256,
        luck_resamples=0,
        luck_resample_critical_only=True,
        risk_mode="visit",
        rollout_stage="mcts_4",
        max_wall_time_s=30.0,
        p0_mcts_invocation_fraction=1.0,
    ),
}


def _normalize_stage_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "off"):
        return None
    if s in _STAGE_PRESETS:
        return s
    raise ValueError(
        f"unknown mcts_rollout_stage {raw!r}; use one of: {', '.join(sorted(_STAGE_PRESETS))} or off/none"
    )


def mcts_config_for_stage(stage_id: str) -> MCTSConfig:
    """Return a copy of the preset for ``mcts_0`` … ``mcts_4``."""
    s = _normalize_stage_id(stage_id)
    if s is None:
        return MCTSConfig()
    return replace(_STAGE_PRESETS[s])


def mcts_config_from_eval_payload(payload: dict[str, Any]) -> MCTSConfig:
    """
    Build :class:`MCTSConfig` from a symmetric eval / worker ``payload`` dict.

    * If ``mcts_rollout_stage`` is set, start from that stage preset; otherwise
      :class:`MCTSConfig` defaults.
    * For each field in the payload map, if that **key is present** in ``payload``,
      its value overrides the preset (so JSON and explicit CLI can tweak one knob).
    """
    s = _normalize_stage_id(payload.get("mcts_rollout_stage"))
    base: MCTSConfig
    if s is not None:
        base = replace(_STAGE_PRESETS[s])
    else:
        base = MCTSConfig()
    d: dict[str, Any] = {f.name: getattr(base, f.name) for f in fields(MCTSConfig)}
    for pkey, fname in _PAYLOAD_FIELD_MAP:
        if pkey in payload:
            d[fname] = payload[pkey]
    return MCTSConfig(**d)


# Legacy default map when *no* rollout stage: argparse may omit args (``SUPPRESS``).
_LEGACY_ARG_DEFAULTS: dict[str, Any] = {
    "mcts_sims": 16,
    "mcts_c_puct": 1.5,
    "mcts_dirichlet_alpha": 0.3,
    "mcts_dirichlet_epsilon": 0.25,
    "mcts_temperature": 1.0,
    "mcts_min_depth": 4,
    "mcts_root_plans": 8,
    "mcts_max_plan_actions": 256,
    "mcts_luck_resamples": 0,
    "mcts_luck_resample_critical_only": True,
    "mcts_risk_mode": "visit",
    "mcts_risk_lambda": 0.35,
    "mcts_catastrophe_value": -0.35,
    "mcts_max_catastrophe_prob": 1.0,
    "mcts_root_decision_log": None,
    "mcts_max_wall_time_s": None,
    "mcts_p0_mcts_invocation_fraction": 1.0,
}

_PAYLOAD_KEYS: frozenset[str] = frozenset(a for a, _ in _PAYLOAD_FIELD_MAP)


def mcts_work_payload_from_argparse(args: Any) -> dict[str, Any]:
    """
    Build the worker JSON fragment for MCTS. When a rollout stage is set, only
    ``mcts_*`` keys that appear on ``args`` (``argparse.SUPPRESS``) are included and
    override the stage preset. Without a stage, all keys are filled using legacy
    defaults for missing attributes (same net behavior as the pre-stage CLI).
    """
    st = _normalize_stage_id(getattr(args, "mcts_rollout_stage", None))
    out: dict[str, Any] = {
        "mcts_mode": str(getattr(args, "mcts_mode", "off")),
        "mcts_rollout_stage": st,
    }
    ad = args.__dict__
    if st is not None:
        for k in _PAYLOAD_KEYS:
            if k in ad:
                out[k] = ad[k]
        return out
    for k, default in _LEGACY_ARG_DEFAULTS.items():
        out[k] = ad[k] if k in ad else default
    return out


__all__ = [
    "MCTSRolloutStage",
    "MCTS_STAGE_PRESETS",
    "mcts_config_for_stage",
    "mcts_config_from_eval_payload",
    "mcts_work_payload_from_argparse",
]


# Back-compat alias
STAGE_PRESETS = _STAGE_PRESETS
MCTS_STAGE_PRESETS = _STAGE_PRESETS
