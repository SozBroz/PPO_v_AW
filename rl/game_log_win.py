"""Shared interpretation of ``game_log.jsonl`` rows for learner-centric wins."""
from __future__ import annotations

from typing import Any


def game_log_row_learner_win(row: dict[str, Any]) -> bool:
    """
    True if the training learner won the episode: natural terminal with ``winner``
    matching ``learner_seat``, or a step-cap tie-break logged under
    ``tie_breaker_property_count`` (property lead in the learner frame, int >= 1).
    """
    ls = int(row.get("learner_seat", 0))
    w = row.get("winner")
    try:
        wi = int(w) if w is not None else -99
    except (TypeError, ValueError):
        wi = -99
    tb = row.get("tie_breaker_property_count")
    try:
        has_tb = tb is not None and int(tb) >= 1
    except (TypeError, ValueError):
        has_tb = False
    return (wi == ls) or has_tb
