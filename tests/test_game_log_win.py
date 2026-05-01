"""rl.game_log_win: learner-centric win including step-cap tie-break."""
from __future__ import annotations

from rl.game_log_win import game_log_row_learner_win


def test_natural_win_learner_p0() -> None:
    assert game_log_row_learner_win({"learner_seat": 0, "winner": 0})


def test_natural_loss_learner_p0() -> None:
    assert not game_log_row_learner_win({"learner_seat": 0, "winner": 1})


def test_tie_breaker_counts_when_winner_null() -> None:
    assert game_log_row_learner_win(
        {
            "learner_seat": 0,
            "winner": None,
            "tie_breaker_property_count": 3,
        }
    )


def test_tie_breaker_requires_min_lead() -> None:
    assert not game_log_row_learner_win(
        {"learner_seat": 0, "winner": None, "tie_breaker_property_count": 0}
    )


def test_tie_breaker_counts_lead_one() -> None:
    assert game_log_row_learner_win(
        {"learner_seat": 0, "winner": None, "tie_breaker_property_count": 1}
    )


def test_learner_p1_win() -> None:
    assert game_log_row_learner_win({"learner_seat": 1, "winner": 1})
