"""Play API helpers and flat-action invariants used by human vs bot."""
import pytest

from engine.action import Action, ActionType, get_legal_actions
from rl.env import _action_to_flat, _flat_to_action

from server.play_human import POOL_PATH, MAPS_DIR, build_play_payload
from engine.game import make_initial_state
from engine.map_loader import load_map


def test_repair_flat_index_range():
    a = Action(
        ActionType.REPAIR,
        unit_pos=(5, 5),
        move_pos=(5, 5),
        target_pos=(5, 6),
    )
    idx = _action_to_flat(a)
    assert 3500 <= idx < 3500 + 30 * 30


def test_end_turn_flat_zero():
    assert _action_to_flat(Action(ActionType.END_TURN)) == 0


def test_build_play_payload_keys():
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    p = build_play_payload("sid", s)
    for k in (
        "session_id",
        "ok",
        "action_stage",
        "active_player",
        "board",
        "legal_global",
        "selectable_unit_tiles",
        "reachable_tiles",
        "attack_targets",
        "repair_targets",
        "action_options",
        "unload_options",
    ):
        assert k in p


def test_demo_row_flat_roundtrip_end_turn_when_legal():
    """BC loader sanity: global flat index 0 (END_TURN) round-trips when legal."""
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    if s.active_player != 0:
        pytest.skip("Asymmetric predeploy can open on P1")
    legal = get_legal_actions(s)
    ends = [a for a in legal if a.action_type == ActionType.END_TURN]
    if not ends:
        pytest.skip("No END_TURN on this opening slice — try another map/CO")
    a = ends[0]
    idx = _action_to_flat(a)
    assert idx == 0
    back = _flat_to_action(idx, s)
    assert back is not None
    assert back.action_type == ActionType.END_TURN


