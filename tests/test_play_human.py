"""Play API helpers and flat-action invariants used by human vs bot."""
import numpy as np
import pytest

from engine.action import Action, ActionType, get_legal_actions
from rl.encoder import encode_state
from rl.env import _action_to_flat, _flat_to_action

from server.play_human import BOT_PLAYER, POOL_PATH, MAPS_DIR, build_play_payload
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


def test_ego_observer_differs_for_p1_turn():
    """Bot path must use observer=BOT_PLAYER; P0 vs P1 ego views must diverge here."""
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 1, tier_name="T3")
    guard = 0
    while int(s.active_player) != BOT_PLAYER and not s.done and guard < 500:
        legal = get_legal_actions(s)
        if not legal:
            break
        s.step(legal[0])
        guard += 1
    if int(s.active_player) != BOT_PLAYER:
        pytest.skip("Could not reach P1 turn on this slice")
    a0, b0 = encode_state(s, observer=0)
    a1, b1 = encode_state(s, observer=1)
    assert not np.array_equal(a0, a1) or not np.array_equal(b0, b1)


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


def test_spawn_std_pool_p1_book_matches_misery_when_co_relaxed():
    """std_pool_precombat.jsonl has P1 rows for Misery map 123858 (fixture uses co_id 19)."""
    import uuid

    from server.play_human import P1_OPENING_BOOK_JSONL, _spawn_p1_opening_book_ctl

    if not P1_OPENING_BOOK_JSONL.is_file():
        pytest.skip("opening book fixture missing")
    ctl = _spawn_p1_opening_book_ctl(123858, 19, str(uuid.uuid4()))
    assert ctl is not None
    assert ctl.book_id


def test_spawn_std_pool_p1_book_none_unknown_map():
    import uuid

    from server.play_human import P1_OPENING_BOOK_JSONL, _spawn_p1_opening_book_ctl

    if not P1_OPENING_BOOK_JSONL.is_file():
        pytest.skip("opening book fixture missing")
    ctl = _spawn_p1_opening_book_ctl(424242424, 1, str(uuid.uuid4()))
    assert ctl is None


