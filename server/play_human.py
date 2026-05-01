"""
Human vs bot play sessions: in-memory GameState, JSON API envelope (plan §4.1),
bot stepping with MaskablePPO, and optional human_demos.jsonl logging.
"""
from __future__ import annotations

import glob
import json
import os
import random
import threading
import uuid
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import numpy as np

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    _build_cost,
    get_attack_targets,
    get_legal_actions,
    get_reachable_tiles,
)
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.unit import UnitType
from rl.encoder import N_SCALARS, N_SPATIAL_CHANNELS, encode_state
from rl.env import (
    _action_label,
    _action_to_flat,
    _flat_to_action,
    _get_action_mask,
)
from rl.paths import HUMAN_DEMOS_PATH
from server.write_watch_state import board_dict

ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
P1_OPENING_BOOK_JSONL = ROOT / "data" / "opening_books" / "std_pool_precombat.jsonl"

_OPENING_BOOK_INDEX: Any | None = None
_OPENING_BOOK_INDEX_KEY: Optional[str] = None


def _p1_opening_book_strict_co() -> bool:
    """When true, P1 uses only book lines whose stored CO matches the bot (opt-in via env)."""

    raw = os.environ.get("AWBW_PLAY_OPENING_BOOK_STRICT_CO", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _std_pool_opening_book_index():
    """Parse ``std_pool_precombat.jsonl`` once per path (lazy, thread-safe-enough under GIL writes)."""

    global _OPENING_BOOK_INDEX, _OPENING_BOOK_INDEX_KEY
    path = P1_OPENING_BOOK_JSONL
    if not path.is_file():
        _OPENING_BOOK_INDEX = None
        _OPENING_BOOK_INDEX_KEY = None
        return None
    key = str(path.resolve())
    if _OPENING_BOOK_INDEX is None or _OPENING_BOOK_INDEX_KEY != key:
        from rl.opening_book import OpeningBookIndex

        _OPENING_BOOK_INDEX = OpeningBookIndex.from_jsonl(path)
        _OPENING_BOOK_INDEX_KEY = key
    return _OPENING_BOOK_INDEX


def _spawn_p1_opening_book_ctl(
    map_id: int, bot_co_id: int, session_id: str
) -> Optional[Any]:
    """P1 OpeningBookController for this session, or None if file missing / no eligible line."""

    from rl.opening_book import OpeningBookController

    idx = _std_pool_opening_book_index()
    if idx is None:
        return None
    try:
        seed = int(UUID(session_id).int % (2**63))
    except Exception:
        seed = hash(session_id)
    ctl = OpeningBookController(
        idx,
        seat=BOT_PLAYER,
        strict_co=_p1_opening_book_strict_co(),
        rng=random.Random((seed ^ 0x9E3779B9) & (2**63 - 1)),
        max_calendar_turn=None,
    )
    ctl.on_episode_start(
        episode_id=seed & 0x7FFFFFFF,
        map_id=int(map_id),
        co_id_for_seat=int(bot_co_id),
        enabled=True,
    )
    if ctl.book_id is None:
        return None
    return ctl

# Seat invariants — match training: human always P0 (red / first seat), bot P1 (blue / second).
# Ego-centric encoder: human demos use observer=HUMAN_PLAYER; bot inference uses observer=BOT_PLAYER.
HUMAN_PLAYER = 0
BOT_PLAYER = 1

# Session TTL (plan §session-api): not enforced in MVP; a future sweep can evict stale UUIDs.
_SESSION_TTL_S: Optional[float] = None

_DEFAULT_MAP_ID = 123858  # Misery — MASTERPLAN Stage 1 / train.py narrow bootstrap
_DEFAULT_TIER = "T4"  # Jess (14) appears under T4 for this map only (Andy etc. remain T3)
_DEFAULT_HUMAN_CO = 14  # Jess mirror for /play defaults
_DEFAULT_BOT_CO = 14

_sessions: dict[str, GameState] = {}
_session_meta: dict[str, dict[str, Any]] = {}
# Flask may run threaded=True; serialize mutations + bot loop per process.
_session_io_lock = threading.Lock()

_model = None
_model_lock = threading.Lock()
_model_load_error: Optional[str] = None


def _checkpoint_path(checkpoint_dir: Path) -> Optional[Path]:
    latest = checkpoint_dir / "latest.zip"
    if latest.is_file():
        return latest
    pattern = str(checkpoint_dir / "checkpoint_*.zip")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        return None
    return Path(ckpts[-1])


def ensure_model_loaded(checkpoint_dir: Path) -> tuple[Optional[Any], Optional[str]]:
    """Return (model, error_message). Model is shared across sessions."""
    global _model, _model_load_error
    with _model_lock:
        if _model is not None:
            return _model, None
        if _model_load_error is not None:
            return None, _model_load_error
        path = _checkpoint_path(checkpoint_dir)
        if path is None:
            # No weights: play UI uses a masked-random legal bot (see _run_bot_turn).
            return None, None
        try:
            from rl.ckpt_compat import load_maskable_ppo_compat

            _model = load_maskable_ppo_compat(path, device="cpu")
        except Exception as exc:  # pragma: no cover - env-specific
            _model_load_error = f"Failed to load {path}: {exc}"
            return None, _model_load_error

        # Shape contract — fail fast if encoder and checkpoint diverge.
        try:
            obs_space = _model.observation_space
            sp = obs_space["spatial"].shape
            sc = obs_space["scalars"].shape
        except Exception:
            return _model, None
        if sp[2] != N_SPATIAL_CHANNELS or sc[0] != N_SCALARS:
            _model = None
            _model_load_error = (
                f"Checkpoint obs mismatch: expected spatial[2]={N_SPATIAL_CHANNELS}, "
                f"scalars[0]={N_SCALARS}; got {sp}, {sc}"
            )
            return None, _model_load_error
        return _model, None


def _power_charge_pct(co) -> tuple[float, float]:
    """(cop progress 0..1, scop progress 0..1) for HUD — uses engine thresholds."""
    cop_t = float(co._cop_threshold)  # type: ignore[attr-defined]
    sco_t = float(co._scop_threshold)  # type: ignore[attr-defined]
    bar = float(co.power_bar)
    cop_pct = min(1.0, bar / cop_t) if cop_t > 0 else 0.0
    scop_pct = min(1.0, bar / sco_t) if sco_t > 0 else 0.0
    return cop_pct, scop_pct


def _co_client_dict(co) -> dict[str, Any]:
    cop_pct, scop_pct = _power_charge_pct(co)
    return {
        "id": co.co_id,
        "name": co.name,
        "cop_active": bool(co.cop_active),
        "scop_active": bool(co.scop_active),
        "cop_pct": cop_pct,
        "scop_pct": scop_pct,
    }


def _legal_global(state: GameState) -> dict[str, bool]:
    if state.action_stage != ActionStage.SELECT or state.active_player != HUMAN_PLAYER:
        return {"cop": False, "scop": False, "end_turn": False}
    legal = get_legal_actions(state)
    types = {a.action_type for a in legal}
    return {
        "cop": ActionType.ACTIVATE_COP in types,
        "scop": ActionType.ACTIVATE_SCOP in types,
        "end_turn": ActionType.END_TURN in types,
    }


def _factory_build_menu(state: GameState) -> list[dict[str, Any]]:
    """Per-factory legal BUILD choices for the play UI (SELECT, human turn only)."""
    if (
        state.done
        or int(state.active_player) != HUMAN_PLAYER
        or state.action_stage != ActionStage.SELECT
    ):
        return []
    by_pos: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for a in get_legal_actions(state):
        if a.action_type != ActionType.BUILD or a.move_pos is None or a.unit_type is None:
            continue
        r, c = a.move_pos
        cost = int(_build_cost(a.unit_type, state, int(state.active_player), (r, c)))
        opt = {"unit_type": a.unit_type.name, "type_id": int(a.unit_type), "cost": cost}
        key = (r, c)
        if key not in by_pos:
            by_pos[key] = []
        # de-dupe same unit type at same factory
        if any(o["type_id"] == opt["type_id"] for o in by_pos[key]):
            continue
        by_pos[key].append(opt)
    return [{"pos": [r, c], "options": opts} for (r, c), opts in sorted(by_pos.items())]


def _stage_hints(state: GameState) -> dict[str, Any]:
    player = state.active_player
    selectable: list[list[int]] = []
    factory_build: list[list[int]] = []
    reachable: list[list[int]] = []
    attack_targets: list[list[int]] = []
    repair_targets: list[list[int]] = []
    unload_options: list[dict[str, Any]] = []
    action_options: list[str] = []

    su_pos: Optional[tuple[int, int]] = None
    sm_pos: Optional[tuple[int, int]] = None
    if state.selected_unit is not None:
        su_pos = state.selected_unit.pos
    if state.selected_move_pos is not None:
        sm_pos = state.selected_move_pos

    if state.action_stage == ActionStage.SELECT and not state.done:
        seen_factory: set[tuple[int, int]] = set()
        for a in get_legal_actions(state):
            if a.action_type == ActionType.SELECT_UNIT and a.unit_pos is not None:
                selectable.append([a.unit_pos[0], a.unit_pos[1]])
            elif a.action_type == ActionType.BUILD and a.move_pos is not None:
                key = (a.move_pos[0], a.move_pos[1])
                if key not in seen_factory:
                    seen_factory.add(key)
                    factory_build.append([a.move_pos[0], a.move_pos[1]])

    elif state.action_stage == ActionStage.MOVE and state.selected_unit is not None:
        tiles = get_reachable_tiles(state, state.selected_unit)
        reachable = [[r, c] for r, c in sorted(tiles)]

    elif state.action_stage == ActionStage.ACTION and state.selected_unit is not None and sm_pos is not None:
        unit = state.selected_unit
        for a in get_legal_actions(state):
            t = a.action_type
            if t == ActionType.ATTACK:
                name = "ATTACK"
            elif t == ActionType.CAPTURE:
                name = "CAPTURE"
            elif t == ActionType.WAIT:
                name = "WAIT"
            elif t == ActionType.DIVE_HIDE:
                name = "DIVE_HIDE"
            elif t == ActionType.LOAD:
                name = "LOAD"
            elif t == ActionType.JOIN:
                name = "JOIN"
            elif t == ActionType.UNLOAD:
                name = "UNLOAD"
            elif t == ActionType.BUILD:
                name = "BUILD"
            elif t == ActionType.REPAIR:
                name = "REPAIR"
            else:
                continue
            if name not in action_options:
                action_options.append(name)
        attack_targets = [list(p) for p in get_attack_targets(state, unit, sm_pos)]
        for a in get_legal_actions(state):
            if a.action_type == ActionType.REPAIR and a.target_pos is not None:
                repair_targets.append([a.target_pos[0], a.target_pos[1]])
        seen_unload: set[tuple[tuple[int, int], int]] = set()
        for a in get_legal_actions(state):
            if (
                a.action_type == ActionType.UNLOAD
                and a.target_pos is not None
                and a.unit_type is not None
            ):
                key = (a.target_pos, int(a.unit_type))
                if key in seen_unload:
                    continue
                seen_unload.add(key)
                unload_options.append(
                    {
                        "target_pos": [a.target_pos[0], a.target_pos[1]],
                        "unit_type": a.unit_type.name,
                    }
                )

    return {
        "selectable_unit_tiles": selectable,
        "factory_build_tiles": factory_build,
        "reachable_tiles": reachable,
        "attack_targets": attack_targets,
        "repair_targets": repair_targets,
        "unload_options": unload_options,
        "action_options": action_options,
        "selected_unit_pos": list(su_pos) if su_pos else None,
        "selected_move_pos": list(sm_pos) if sm_pos else None,
    }


def build_play_payload(
    session_id: str,
    state: GameState,
    *,
    ok: bool = True,
    error: Optional[str] = None,
) -> dict[str, Any]:
    hints = _stage_hints(state)
    board = board_dict(state, include_terrain=True)
    return {
        "session_id": session_id,
        "ok": ok,
        "error": error,
        "action_stage": state.action_stage.name,
        "active_player": int(state.active_player),
        "done": bool(state.done),
        "winner": state.winner if state.done else None,
        "turn": int(state.turn),
        "funds": [int(state.funds[0]), int(state.funds[1])],
        "legal_global": _legal_global(state),
        "co_p0": _co_client_dict(state.co_states[0]),
        "co_p1": _co_client_dict(state.co_states[1]),
        "selected_unit_pos": hints["selected_unit_pos"],
        "selected_move_pos": hints["selected_move_pos"],
        "selectable_unit_tiles": hints["selectable_unit_tiles"],
        "factory_build_tiles": hints["factory_build_tiles"],
        "factory_build_menu": _factory_build_menu(state),
        "reachable_tiles": hints["reachable_tiles"],
        "attack_targets": hints["attack_targets"],
        "repair_targets": hints["repair_targets"],
        "unload_options": hints["unload_options"],
        "action_options": hints["action_options"],
        "board": board,
        "map_id": getattr(state.map_data, "map_id", None),
        "tier": getattr(state, "tier_name", None),
        "bot_mode": _session_meta.get(session_id, {}).get("bot_mode", "ppo"),
        **_session_meta_opening_book_log(session_id),
    }


def _session_meta_opening_book_log(session_id: str) -> dict[str, Any]:
    ctl = _session_meta.get(session_id, {}).get("p1_opening_book")
    return ctl.log_fields() if ctl is not None else {}


def _run_bot_turn(
    state: GameState,
    model: Optional[Any],
    *,
    p1_opening_book: Any | None = None,
) -> None:
    """Step bot until human SELECT or terminal. Caller holds game lock per session.

    ``model`` is MaskablePPO when a checkpoint loaded; ``None`` means uniform random
    over legal flat actions (same mask contract as training).
    When ``p1_opening_book`` is active (same flat schema as AWBWEnv), P1 consumes
    ``std_pool_precombat.jsonl`` lines until desync/exhaust then falls back.
    """
    while not state.done and state.active_player == BOT_PLAYER:
        mask = _get_action_mask(state)
        calendar_turn = int(getattr(state, "turn", 0) or 0)
        idx: int | None = None
        use_book = (
            p1_opening_book is not None
            and getattr(p1_opening_book, "book_id", None) is not None
            and not getattr(p1_opening_book, "desync", False)
        )
        if use_book:
            picked = p1_opening_book.suggest_flat(
                calendar_turn=calendar_turn, action_mask=mask
            )
            idx = int(picked) if picked is not None else None
        if idx is None and model is not None:
            obs_sp, obs_sc = encode_state(state, observer=BOT_PLAYER)
            obs = {"spatial": obs_sp, "scalars": obs_sc}
            with _model_lock:
                action_arr, _ = model.predict(obs, action_masks=mask, deterministic=False)
            idx = int(np.asarray(action_arr).reshape(-1)[0])
        elif idx is None:
            legal_flat = np.flatnonzero(mask)
            if legal_flat.size == 0:
                break
            idx = int(np.random.choice(legal_flat))
        action = _flat_to_action(idx, state)
        if action is None:
            legal = get_legal_actions(state)
            if not legal:
                break
            action = random.choice(legal)
        state.step(action)


def _append_human_demo(
    session_id: str,
    state: GameState,
    action: Action,
    map_id: Optional[int],
    tier: Optional[str],
) -> None:
    spatial, scalars = encode_state(state, observer=HUMAN_PLAYER)
    mask = _get_action_mask(state)
    row = {
        "encoder_version": [int(N_SPATIAL_CHANNELS), int(N_SCALARS)],
        "spatial": spatial.tolist(),
        "scalars": scalars.tolist(),
        "action_mask": mask.tolist(),
        "action_idx": int(_action_to_flat(action, state)),
        "action_stage": state.action_stage.name,
        "action_label": _action_label(action),
        "active_player": int(state.active_player),
        "map_id": map_id,
        "tier": tier,
        "session_id": session_id,
    }
    HUMAN_DEMOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_DEMOS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()  # so demos survive process kill / server stop between turns


def _parse_step_action(state: GameState, data: dict[str, Any]) -> Action:
    kind = (data.get("kind") or "").lower()
    if kind == "end_turn":
        return Action(ActionType.END_TURN)
    if kind == "cop":
        return Action(ActionType.ACTIVATE_COP)
    if kind == "scop":
        return Action(ActionType.ACTIVATE_SCOP)
    if kind == "build":
        fp = data.get("factory_pos")
        if fp is None or len(fp) != 2:
            raise ValueError("build requires factory_pos [r, c]")
        r, c = int(fp[0]), int(fp[1])
        ut_raw = data.get("unit_type")
        if ut_raw is None:
            raise ValueError("build requires unit_type")
        ut = UnitType[ut_raw] if isinstance(ut_raw, str) else UnitType(int(ut_raw))
        return Action(ActionType.BUILD, unit_pos=None, move_pos=(r, c), unit_type=ut)

    def pos(key: str) -> tuple[int, int]:
        p = data[key]
        return int(p[0]), int(p[1])

    if kind == "select_unit":
        return Action(ActionType.SELECT_UNIT, unit_pos=pos("unit_pos"))
    if kind == "move_unit":
        return Action(
            ActionType.SELECT_UNIT,
            unit_pos=pos("unit_pos"),
            move_pos=pos("move_pos"),
        )
    if kind == "wait":
        return Action(ActionType.WAIT, unit_pos=pos("unit_pos"), move_pos=pos("move_pos"))
    if kind in ("dive_hide", "dive", "hide"):
        return Action(ActionType.DIVE_HIDE, unit_pos=pos("unit_pos"), move_pos=pos("move_pos"))
    if kind == "attack":
        return Action(
            ActionType.ATTACK,
            unit_pos=pos("unit_pos"),
            move_pos=pos("move_pos"),
            target_pos=pos("target_pos"),
        )
    if kind == "capture":
        return Action(ActionType.CAPTURE, unit_pos=pos("unit_pos"), move_pos=pos("move_pos"))
    if kind == "load":
        return Action(ActionType.LOAD, unit_pos=pos("unit_pos"), move_pos=pos("move_pos"))
    if kind == "join":
        return Action(ActionType.JOIN, unit_pos=pos("unit_pos"), move_pos=pos("move_pos"))
    if kind == "repair":
        return Action(
            ActionType.REPAIR,
            unit_pos=pos("unit_pos"),
            move_pos=pos("move_pos"),
            target_pos=pos("target_pos"),
        )
    if kind == "unload":
        ut_raw = data.get("unit_type")
        if ut_raw is None:
            raise ValueError("unload requires unit_type")
        ut = UnitType[ut_raw] if isinstance(ut_raw, str) else UnitType(int(ut_raw))
        return Action(
            ActionType.UNLOAD,
            unit_pos=pos("unit_pos"),
            move_pos=pos("move_pos"),
            target_pos=pos("target_pos"),
            unit_type=ut,
        )
    raise ValueError(f"Unknown kind {kind!r}")


def _validate_action(state: GameState, action: Action) -> None:
    legal = get_legal_actions(state)
    if action not in legal:
        raise ValueError("Illegal action for current state")


def _pick_tier(meta: dict, tier_name: str) -> dict:
    for t in meta["tiers"]:
        if t.get("tier_name") == tier_name and t.get("enabled") and t.get("co_ids"):
            return t
    raise ValueError(f"No enabled tier {tier_name!r} for map {meta.get('map_id')}")


def _is_co_allowed_in_tier(meta: dict, tier_name: str, co_id: int) -> bool:
    """
    Check if a CO is allowed in a tier based on hierarchy.
    Lower tiers (numerically lower) can use COs from higher tiers.
    Based on request: T2 can use T2, T3, T4; T3 can use T3, T4; T4 can use T4.
    Tier order: TL, T0, T1, T2, T3, T4, T5 (T2 < T3 < T4)
    """
    # Parse tier number from tier_name (e.g., "T2" -> 2, "TL" -> -1, "T0" -> 0)
    if tier_name.startswith("T"):
        try:
            if tier_name[1:].isdigit():
                tier_num = int(tier_name[1:])
            elif tier_name == "TL":
                tier_num = -1  # TL is lowest
            else:
                tier_num = -2  # Unknown tier
        except ValueError:
            tier_num = -2
    else:
        tier_num = -2
    
    # Check all tiers in the map
    for tier in meta.get("tiers", []):
        tname = tier.get("tier_name", "")
        if tname.startswith("T"):
            try:
                if tname[1:].isdigit():
                    t_num = int(tname[1:])
                elif tname == "TL":
                    t_num = -1
                else:
                    t_num = -2
            except ValueError:
                t_num = -2
        else:
            t_num = -2
        
        # CO is allowed if it's in this tier AND this tier number >= requested tier number
        # (higher or equal tier number means it's a higher or equal tier)
        # Example: For T2 (tier_num=2), allow tiers with t_num >= 2 (T2, T3, T4, T5)
        if co_id in tier.get("co_ids", []) and t_num >= tier_num:
            return True
    
    return False


def new_session(
    *,
    map_id: Optional[int],
    tier: Optional[str],
    human_co_id: Optional[int],
    bot_co_id: Optional[int],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], Optional[str]]:
    """
    Create a new game. Returns (payload, error).
    If checkpoint is missing, payload still describes the failure in ``error``.
    """
    model, err = ensure_model_loaded(checkpoint_dir)
    if err is not None:
        return (
            {
                "session_id": "",
                "ok": False,
                "error": err,
            },
            err,
        )
    bot_mode = "ppo" if model is not None else "random"

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    mid = int(map_id or _DEFAULT_MAP_ID)
    meta = next((m for m in pool if m["map_id"] == mid), None)
    if meta is None:
        msg = f"Unknown map_id {mid}"
        return {"session_id": "", "ok": False, "error": msg}, msg

    tier_name = tier or _DEFAULT_TIER
    try:
        tier_obj = _pick_tier(meta, tier_name)
    except ValueError as e:
        msg = str(e)
        return {"session_id": "", "ok": False, "error": msg}, msg
    co_ids: list[int] = tier_obj["co_ids"]

    p0 = int(human_co_id or _DEFAULT_HUMAN_CO)
    p1 = int(bot_co_id or _DEFAULT_BOT_CO)
    if not _is_co_allowed_in_tier(meta, tier_name, p0):
        msg = f"human co_id {p0} not allowed in tier {tier_name}"
        return {"session_id": "", "ok": False, "error": msg}, msg
    if not _is_co_allowed_in_tier(meta, tier_name, p1):
        msg = f"bot co_id {p1} not allowed in tier {tier_name}"
        return {"session_id": "", "ok": False, "error": msg}, msg

    map_data = load_map(mid, POOL_PATH, MAPS_DIR)
    _mkp: dict = {"starting_funds": 0, "tier_name": tier_name}
    rfm = getattr(map_data, "replay_first_mover", None)
    if rfm is not None:
        _mkp["replay_first_mover"] = int(rfm)
    state = make_initial_state(map_data, p0, p1, **_mkp)

    sid = str(uuid.uuid4())
    p1_book = _spawn_p1_opening_book_ctl(mid, p1, sid)
    display_bot_mode = f"book+{bot_mode}" if p1_book is not None else bot_mode

    with _session_io_lock:
        _sessions[sid] = state
        _session_meta[sid] = {
            "map_id": mid,
            "tier": tier_name,
            "p0_co": p0,
            "p1_co": p1,
            "bot_mode": display_bot_mode,
            "p1_opening_book": p1_book,
        }

        # make_initial_state can open on P1 (asymmetric predeploy); run the bot first so the human view is always on P0's clock.
        if state.active_player == BOT_PLAYER:
            _run_bot_turn(state, model, p1_opening_book=p1_book)

    payload = build_play_payload(sid, state, ok=True, error=None)
    return payload, None


def get_session_state(session_id: str) -> tuple[dict[str, Any], Optional[str]]:
    state = _sessions.get(session_id)
    if state is None:
        return {
            "session_id": session_id,
            "ok": False,
            "error": "Unknown session_id",
        }, "Unknown session_id"
    return build_play_payload(session_id, state), None


def cancel_selection(session_id: str) -> tuple[dict[str, Any], Optional[str]]:
    with _session_io_lock:
        state = _sessions.get(session_id)
        if state is None:
            return {
                "session_id": session_id,
                "ok": False,
                "error": "Unknown session_id",
            }, "Unknown session_id"
        if state.active_player != HUMAN_PLAYER:
            return build_play_payload(session_id, state, ok=False, error="Not human turn"), "Not human turn"
        state.selected_unit = None
        state.selected_move_pos = None
        state.action_stage = ActionStage.SELECT
        return build_play_payload(session_id, state), None


def apply_human_step(
    session_id: str,
    data: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], Optional[str]]:
    with _session_io_lock:
        state = _sessions.get(session_id)
        if state is None:
            return {
                "session_id": session_id,
                "ok": False,
                "error": "Unknown session_id",
            }, "Unknown session_id"
        if state.done:
            return build_play_payload(session_id, state, ok=False, error="Game already finished"), "done"
        if state.active_player != HUMAN_PLAYER:
            return build_play_payload(session_id, state, ok=False, error="Not human turn"), "turn"

        model, err = ensure_model_loaded(checkpoint_dir)
        if err is not None:
            return build_play_payload(session_id, state, ok=False, error=err), err

        try:
            action = _parse_step_action(state, data)
        except ValueError as e:
            return build_play_payload(session_id, state, ok=False, error=str(e)), str(e)

        try:
            _validate_action(state, action)
        except ValueError as e:
            return build_play_payload(session_id, state, ok=False, error=str(e)), str(e)

        meta = _session_meta.get(session_id, {})
        _append_human_demo(session_id, state, action, meta.get("map_id"), meta.get("tier"))

        state.step(action)

        if not state.done and state.active_player == BOT_PLAYER:
            ob = _session_meta.get(session_id, {}).get("p1_opening_book")
            _run_bot_turn(state, model, p1_opening_book=ob)

        return build_play_payload(session_id, state), None
