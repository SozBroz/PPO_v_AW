# -*- coding: utf-8 -*-
"""Phase 11f: MCTS sim-budget escalator.

Once Phase 11d health gate flips a machine to --mcts-mode eval_only at sims=16,
this module decides per-cycle whether to double sims (16→32→64→128), hold,
drop back to off (on engine desync), or stop and ask operator (at 128 with
positive ROI).

Pure library — no orchestrator wiring this session. Future composer wires
compute_sims_proposal() into either curriculum_advisor.py or
fleet_orchestrator.py once first real MCTS telemetry exists.

Alignment: :class:`rl.mcts.MCTSConfig` uses ``num_sims`` (train CLI often
``--mcts-sims``). Escalator ``current_sims`` / ``proposed_sims`` match that
budget.

# Integration plan (next session, NOT this composer):
#
# 1. Wire into scripts/fleet_orchestrator.py per-tick loop AFTER read_mcts_health:
#    - If mcts_health.pass_overall AND mcts_mode != "off":
#        - cycle = build_cycle_result_from_recent_eval_jsons(machine_id)
#        - proposal = compute_sims_proposal(state_path, cycle)
#        - if proposal.action == DOUBLE: write proposed --mcts-sims into proposed_args.json
#        - if proposal.action == DROP_TO_OFF: write --mcts-mode off + audit alert
#        - always: append_cycle_log + write proposed to audit log
#
# 2. Build cycle result from existing fleet/<id>/eval/*.json (Composer G's mcts_health
#    pattern) plus a new field engine_desyncs_in_cycle from desync_audit results.
#
# 3. Update plan §11f frontmatter: phase-11f-shipped + ship date.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_state_path(machine_id: str, shared_root: Path | None = None) -> Path:
    root = REPO_ROOT if shared_root is None else shared_root
    return root / "fleet" / machine_id / "mcts_escalator_state.json"


def default_cycle_log_path(shared_root: Path | None = None) -> Path:
    root = REPO_ROOT if shared_root is None else shared_root
    return root / "logs" / "mcts_escalator.jsonl"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EscalatorAction(str, Enum):
    HOLD = "hold"
    DOUBLE = "double"
    DROP_TO_OFF = "drop_to_off"
    STOP_ASK_OPERATOR = "stop_ask_operator"


@dataclass(slots=True)
class EscalatorCycleResult:
    """One row for ``logs/mcts_escalator.jsonl``."""

    cycle_ts: float
    sims: int
    winrate_vs_pool: float
    mcts_off_baseline: float
    games_decided: int
    explained_variance: float
    engine_desyncs_in_cycle: int
    wall_s_per_decision_p50: float


@dataclass(slots=True)
class EscalatorState:
    current_sims: int
    mcts_off_baseline: float
    last_double_at_ts: float
    sims_plateau_at: int | None
    cycles_at_current_sims: int


@dataclass(slots=True)
class EscalatorThresholds:
    min_winrate_lift: float = 0.02
    min_games_decided: int = 200
    min_explained_variance: float = 0.6
    regress_explained_variance: float = 0.55
    max_sims_auto: int = 128
    min_cycles_at_sims_before_double: int = 1


DEFAULT_THRESHOLDS = EscalatorThresholds()


@dataclass(slots=True)
class EscalatorProposal:
    action: EscalatorAction
    proposed_sims: int
    reason: str
    state_after: EscalatorState
    cycle_metrics: EscalatorCycleResult


def _default_escalator_state() -> EscalatorState:
    return EscalatorState(
        current_sims=16,
        mcts_off_baseline=0.0,
        last_double_at_ts=0.0,
        sims_plateau_at=None,
        cycles_at_current_sims=0,
    )


def _lift(cycle: EscalatorCycleResult) -> float:
    return float(cycle.winrate_vs_pool) - float(cycle.mcts_off_baseline)


def _meets_double_evidence(
    cycle: EscalatorCycleResult, thresholds: EscalatorThresholds
) -> bool:
    lift = _lift(cycle)
    return (
        lift >= thresholds.min_winrate_lift
        and int(cycle.games_decided) >= thresholds.min_games_decided
        and float(cycle.explained_variance) >= thresholds.min_explained_variance
    )


def decide_action(
    state: EscalatorState,
    latest_cycle: EscalatorCycleResult,
    thresholds: EscalatorThresholds = DEFAULT_THRESHOLDS,
) -> EscalatorProposal:
    """Pure decision: given persisted state and one cycle's metrics, next action."""
    lift = _lift(latest_cycle)
    ev = float(latest_cycle.explained_variance)
    min_c = int(thresholds.min_cycles_at_sims_before_double)

    if int(latest_cycle.engine_desyncs_in_cycle) > 0:
        st = replace(
            state,
            current_sims=0,
            cycles_at_current_sims=0,
        )
        return EscalatorProposal(
            action=EscalatorAction.DROP_TO_OFF,
            proposed_sims=0,
            reason="engine desync in cycle; drop MCTS (caller sets --mcts-mode off)",
            state_after=st,
            cycle_metrics=latest_cycle,
        )

    if (
        int(state.current_sims) == int(thresholds.max_sims_auto)
        and _meets_double_evidence(latest_cycle, thresholds)
    ):
        st = replace(
            state,
            cycles_at_current_sims=int(state.cycles_at_current_sims) + 1,
        )
        cap = int(thresholds.max_sims_auto)
        return EscalatorProposal(
            action=EscalatorAction.STOP_ASK_OPERATOR,
            proposed_sims=cap,
            reason=(
                f"at max auto sims={cap} with ROI gates passed; operator approval required "
                "to raise further"
            ),
            state_after=st,
            cycle_metrics=latest_cycle,
        )

    if int(state.cycles_at_current_sims) < min_c:
        st = replace(
            state,
            cycles_at_current_sims=int(state.cycles_at_current_sims) + 1,
        )
        return EscalatorProposal(
            action=EscalatorAction.HOLD,
            proposed_sims=int(state.current_sims),
            reason=f"warming up at sims={int(state.current_sims)}",
            state_after=st,
            cycle_metrics=latest_cycle,
        )

    if _meets_double_evidence(latest_cycle, thresholds) and int(state.current_sims) < int(
        thresholds.max_sims_auto
    ):
        new_sims = int(state.current_sims) * 2
        new_sims = min(new_sims, int(thresholds.max_sims_auto))
        st = replace(
            state,
            current_sims=new_sims,
            cycles_at_current_sims=0,
            last_double_at_ts=float(latest_cycle.cycle_ts),
            sims_plateau_at=None,
        )
        return EscalatorProposal(
            action=EscalatorAction.DOUBLE,
            proposed_sims=new_sims,
            reason=(
                f"winrate lift {lift:.4f}>={thresholds.min_winrate_lift}, "
                f"games>={thresholds.min_games_decided}, "
                f"EV>={thresholds.min_explained_variance}; double to {new_sims}"
            ),
            state_after=st,
            cycle_metrics=latest_cycle,
        )

    if ev < float(thresholds.regress_explained_variance) or lift < 0.0:
        st = replace(
            state,
            cycles_at_current_sims=int(state.cycles_at_current_sims) + 1,
            sims_plateau_at=int(state.current_sims),
        )
        parts: list[str] = []
        if ev < float(thresholds.regress_explained_variance):
            parts.append(f"explained_variance {ev:.3f} < {thresholds.regress_explained_variance}")
        if lift < 0.0:
            parts.append(f"winrate_vs_pool below baseline (lift {lift:.4f})")
        return EscalatorProposal(
            action=EscalatorAction.HOLD,
            proposed_sims=int(state.current_sims),
            reason="hold; plateau at sims=%d (%s)" % (int(state.current_sims), "; ".join(parts)),
            state_after=st,
            cycle_metrics=latest_cycle,
        )

    st = replace(
        state,
        cycles_at_current_sims=int(state.cycles_at_current_sims) + 1,
    )
    return EscalatorProposal(
        action=EscalatorAction.HOLD,
        proposed_sims=int(state.current_sims),
        reason="hold; ROI gates not met for double",
        state_after=st,
        cycle_metrics=latest_cycle,
    )


def read_state(path: Path) -> EscalatorState:
    if not path.is_file():
        return _default_escalator_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_escalator_state()
    try:
        plateau = raw.get("sims_plateau_at")
        plateau_i: int | None
        if plateau is None:
            plateau_i = None
        else:
            plateau_i = int(plateau)
        return EscalatorState(
            current_sims=int(raw["current_sims"]),
            mcts_off_baseline=float(raw["mcts_off_baseline"]),
            last_double_at_ts=float(raw["last_double_at_ts"]),
            sims_plateau_at=plateau_i,
            cycles_at_current_sims=int(raw["cycles_at_current_sims"]),
        )
    except (KeyError, TypeError, ValueError):
        return _default_escalator_state()


def write_state(path: Path, state: EscalatorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_state_to_jsonable(state), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix="mcts_escalator_state_", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _state_to_jsonable(state: EscalatorState) -> dict[str, Any]:
    d = asdict(state)
    return d


def append_cycle_log(path: Path, cycle: EscalatorCycleResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(cycle), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def compute_sims_proposal(
    state_path: Path,
    cycle: EscalatorCycleResult,
    *,
    log_path: Path | None = None,
    apply: bool = False,
    thresholds: EscalatorThresholds = DEFAULT_THRESHOLDS,
) -> EscalatorProposal:
    """Load state, decide, optionally persist ``state_after`` and append JSONL row."""
    state = read_state(state_path)
    proposal = decide_action(state, cycle, thresholds=thresholds)
    if apply:
        write_state(state_path, proposal.state_after)
        if log_path is not None:
            append_cycle_log(log_path, cycle)
    return proposal


def _parse_simulate_kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _proposal_to_jsonable(p: EscalatorProposal) -> dict[str, Any]:
    return {
        "action": p.action.value,
        "proposed_sims": p.proposed_sims,
        "reason": p.reason,
        "state_after": asdict(p.state_after),
        "cycle_metrics": asdict(p.cycle_metrics),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MCTS sim-budget escalator (Phase 11f).")
    parser.add_argument("--machine-id", default="pc-b", help="Fleet machine id (default pc-b)")
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help="Repo or fleet root (default: parent of tools/)",
    )
    parser.add_argument(
        "--simulate-cycle",
        nargs="+",
        metavar="KEY=VAL",
        help="e.g. winrate=0.55 ev=0.7 desyncs=0 games=250 baseline=0.45",
    )
    args = parser.parse_args(argv)
    root = Path(args.shared_root) if args.shared_root is not None else REPO_ROOT
    st_path = default_state_path(args.machine_id, shared_root=root)

    if not args.simulate_cycle:
        parser.error("--simulate-cycle is required for inspection")
    kv = _parse_simulate_kv(list(args.simulate_cycle))
    state = read_state(st_path)

    def _f(key: str, default: str) -> float:
        return float(kv.get(key, default))

    def _i(key: str, default: str) -> int:
        return int(kv.get(key, default))

    cycle = EscalatorCycleResult(
        cycle_ts=_f("ts", str(time.time())),
        sims=_i("sims", str(state.current_sims)),
        winrate_vs_pool=_f("winrate", "0.0"),
        mcts_off_baseline=_f("baseline", str(state.mcts_off_baseline)),
        games_decided=_i("games", "0"),
        explained_variance=_f("ev", "0.0"),
        engine_desyncs_in_cycle=_i("desyncs", "0"),
        wall_s_per_decision_p50=_f("wall_p50", "0.0"),
    )
    proposal = decide_action(state, cycle)
    print(json.dumps(_proposal_to_jsonable(proposal), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
