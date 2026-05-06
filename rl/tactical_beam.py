"""Tactical beam search for RHEA."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence, Tuple, Optional

from engine.game import GameState
from engine.search_clone import clone_for_search
from rl.candidate_actions import CandidateAction, CandidateKind, candidate_arrays, MAX_CANDIDATES
from rl.rhea_fitness import RheaFitness, RheaFitnessBreakdown

# Cython accelaration
USE_CYTHON_TB = True
try:
    from rl._tactical_beam_cython import (
        bucket_for_candidate_cython,
        juicy_score_cython,
        dynamic_budget_cython,
        dedupe_cython as dedupe_cython_fast,
    )
except ImportError:
    USE_CYTHON_TB = False


@dataclass(slots=True)
class TacticalBeamConfig:
    enabled: bool = True
    min_width: int = 8
    max_width: int = 48
    min_depth: int = 3
    max_depth: int = 14
    max_candidates_per_expand: int = 24
    max_finish_capture: int = 16
    max_start_capture: int = 16
    max_killshot: int = 20
    max_strike: int = 20
    max_build: int = 8
    max_power: int = 4
    max_position: int = 8
    partial_phi_weight: float = 1.0
    partial_value_weight: float = 0.0
    dedupe_states: bool = True
    dedupe_plans: bool = True


@dataclass(slots=True)
class BeamLine:
    state: GameState
    actions: list
    score: float
    breakdown: Optional[RheaFitnessBreakdown] = None
    tactical_count: int = 0
    buckets_seen: set[str] = field(default_factory=set)


@dataclass(slots=True)
class TacticalBeamResult:
    lines: list[BeamLine]
    width: int
    depth: int
    initial_juicy_count: int
    bucket_counts: dict[str, int]


class TacticalBeamPlanner:
    def __init__(self, fitness: RheaFitness, config: Optional[TacticalBeamConfig] = None) -> None:
        self.fitness = fitness
        self.cfg = config or TacticalBeamConfig()

    def search(self, root: GameState) -> TacticalBeamResult:
        acting = int(root.active_player)
        initial = self._juicy_candidates(root)
        budget = self._dynamic_budget(root, initial)

        if not self.cfg.enabled or budget["width"] <= 0 or budget["depth"] <= 0 or not initial:
            return TacticalBeamResult(
                lines=[],
                width=0,
                depth=0,
                initial_juicy_count=len(initial),
                bucket_counts=self._bucket_counts(initial),
            )

        beam = [BeamLine(state=clone_for_search(root), actions=[], score=0.0)]
        best_lines: list[BeamLine] = []

        for _depth_idx in range(budget["depth"]):
            expanded: list[BeamLine] = []

            for line in beam:
                if line.state.winner is not None or int(line.state.active_player) != acting:
                    best_lines.append(line)
                    continue

                cands = self._juicy_candidates(line.state)
                if not cands:
                    best_lines.append(line)
                    continue

                for cand in cands[: budget["expand"]]:
                    child_state = clone_for_search(line.state)
                    ok = self._apply_candidate(child_state, cand)
                    if not ok:
                        continue

                    child_actions = list(line.actions)
                    child_actions.append(cand.first)
                    if cand.second is not None:
                        child_actions.append(cand.second)

                    bucket = self._bucket_for_candidate(cand)
                    buckets_seen = set(line.buckets_seen)
                    buckets_seen.add(bucket)

                    br = self.fitness.score(
                        root, child_state, observer_seat=acting, illegal_genes=0
                    )
                    score = (
                        self.cfg.partial_phi_weight * br.phi_delta
                        + self.cfg.partial_value_weight * br.value
                    )

                    expanded.append(
                        BeamLine(
                            state=child_state,
                            actions=child_actions,
                            score=score,
                            breakdown=br,
                            tactical_count=line.tactical_count + 1,
                            buckets_seen=buckets_seen,
                        )
                    )

            if not expanded:
                break

            expanded = self._dedupe(expanded)
            expanded.sort(key=lambda x: self._beam_sort_key(x), reverse=True)
            beam = expanded[: budget["width"]]
            best_lines.extend(beam[: max(1, budget["width"] // 4)])

        all_lines = self._dedupe(best_lines + beam)
        all_lines.sort(key=lambda x: self._beam_sort_key(x), reverse=True)
        return TacticalBeamResult(
            lines=all_lines[: budget["width"]],
            width=budget["width"],
            depth=budget["depth"],
            initial_juicy_count=len(initial),
            bucket_counts=self._bucket_counts(initial),
        )

    def _dynamic_budget(
        self, state: GameState, initial: Sequence[CandidateAction]
    ) -> dict[str, int]:
        if USE_CYTHON_TB:
            counts = self._bucket_counts(initial)
            owned_units = self._owned_unit_count(state, int(state.active_player))
            co_state = state.co_states[int(state.active_player)]
            cop_ready = getattr(co_state, "cop_ready", False)
            scop_ready = getattr(co_state, "scop_ready", False)
            juicy = len(initial)
            return dynamic_budget_cython(
                owned_units,
                juicy,
                counts,
                cop_ready,
                scop_ready,
                self.cfg.min_width,
                self.cfg.max_width,
                self.cfg.min_depth,
                self.cfg.max_depth,
                4,  # min_expand: floor for expand (matches Python logic)
                self.cfg.max_candidates_per_expand,  # max_expand
            )
        
        acting = int(state.active_player)
        owned_units = self._owned_unit_count(state, acting)
        counts = self._bucket_counts(initial)
        juicy = len(initial)
        active_buckets = sum(1 for v in counts.values() if v > 0)

        complexity = (
            0.40 * owned_units
            + 1.20 * counts.get("finish_capture", 0)
            + 1.00 * counts.get("start_capture", 0)
            + 1.25 * counts.get("killshot", 0)
            + 0.90 * counts.get("strike", 0)
            + 0.60 * counts.get("build", 0)
            + 0.80 * counts.get("power", 0)
            + 0.50 * active_buckets
        )

        co_state = state.co_states[acting]
        cop_ready = getattr(co_state, "cop_ready", False)
        scop_ready = getattr(co_state, "scop_ready", False)
        if cop_ready or scop_ready:
            complexity += 3.0

        width = int(round(self.cfg.min_width + 1.25 * complexity))
        depth = int(round(self.cfg.min_depth + math.sqrt(max(0.0, complexity)) / 1.35))
        expand = int(round(8 + 0.60 * juicy + 0.25 * owned_units))

        return {
            "width": min(self.cfg.max_width, max(self.cfg.min_width, width)),
            "depth": min(self.cfg.max_depth, max(self.cfg.min_depth, depth)),
            "expand": min(self.cfg.max_candidates_per_expand, max(4, expand)),
        }

    def _juicy_candidates(self, state: GameState) -> list[CandidateAction]:
        _feats, mask, cands = candidate_arrays(state, max_candidates=MAX_CANDIDATES)
        legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]

        buckets: dict[str, list[Tuple[float, CandidateAction]]] = {
            "finish_capture": [],
            "start_capture": [],
            "killshot": [],
            "strike": [],
            "build": [],
            "power": [],
            "position": [],
        }

        for cand in legal:
            bucket = self._bucket_for_candidate(cand)
            if bucket not in buckets:
                continue
            score = self._juicy_score(cand, bucket)
            if score <= 0.0:
                continue
            buckets[bucket].append((score, cand))

        caps = {
            "finish_capture": self.cfg.max_finish_capture,
            "start_capture": self.cfg.max_start_capture,
            "killshot": self.cfg.max_killshot,
            "strike": self.cfg.max_strike,
            "build": self.cfg.max_build,
            "power": self.cfg.max_power,
            "position": self.cfg.max_position,
        }

        selected: list[Tuple[float, CandidateAction]] = []
        for bucket, scored in buckets.items():
            scored.sort(key=lambda x: x[0], reverse=True)
            selected.extend(scored[: caps[bucket]])

        selected.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in selected]

    def _bucket_for_candidate(self, cand: CandidateAction) -> str:
        if USE_CYTHON_TB:
            return bucket_for_candidate_cython(cand, CandidateKind)
        f = cand.preview
        kind = cand.kind
        
        # Safely get terminal action name
        terminal_name = None
        if cand.terminal_action is not None:
            action_type = getattr(cand.terminal_action, 'action_type', None)
            if action_type is not None:
                terminal_name = getattr(action_type, 'name', None)
            # Fallback: check if terminal_action itself has a name
            if terminal_name is None:
                terminal_name = getattr(cand.terminal_action, 'name', None)

        if kind == CandidateKind.POWER:
            return "power"

        if terminal_name == "BUILD" or kind == CandidateKind.BUILD:
            return "build"

        if f is not None:
            capture_progress = float(f[8]) if len(f) > 8 else 0.0
            capture_completes = float(f[10]) if len(f) > 10 else 0.0
            target_killed_max = float(f[21]) if len(f) > 21 else 0.0
            enemy_removed_max = float(f[17]) if len(f) > 17 else 0.0

            if capture_completes > 0.0:
                return "finish_capture"
            if capture_progress > 0.0:
                return "start_capture"
            if target_killed_max > 0.0:
                return "killshot"
            if enemy_removed_max > 0.0:
                return "strike"

        if terminal_name == "CAPTURE":
            return "start_capture"
        if terminal_name == "ATTACK":
            return "strike"

        if kind == CandidateKind.MOVE_WAIT:
            return "position"

        return "other"

    def _juicy_score(self, cand: CandidateAction, bucket: str) -> float:
        if USE_CYTHON_TB:
            return juicy_score_cython(cand, bucket)
        if cand is None or cand.preview is None:
            return 1.0 if bucket in {"build", "position", "power"} else 0.0

        f = cand.preview

        def at(i: int) -> float:
            return float(f[i]) if len(f) > i else 0.0

        if bucket == "finish_capture":
            return 10.0 * at(10) + 2.0 * at(11) + at(8)
        if bucket == "start_capture":
            return 4.0 * at(8) + 1.5 * at(11)
        if bucket == "killshot":
            return 6.0 * at(21) + at(17) - 0.75 * at(19) - 2.0 * at(22)
        if bucket == "strike":
            return at(17) + 0.5 * at(16) - 0.75 * at(19) - at(22)
        if bucket == "build":
            return 2.0
        if bucket == "power":
            return 8.0
        if bucket == "position":
            return 0.25
        return 0.0

    def _bucket_counts(self, cands: Sequence[CandidateAction]) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in cands:
            b = self._bucket_for_candidate(c)
            out[b] = out.get(b, 0) + 1
        return out

    def _beam_sort_key(self, line: BeamLine) -> Tuple[float, int, int]:
        return (line.score, line.tactical_count, len(line.buckets_seen))

    def _dedupe(self, lines: Iterable[BeamLine]) -> list[BeamLine]:
        out: list[BeamLine] = []
        seen: set[tuple] = set()
        for line in lines:
            key_parts: list = []
            if self.cfg.dedupe_plans:
                key_parts.append(tuple(self._action_signature(a) for a in line.actions))
            if self.cfg.dedupe_states:
                key_parts.append(self._state_signature(line.state))
            key = tuple(key_parts)
            if key in seen:
                continue
            seen.add(key)
            out.append(line)
        return out

    def _state_signature(self, state: GameState) -> tuple:
        units = []
        for u in getattr(state, "units", []):
            units.append((
                getattr(u, "id", None),
                getattr(u, "owner", None),
                getattr(u, "x", None),
                getattr(u, "y", None),
                getattr(u, "hp", None),
            ))
        units.sort()
        return (
            int(getattr(state, "active_player", -1)),
            tuple(units),
            getattr(state, "turn", None),
        )

    def _action_signature(self, action) -> tuple:
        return (
            getattr(action, "action_type", None).name
            if getattr(action, "action_type", None) is not None
            else None,
            getattr(action, "unit_id", None),
            getattr(action, "x", None),
            getattr(action, "y", None),
            getattr(action, "target_x", None),
            getattr(action, "target_y", None),
            getattr(action, "build_unit_type", None),
        )

    def _apply_candidate(self, state: GameState, cand: CandidateAction) -> bool:
        try:
            state.step(cand.first)
            if cand.second is not None and state.winner is None:
                state.step(cand.second)
            return True
        except Exception:
            return False

    def _owned_unit_count(self, state: GameState, seat: int) -> int:
        return sum(1 for u in getattr(state, "units", []) if int(getattr(u, "owner", -999)) == seat)
