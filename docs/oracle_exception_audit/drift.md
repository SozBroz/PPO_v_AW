# Thread DRIFT — verdicts

Audit of `tools/oracle_zip_replay.py` drift-spawn family (Phase 1 of `desync_purge_engine_harden` campaign). STRICT bar: only AWBW-canon citations earn KEEP.

## Site: `_oracle_drift_spawn_unloaded_cargo` (`tools/oracle_zip_replay.py`:851–1055)

**Verdict:** DELETE
**Justification:** Fabricates unload outcomes (relocating by AWBW id, teleporting carriers, zeroing enemy HP on the drop tile, Manhattan≤8 carrier heuristics, spawning drift cargo) when the engine no longer matches AWBW. That reconciles broken or divergent simulation state, not a citable Advance Wars / AWBW site rule. `docs/desync_audit.md` warns against auto-spawning units to mask bugs. Docstring cites operational scenarios only, not wiki/Fandom canon.
**Replacement message:** `raise UnsupportedOracleAction("Drift spawn unloaded cargo: engine transport/cargo state diverged from AWBW; fix load/unload resolution or sync instead of fabricating unload outcome")`

---

## Site: `_oracle_drift_spawn_mover_from_global` (`tools/oracle_zip_replay.py`:1058–1203)

**Verdict:** REPLACE-WITH-ENGINE-FIX
**Justification:** Explicitly a "last resort" when the engine has "fully lost" the mover after upstream drift; also zeros enemy HP and teleports friendly carriers to free path-start. Repo test `tests/test_oracle_move_no_unit_drift_spawn.py` ties the carrier teleport + cargo spawn to game **1632702** — a Move/Load vs engine consolidation issue that belongs in envelope resolution, not inventing units.
**Engine fix target:** `tools/oracle_zip_replay.py`: `_apply_move_paths_then_terminator` and the `Move`/`Load` path that consumes `unit.global` — resolve the mover (and load geometry) so the unit exists before stepping; if drift is luck/state only, use `tools/oracle_state_sync.py` per `docs/desync_audit.md` instead of spawn. Supplement only if proven: `engine/game.py::_apply_load` / `_apply_unload`.

---

## Site: `_oracle_drift_spawn_capturer_for_property` (`tools/oracle_zip_replay.py`:1206–1269)

**Verdict:** DELETE
**Justification:** Spawns Infantry on an empty property so a no-path `Capt` can run when no capturer exists — pure stream continuation after drift, with no quoted AWBW rules citation.
**Replacement message:** `raise UnsupportedOracleAction("Capt no-path: no capturer on property; drift spawn capturer disabled")`

---

## Site: `_oracle_drift_spawn_mover_from_global` call site (`tools/oracle_zip_replay.py`:4585–4589)

**Verdict:** REPLACE-WITH-ENGINE-FIX
**Justification:** Sole in-range hook for mover drift inside `_apply_move_paths_then_terminator`; deleting the helper without fixing resolution only increases `Move: no unit` failures. Same remediation as the helper: correct mover resolution from AWBW identity before the path walk.
**Engine fix target:** `tools/oracle_zip_replay.py::_apply_move_paths_then_terminator` — ensure the mover is located or state is synchronized so `gu` / `units_id` maps to a real `Unit` without drift spawn.

---

## Site: `Capt (no path)` — `capture_points` early return (`tools/oracle_zip_replay.py`:6162–6174)

**Verdict:** DELETE (escalation resolved 2026-04-20 by commander — replays are ground truth, no silent capture-progress writes without a real engine capturer action)
**Justification:** With no capturer `u`, this sets `ph.capture_points` from the building snapshot and returns — without `GameState._apply_capture`. Commander ruled this is silent state fudging: if the engine has no capturer bound, the oracle must surface the gap, not paper over it from PHP.
**Replacement message:** `raise UnsupportedOracleAction("Capt no-path: no engine capturer bound; refuse to copy capture_points from PHP snapshot")`

---

## Site: `Capt (no path)` — `_oracle_drift_spawn_capturer_for_property` (`tools/oracle_zip_replay.py`:6175–6183)

**Verdict:** DELETE
**Justification:** Default Infantry spawn to absorb `Capt` when all pools failed — same drift-spawn class as unload/mover; no AWBW canonical citation.
**Replacement message:** `raise UnsupportedOracleAction("Capt no-path: drift spawn capturer disabled; no reachable capturer for property")`

---

## Site: `Unload` handler — drift call sites (`tools/oracle_zip_replay.py`:7035–7190)

**Verdict:** DELETE
**Justification:** (7035–7047) Catches `_resolve_unload_transport` only when message contains `"no transport adjacent"` and runs `_oracle_drift_spawn_unloaded_cargo`. (7165–7183) Second drift when no legal `UNLOAD` exists in `get_legal_actions` but stage allows it. Both paper over mismatch with `engine/game.py::_apply_unload` (orthogonal adjacent drop, empty tile, cargo in `loaded_units`). No AWBW rules quote supports spawning cargo without a valid unload chain.
**Replacement message:** `raise UnsupportedOracleAction("Unload: drift recovery disabled; transport/target/loaded cargo do not support UNLOAD—fix resolver or engine state")`

---

## ESCALATIONS (resolved)

1. **`Capt (no path)` — `capture_points` direct write (6162–6174):** **RESOLVED 2026-04-20 → DELETE.** Commander: replays are ground truth; no silent capture-progress writes without a real engine capturer action.

---

## Summary

| Verdict | Count |
|---------|------:|
| KEEP | 0 |
| DELETE | 5 |
| REPLACE-WITH-ENGINE-FIX | 2 |
| ESCALATE | 0 (all resolved) |
