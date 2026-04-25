# Phase 11J-DELETE-GUARD-PIN — Lock the RL bot out of the Delete Unit action

**Date:** 2026-04-21
**Scope:** `engine/action.py` (comment banner + import-time assertion),
`tests/test_no_delete_action_legality.py` (new, 5 tests),
`docs/oracle_exception_audit/CAMPAIGN_SUMMARY.md` (Locked-in Safeguards section).
**Sister threads (read-only on their files):** SONJA-D2D-SHIP (`engine/combat.py`,
`engine/co.py`), RACHEL-SCOP-COVERING-FIRE (`engine/game.py`,
`tools/oracle_zip_replay.py`), L2-BUILD-OCCUPIED-SHIP (the just-shipped
oracle Delete handler that triggered this pin).

---

## 1. Strategic concern

AWBW players can issue a **Delete unit** action via the Game Page UI to scrap
their own unit on the spot (no funds refund). The replay oracle reproduces
that envelope through `tools/oracle_zip_replay.py::_oracle_kill_friendly_unit`
(shipped as Phase 11J-L2-BUILD-OCCUPIED-SHIP) so AWBW zip replays where a
player scraps a blocker to free a production tile can be reconstructed
faithfully.

The RL bot must **NEVER** be able to emit this action. Allowing it would
unlock a degenerate scrap-and-rebuild policy:

> Scrap a low-value blocker on a production tile → spawn a stronger
> replacement on the same tile → repeat next turn.

That loop lets the policy print arbitrary value out of the production system
without paying the opportunity cost AWBW intends. Every reasonable RL signal
(territory, force composition, expected-value of attacks) becomes vulnerable
to the trivial reward-hack of trading away cheap units for stronger ones at
will. The action belongs in the oracle path only, where AWBW's own
recorded envelope drives it.

## 2. Current state verification (pre-pin)

- `engine/action.py::ActionType` enum has **no DELETE-shaped member**. The 14
  members are `SELECT_UNIT, END_TURN, ACTIVATE_COP, ACTIVATE_SCOP, ATTACK,
  CAPTURE, WAIT, LOAD, UNLOAD, BUILD, REPAIR, JOIN, DIVE_HIDE, RESIGN`. The
  bot literally cannot construct a Delete action today.
- `_oracle_kill_friendly_unit` lives only in `tools/oracle_zip_replay.py`
  (line 740). Engine code never imports `tools.oracle_zip_replay`; AST scan
  across `engine/**/*.py` returns zero references to the helper or the
  module name (the test scaffolding is the runtime guarantee).
- `get_legal_actions(state)` cannot return a Delete action because there is
  no enum value to construct one with — the legal-set generator only
  produces members of `ActionType`.
- `GameState.step` (line 293) gates every non-oracle call against
  `get_legal_actions(self)`; any action not in that mask raises
  `IllegalActionError` (subclass of `ValueError`).

The contract is correct today. This phase **pins** it so a future refactor
cannot break it quietly.

## 3. Pin description

### 3a. Comment banner (engine/action.py)

A multi-line `#`-comment block immediately above the `ActionType` enum
declaring the RL legality contract: enum is the complete RL action space,
Delete is intentionally absent, the oracle helper is the only legal path
for Delete reproduction, and any addition of a DELETE member requires
Imperator approval. Cross-references `tests/test_no_delete_action_legality.py`.

### 3b. Import-time assertion (engine/action.py, end of file)

```python
_FORBIDDEN_RL_ACTION_NAMES = {
    "DELETE", "DELETE_UNIT", "SCRAP", "SCRAP_UNIT",
    "DESTROY_OWN_UNIT", "KILL_OWN_UNIT",
}
_existing = {m.name for m in ActionType}
_collision = _FORBIDDEN_RL_ACTION_NAMES & _existing
assert not _collision, (
    f"ActionType contains forbidden RL action(s): {_collision}. "
    f"See Phase 11J-DELETE-GUARD-PIN. Delete must remain oracle-only."
)
```

Cheap, fires the moment `engine.action` is imported. Every Python process
that touches the engine — RL training, agents, fuzzer, tests, server —
breaks immediately if a forbidden member is added.

### 3c. Regression tests (`tests/test_no_delete_action_legality.py`, 5 tests)

| # | Test | What it pins |
|---|------|--------------|
| 1 | `test_action_type_enum_has_no_delete_member` | Mirrors the import-time assertion in CI. Forbidden-set ∩ enum-members must be empty. |
| 2 | `test_get_legal_actions_never_returns_delete_across_random_states` | Sweeps **200** random small `GameState`s (3-6 wide, 3-6 tall, 0-4 units per player, varied owned bases, funds, active player) through `get_legal_actions` and asserts no member's `action_type.name` is in the forbidden set. Property-style backstop. |
| 3 | `test_step_rejects_synthetic_delete_action_via_step_gate` | Constructs an `Action` with a sentinel `IntEnum` value 999 (carries `.name="DELETE"` so the STEP-GATE error path can `repr` it) and asserts `state.step(bogus)` outside `oracle_mode` raises `IllegalActionError` / `ValueError`. Second line of defence behind the enum contract. |
| 4 | `test_oracle_kill_friendly_helper_is_not_imported_by_engine` | AST-walks every `engine/**/*.py` looking for: imports of `tools.oracle_zip_replay` / `oracle_zip_replay`, and identifier references (`Name`/`Attribute`/`FunctionDef`) named `_oracle_kill_friendly_unit` or `kill_friendly_unit`. Comments and docstrings are excluded so the guard banner does not trip itself. Expected count: zero. |
| 5 | `test_oracle_delete_helper_only_callable_from_oracle_path` | Confirms `tools.oracle_zip_replay._oracle_kill_friendly_unit` exists and is callable; confirms `engine` module does **not** expose it (neither `hasattr` nor any AST `Import`/`ImportFrom` in `engine/__init__.py` references it). |

Each test has a docstring linking to Phase 11J-DELETE-GUARD-PIN and
restating the strategic concern.

### 3d. Doc update

`docs/oracle_exception_audit/CAMPAIGN_SUMMARY.md` gains a final
**Locked-in Safeguards → RL Action Space Safeguards** section listing
Delete (oracle-only, this pin), indirect-on-seam (Phase 6), and stunned
units (Phase 11J-VONBOLT-SCOP-SHIP).

## 4. Negative-control result

**Procedure:** temporarily added `DELETE = 99` as the last member of
`ActionType` in `engine/action.py`, then ran the canonical import probe.

**Command:**
```
python -c "from engine.action import ActionType; print('SHOULD NOT PRINT')"
```

**Result (exit code 1):**
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\phili\AWBW\engine\action.py", line 827, in <module>
    assert not _collision, (
           ^^^^^^^^^^^^^^
AssertionError: ActionType contains forbidden RL action(s): {'DELETE'}.
See Phase 11J-DELETE-GUARD-PIN. Delete must remain oracle-only.
```

The assertion fires at import time with the exact pin message. The
follow-on `print` never executes — every downstream caller is denied the
engine module the moment a forbidden member is present.

**Revert:** the temporary `DELETE = 99` line was removed immediately;
re-running the probe printed `OK 14 members` (no DELETE), confirming the
pin only fires under genuine collision and is silent on the clean tree.

## 5. Test inventory and gate results

### 5a. New test file

```
$ python -m pytest tests/test_no_delete_action_legality.py -v
============================== 5 passed in 0.15s ==============================
```

All 5 green.

### 5b. Full pytest gate (per Imperator's spec, ignoring the pre-existing
`test_trace_182065_seam_validation.py` outlier)

```
$ python -m pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py
586 passed, 5 skipped, 2 xfailed, 3 xpassed, 2 warnings,
3853 subtests passed in 69.72s
```

**0 failures** with the spec-mandated exclusion. The single non-excluded
failure observed in the unfiltered run is `test_trace_182065_seam_validation`
itself (a pre-existing seam-trace replay test with no relation to
`engine/action.py`), which the Imperator's spec explicitly carves out
(`≤2 failures` budget, this pin contributes zero new failures).

### 5c. Import sanity

```
$ python -c "from engine.action import ActionType; print(list(ActionType))"
[<ActionType.SELECT_UNIT: 0>, <ActionType.END_TURN: 1>,
 <ActionType.ACTIVATE_COP: 2>, <ActionType.ACTIVATE_SCOP: 3>,
 <ActionType.ATTACK: 10>, <ActionType.CAPTURE: 11>, <ActionType.WAIT: 12>,
 <ActionType.LOAD: 13>, <ActionType.UNLOAD: 14>, <ActionType.BUILD: 15>,
 <ActionType.REPAIR: 16>, <ActionType.JOIN: 17>, <ActionType.DIVE_HIDE: 18>,
 <ActionType.RESIGN: 19>]
```

14 members, no DELETE, import succeeds cleanly under normal conditions.

## 6. Files touched

| File | Change |
|------|--------|
| `engine/action.py` | +banner above `ActionType`, +import-time assertion at file end (cleans up its scratch names with `del`). No behavioural change. |
| `tests/test_no_delete_action_legality.py` | New — 5 tests, ~340 LoC including builders. |
| `docs/oracle_exception_audit/CAMPAIGN_SUMMARY.md` | +Locked-in Safeguards / RL Action Space Safeguards trailing section. |
| `docs/oracle_exception_audit/phase11j_delete_guard_pin.md` | New — this report. |

**Untouched (per spec):** `tools/oracle_zip_replay.py`, `engine/game.py`,
`engine/unit.py`, `engine/combat.py`. Read-only confirmation only.

## 7. Verdict

**SHIPPED — pin holds.** The RL action space is now contractually closed
to Delete at three independent layers:

1. **Source-of-truth enum** — `ActionType` cannot be extended with a
   forbidden name without breaking module load.
2. **Runtime gate** — `get_legal_actions` cannot surface a member that
   does not exist; STEP-GATE rejects any synthetic action not in the mask.
3. **Architectural separation** — the oracle helper stays in `tools/`,
   never imported by `engine/`, never exposed from `engine.__init__`.

A future refactor that tries to add `DELETE` to `ActionType` will fail
at the first `import engine.action` — well before any test, training run,
agent invocation, or server boot. The Imperator's strategic concern
(degenerate scrap-and-rebuild) is now structurally unreachable from
the RL stack.

*"Si vis pacem, para bellum."* (Latin, ~5th century AD attributed to Vegetius, *Epitoma rei militaris*, Book III)
*"If you want peace, prepare for war."* — Publius Flavius Vegetius Renatus, Roman writer on military matters under the late Empire.
*Vegetius: late-4th- or early-5th-century author of the most-copied military manual of the Middle Ages; the maxim is the conventional shorthand for deterrence by readiness.*
