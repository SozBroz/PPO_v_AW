# Thread BUILD — verdicts

Audit of `tools/oracle_zip_replay.py` BUILD repair / property-owner / funds-injection family (Phase 1 of `desync_purge_engine_harden` campaign). STRICT bar: only AWBW-canon citations earn KEEP.

## Site: `_oracle_assign_production_property_owner` (`tools/oracle_zip_replay.py`:713–732)

**Verdict:** REPLACE-WITH-ENGINE-FIX
**Justification:** The mutations mirror the completion branch of capture (`prop.owner`, `capture_points = 20`, terrain swap, comm refresh) — the same structure as `GameState._apply_capture` when `capture_points <= 0` — but applied from the oracle without that capture having been stepped in the engine. No separate quoted AWBW rule says the replay harness may assign ownership; the canonical fix is for capture (or map load) to leave `PropertyState` and terrain consistent before any `BUILD` envelope is applied.
**Engine fix target:** `engine/game.py` — `GameState._apply_capture` (691–759): ensure capture completion and property/terrain updates match AWBW when a property flips; plus verify replay ordering so a same-turn capture is always applied before `Build`. If the tile was never captured in AWBW, the oracle must not synthesize ownership here.

## Site: `_oracle_snap_neutral_production_owner_for_build` (`tools/oracle_zip_replay.py`:735–766)

**Verdict:** REPLACE-WITH-ENGINE-FIX
**Justification:** Docstring cites an internal register pattern ("Build no-op with `property_owner=None`"), not an external AWBW rule. Under the engine's own `_apply_build` guard (`prop.owner != player` → return), a neutral property (`owner is None`) cannot legally accept `BUILD` (1244–1258). If the live site still emits `Build` while PHP shows neutral, the honest fix is earlier capture/state sync or export ordering — not assigning `eng` from the oracle.
**Engine fix target:** Same as above: `engine/game.py::_apply_capture` and/or the oracle pipeline's ordering relative to `Capt` / turn boundaries so `PropertyState.owner` is non-`None` for the builder when AWBW allows production.

## Site: `_oracle_build_discovered_matches_awbw_player_map` (`tools/oracle_zip_replay.py`:769–780)

**Verdict:** DELETE
**Justification:** Key-shape equality and "all values `None`" are heuristics for "trusted" envelopes; nothing in the required reading quotes AWBW or Replay Player docs defining `discovered` as proof of ground truth for ownership or funds.
**Replacement message:** `raise UnsupportedOracleAction("Build repair gated on discovered dict shape rejected: no citable AWBW canon for all-null discovered as authenticity proof")`

## Site: `_oracle_site_trusted_build_envelope` (`tools/oracle_zip_replay.py`:783–796)

**Verdict:** DELETE
**Justification:** Combines envelope seat id, two PHP seats, and the unauthenticated `discovered` heuristic; expands repair surface beyond what STRICT allows without a cited site spec.
**Replacement message:** `raise UnsupportedOracleAction("Build trusted-envelope gate removed: two-seat discovered is not citable AWBW authority for ownership/funds/blocker repairs")`

## Site: `_oracle_optional_apply_build_funds_hint` (`tools/oracle_zip_replay.py`:799–815)

**Verdict:** DELETE
**Justification:** Raises `state.funds[eng]` from `funds.global` when the engine is short — economy fabrication. `docs/desync_audit.md` treats PHP snapshot funds as the unconditional snap target in `oracle_state_sync` (117–118), not `p:` envelope `funds` as spendable truth at BUILD time; no quoted AWBW rule that this field overrides engine `_build_cost` checks.
**Replacement message:** `raise UnsupportedOracleAction("Build funds hint rejected: envelope funds.global must not bump engine funds without citable AWBW authority")`

## Site: `_oracle_drift_teleport_blocker_off_build_tile` (`tools/oracle_zip_replay.py`:1272–1317)

**Verdict:** DELETE
**Justification:** Teleporting a unit to an arbitrary neighbour or setting `u.hp = 0` to clear the factory tile is not Advance Wars / AWBW rules resolution; it hides combat/movement desync.
**Replacement message:** `raise UnsupportedOracleAction("Build blocked: factory tile occupier cannot be teleported or killed by oracle drift recovery")`

## Site: `_oracle_nudge_eng_occupier_off_production_build_tile` (`tools/oracle_zip_replay.py`:1320–1419)

**Verdict:** PARTIAL — KEEP legal-step branch (unmoved friendly: `SELECT_UNIT` → orth move → `WAIT`/`DIVE_HIDE`); DELETE the `u.moved` teleport branch (1356–1367) and the drift-teleport fallthrough (1416–1418). (Escalation resolved 2026-04-20 by commander.)
**Justification:** The unmoved path issues real engine actions through `_engine_step` — same mechanism a player uses when AWBW's site implicitly handles "move off factory before build" within a single user click. The teleport branches mutate state without legal action stepping and must die under STRICT.
**Action items:**
1. Keep lines 1320–1355 + 1368–1415 (unmoved legal-step path).
2. Delete lines 1356–1367 (`u.moved` teleport branch).
3. Delete lines 1416–1418 (drift-teleport fallthrough).

## Site: `kind == "Build"` handler — silent no-op path (`tools/oracle_zip_replay.py`:5535–5639, non-strict)

**Verdict:** DELETE
**Justification:** When `ORACLE_STRICT_BUILD` is `0`/`false`/`no`/`off`, the `if strict:` block (5580–5638) is skipped and the function returns after `_engine_step(BUILD)` even if `_apply_build` no-oped — `docs/desync_audit.md` (281) explicitly frames this as hiding gaps for batch triage, not as AWBW-authorized behavior.
**Replacement message:** N/A — remove the env gate and always run the post-BUILD verification that surfaces refusal (or always `raise UnsupportedOracleAction` when BUILD is a no-op).

## Site: Strict-mode funds bump + BUILD retry (`tools/oracle_zip_replay.py`:5585–5592)

**Verdict:** DELETE
**Justification:** Sets `state.funds[eng] = max(..., need)` then retries `BUILD` — direct gold injection to mask `_oracle_diagnose_build_refusal` "insufficient funds" without fixing Colin/Hachi/discount/income parity in the engine (`docs/desync_audit.md` 281 names this pattern). No quoted AWBW exception for oracle-side funding.
**Replacement message:** `raise UnsupportedOracleAction("Build refused: insufficient funds — oracle must not inject funds to force BUILD")`

## Site: Strict-mode ownership / occupancy retries (`tools/oracle_zip_replay.py`:5595–5632)

**Verdict:** DELETE
**Justification:** Retries after `_oracle_snap_wrong_owner_production_for_trusted_site_build` and `_oracle_drift_teleport_blocker_off_build_tile` use envelope-trust reasoning ("AWBW emitted Build…") without a cited AWBW spec; the occupancy path even invokes drift teleport. Same cover-up class as the helpers above.
**Replacement message:** `raise UnsupportedOracleAction("Build refused after strict diagnosis: ownership/occupancy repair via snap/teleport removed — fix engine/oracle replay instead")`

## ESCALATIONS (resolved)

1. **`_oracle_nudge_eng_occupier_off_production_build_tile` (1320–1419):** **RESOLVED 2026-04-20 → KEEP legal-step branch only.** Commander: synthetic legal action chains are acceptable; teleports and drift fallthroughs are not.
2. **`ORACLE_STRICT_BUILD` env flag:** **RESOLVED 2026-04-20 → KILL the env flag entirely.** Commander: BUILD failures must always raise `UnsupportedOracleAction`; no opt-out, no funds injection, no silent absorption. Audit always surfaces the gap.

## Recommendation on `ORACLE_STRICT_BUILD`

**Short answer:** The env-flag-controlled silent branch should NOT survive the STRICT campaign: default is already strict in code (`ORACLE_STRICT_BUILD` unset → `"1"`), but allowing `=0` to skip all post-BUILD checks lets `desync_audit` report `ok` while `_apply_build` no-ops (`docs/desync_audit.md` 281) — the opposite of "surface the gap." Remove the opt-out (or invert semantics so only an explicit "legacy triage" flag enables silence) so BUILD failures cannot be absorbed.

**Funds injection (5585–5592):** Confirm — it should die outright under the commander's bar. Explicit economy forgery; duplicates the problem class of `_oracle_optional_apply_build_funds_hint`, only worse because it runs even without `funds.global` in the envelope.

## Summary

| Verdict | Count |
|---------|------:|
| KEEP | 0 |
| PARTIAL (legal-step kept, teleport branches DELETE) | 1 |
| DELETE | 6 |
| REPLACE-WITH-ENGINE-FIX | 2 |
| ESCALATE | 0 (all resolved) |
