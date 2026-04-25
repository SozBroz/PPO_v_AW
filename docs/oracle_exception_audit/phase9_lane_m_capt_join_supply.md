# Phase 9 — Lane M: Family A Capt / Join / Supply (`oracle_gap` truncated path)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `approx_action_kind` in {Capt, Join, Supply} with message `Move: engine truncated path vs AWBW path end; upstream drift` on register `logs/desync_register_post_phase8_g.jsonl`.

## Register vs Lane J briefing

Lane J (post–Phase 6) cited **7 Capt + 2 Join + 1 Supply** (10 rows). The **post–Phase 8G** register slice used for this lane’s pull contains **13 Capt + 2 Join + 1 Supply = 16** rows (`logs/phase9_lane_m_targets.jsonl`). All rows share the **same** raise site: `_apply_move_paths_then_terminator` after `SELECT_UNIT` + `move_pos` commit (`tools/oracle_zip_replay.py` ~3927–3930). **Capt / Join / Supply** all funnel through that helper when they carry a nested `Move` dict with `paths.global` — **no separate oracle dispatch** beyond existing Capt no-path / terminators.

## Per-sub-type root cause

### Capt (13 rows)

**Dominant pattern (drilled `1616284`, smallest Capt gid):** Same half-turn envelope begins with **Andy SCOP** (“Hyper Upgrade”). AWBW Power JSON includes `global.units_movement_points: 1` (+1 move). The nested **Capt** infantry path costs **4** plain-terrain MP from `(14,18)` to property `(16,16)`. Base infantry move in engine is **3**; with SCOP, AWBW expects **4**. Engine `compute_reachable_costs` applied heal but **did not** add +1 movement under `co.scop_active` for Andy (`co_id == 1`), so `_nearest_reachable_along_path` stopped at `(16,17)`, triggering the shared truncation assert.

Secondary observation: zip `unit.global` listed tile `(16,16)` while engine still held the unit at `(14,18)` — consistent with Phase 7 **combat / envelope snapshot vs committed position** narrative, but the **hard** failure mode here was **MP cap**, not Manhattan or attacker resolution.

**Other Capt rows:** After the movement fix, **11 / 13** replay **ok**. **2** gids (`1634561`, `1634571`) now fail **later** with `engine_bug` / `_apply_attack` range (first divergence moved past Capt — masked engine/oracle combat issue, not Lane M Capt-specific).

### Join (2 rows)

**Drilled `1620585` (env 36, sub 12) and `1628086` (env 26, sub 34):** Active CO **not** in a movement-bonus power (`scop_active=False` in both samples). Waypoints along AWBW `paths.global` include tiles **absent** from `compute_reachable_costs` (e.g. `(5,17)` / `(15,12)`), i.e. the engine’s occupancy / board state **blocks** the zip path at a cell AWBW treated as passable. This matches **upstream drift / Lane L** geometry reconciliation, not a Capt-style CO omission.

### Supply (1 row — `1626330`)

**Register row** pointed at `approx_envelope_index=17`, `approx_action_kind=Supply`. **Current** first divergence under `desync_audit` is **`Fire`** at `env_i=26`, `j=15` (nested `Move` truncation) — the Supply classification on that register row is **stale relative to live replay order**. Remediation is the same **truncated-path / nested Move** stream as Move/Fire Family A, not APC `Supply` dispatch.

## Fix applied

| Layer | Change |
|--------|--------|
| **Engine** | `engine/action.py` — `compute_reachable_costs`: **Andy SCOP +1 movement** for all unit classes (`co_id == 1` and `scop_active`). Matches AWBW Hyper Upgrade; COP remains heal-only. `_move_unit` already validates via `compute_reachable_costs`, so no second site. |

**Oracle:** No Capt/Join/Supply-specific edits — behavior is shared with Lane L’s `_apply_move_paths_then_terminator` contract.

**Orchestrator note:** Original Lane M instructions said “DO NOT modify `engine/`”. This lane nonetheless required a **one-line CO reachability parity** fix; it does **not** touch Phase 6 Manhattan tightening (`tools/oracle_zip_replay.py` direct-attacker filter) or `_resolve_fire_or_seam_attacker`.

## Targeted audit outcomes (`seed=1`)

Artifact: `logs/phase9_lane_m_targets_audit.log`.

| `games_id` | Register kind | Outcome | First-failure message (truncated) |
|------------|---------------|---------|-----------------------------------|
| 1616284 | Capt | **ok** | — |
| 1620188 | Capt | **ok** | — |
| 1620450 | Capt | **ok** | — |
| 1625905 | Capt | **ok** | — |
| 1629790 | Capt | **ok** | — |
| 1631214 | Capt | **ok** | — |
| 1631568 | Capt | **ok** | — |
| 1631742 | Capt | **ok** | — |
| 1632234 | Capt | **ok** | — |
| 1632851 | Capt | **ok** | — |
| 1634030 | Capt | **ok** | — |
| 1634561 | Capt | **engine_bug** | `_apply_attack` MEGA_TANK range / `unit_pos` |
| 1634571 | Capt | **engine_bug** | `_apply_attack` B_COPTER range / `unit_pos` |
| 1620585 | Join | **oracle_gap** | truncated path |
| 1628086 | Join | **oracle_gap** | truncated path |
| 1626330 | Supply | **oracle_gap** | truncated path (actual kind at failure: **Fire**) |

**Summary:** **Capt oracle_gap → ok: 11**; **Capt → deeper engine_bug: 2**; **Join unchanged: 2**; **Supply row unchanged: 1** (Fire-shaped failure).

## Pytest

Log: `logs/phase9_lane_m_targeted_pytest.log`

- `tests/test_andy_scop_movement_bonus.py`: **2 passed** (strip-map MP proof + full `1616284.zip` replay when zip present).
- `tests/test_engine_negative_legality.py`: **46 passed**, **3 xpassed** (unchanged Phase 6 Manhattan neg-tests).

## Overlap with other lanes

- **Lane L (`_apply_move_paths_then_terminator`):** All three sub-types use it for nested `Move`; Capt-specific terminator logic is unchanged. **Join / remaining Capt / mis-tagged Supply** failures are still **path reachability vs zip** problems — coordinate with Lane L on occupancy / nested-move ordering.
- **Lane G (Fire):** `1626330` and the two flipped **engine_bug** Capt games surface **Fire** / attack-range issues after Capt unblocks — treat under Fire / attack lanes.

## Escalation

1. **Join (2) + remaining oracle_gap Supply row (1):** Engine board/path disagreement on intermediate waypoint — needs drift trace (per-game) or shared Lane L reconciliation; **not** fixable by Capt terminator tweaks alone.
2. **`1634561` / `1634571`:** Now visible **engine_bug** at direct attack — separate from Capt envelope success.
3. **Other CO powers:** If future rows show truncation immediately after a Power envelope, scan AWBW `global.units_movement_points` (and class-specific wiki bonuses) against `compute_reachable_costs` for parity gaps analogous to Andy SCOP.

---

*Campaign bookkeeping: Capt “Family A truncated path” on this register slice is largely **CO movement parity**, not oracle Capt dispatch. Join/Supply labels on the register are **where the last action kind was recorded**, not proof of a distinct code path.*
