# Phase 7 — Drift triage (oracle_gap → engine_bug after Phase 6)

**Campaign:** `desync_purge_engine_harden`  
**Lane:** D — classification only (no engine/oracle edits in this phase)  
**Inputs:** `logs/phase6_diff_vs_post_phase5.log` (UTF-16 LE — use `encoding="utf-16"` in Python), `logs/desync_register_post_phase6.jsonl`  
**Artifacts:** `logs/phase7_pull.log`, `logs/phase7_44_rows.jsonl`, `logs/phase7_44_classified.json`, `logs/phase7_drill_1618770_window.json`

## Headline counts

| Metric | Value |
|--------|------:|
| Drift rows (oracle_gap → engine_bug) | **44** |
| **Bucket A** — `unit_pos != from` (engine vs replay attacker tile) | **42** |
| **Bucket B** — `unit_pos == from` and `manhattan(from, target) > 1` | **2** |
| **Bucket C** — other / parse fail | **0** |

### Unit type histogram (44 rows)

| Unit type | Count |
|-----------|------:|
| INFANTRY | 14 |
| TANK | 12 |
| B_COPTER | 7 |
| ANTI_AIR | 5 |
| MECH | 3 |
| MED_TANK | 1 |
| NEO_TANK | 1 |
| RECON | 1 |

### Position drift histogram (`manhattan(unit_pos, from)`)

| Drift | Rows | Notes |
|------:|-----:|------|
| 0 | 2 | Both **Bucket B** (no tile disagreement; attacker–target separation is the issue) |
| 1 | 0 | — |
| 2 | 9 | |
| 3+ | 33 | |

### Oracle `from` vs target (Manhattan)

Almost all rows use a replay “from” tile at **Manhattan distance 2** from the target (41/44). In AWBW, that pattern is consistent with **Chebyshev-1 (including diagonal) direct fire**, which Phase 6’s stricter **Manhattan** attacker filter / range gate treats as non-adjacent. The register still shows **dominant Bucket A** (`unit_pos != from`), so **board drift** remains the primary bucket even when diagonal-vs-Manhattan is part of the failure mode.

| `from_to_target` (Manhattan) | Count |
|-----------------------------|------:|
| 2 | 41 |
| 3 | 1 (`games_id` 1628357) |
| 4 | 2 (both Bucket B: 1628198, 1633184) |

## Top 10 representative rows (one shape each)

| `games_id` | Bucket | Unit | Drift | `from→target` MD | Shape |
|------------|--------|------|------:|------------------:|--------|
| 1618770 | A | TANK | 4 | 2 | Earliest GID; tank last moved in replay day 12; failing Fire on day 13 — **case study below** |
| 1620450 | A | B_COPTER | 6 | 2 | Air unit, large drift |
| 1622328 | A | B_COPTER | 6 | 2 | Air + vertical separation in msg |
| 1623738 | A | INFANTRY | 3 | 2 | Typical infantry drift |
| 1626642 | A | MED_TANK | 3 | 2 | Rare heavy type |
| 1628198 | B | NEO_TANK | 0 | 4 | **No position drift**; attacker tile far from target — oracle / attacker-resolution stress |
| 1629722 | A | RECON | 9 | 2 | Rare chassis, very large drift |
| 1630005 | A | TANK | 7 | 2 | Large drift tank |
| 1631742 | A | ANTI_AIR | 6 | 2 | Large drift AA |
| 1633184 | B | INFANTRY | 0 | 4 | Second pure “range / wrong attacker” shape |

Full machine-readable table: `logs/phase7_44_classified.json` → `rows`.

## Case study — smallest Bucket A `games_id`: **1618770**

- **Replay URL:** https://awbw.amarriner.com/replay_viewer.php?games_id=1618770  
- **Register (surrogate locator):** `actions_applied=443`, `approx_envelope_index=25`, `approx_day=13`, `approx_action_kind=Fire`  
- **Full `message`:** `_apply_attack: target (14, 15) not in attack range for TANK from (15, 16) (unit_pos=(17, 18))`

### Failing action (flattened stream)

- **Flat index:** 443 (0-based) — first action that throws; `actions_applied` in the register counts successful applies **before** this action (`tools/desync_audit.py` increments only after a successful `apply_oracle_action_json`).
- **Envelope:** index **25**, sub-index **1**, day **13**, AWBW player id **3742125** (Blue Moon in this match).
- **Action kind:** `Fire`, attacker unit id **191416233** (Tank). Nested `Move` path ends at AWBW path node `{x:16, y:14}`; combat snapshot uses `units_y=14`, `units_x=16` for the attacker pre-walk state in the nested `Move`.

### Backward walk — same unit id **191416233**

Prior flat indices where this `units_id` appears **at or before** index 443:

| Flat idx | Env | Sub | Day | Player id | Kind | Note |
|---------:|----:|----:|----:|----------:|------|------|
| 307 | 19 | 19 | 10 | 3742125 | Build | Unit appears in build payload |
| 335 | 20 | 21 | 11 | 3742126 | End | Opponent turn end |
| 354 | 21 | 18 | 11 | 3742125 | Move | Path end `{x:17,y:19}` |
| 399 | 23 | 12 | 12 | 3742125 | Move | Path end `{x:18,y:17}` → engine-style **(row,col) = (17,18)** matches failing `unit_pos` |
| 441 | 24 | 29 | 13 | 3742126 | End | OS turn end; `updatedInfo.repaired` lists this tank (property repair tick) |
| 443 | 25 | 1 | 13 | 3742125 | Fire | **Fails** — attack toward `(14,15)` with oracle `from` `(15,16)` while engine still holds the unit at `(17,18)` |

### Hypothesis (upstream cause)

1. **Stale board vs Fire envelope (primary):** The last explicit **Move** for this tank in the replay stream ends on day **12** at the tile that matches engine `unit_pos=(17,18)`. The day **13** `Fire` expects an approach march to the firing tile (oracle `from` `(15,16)`). The engine still has the unit at `(17,18)` when `_apply_attack` runs, so `unit_pos` and the attack action’s firing tile disagree. That points to **ordering or application of the nested Fire Move**, or **earlier divergence** that prevented the engine from matching AWBW’s notion of “where the tank is” before combat — not a one-off tank rule edge case.

2. **Manhattan vs diagonal (secondary, same row):** `(15,16)` vs `(14,15)` has **Manhattan 2** (diagonal neighbor). Even with a correct unit position, the current Manhattan range gate may reject that strike unless `get_attack_targets` / `_apply_attack` and oracle agree on **direct-fire neighborhood** (Chebyshev 1 vs Manhattan 1). Phase 6 made this visible; it overlaps with the drift story but is not sufficient alone to explain `unit_pos != from`.

### Recommended Phase 8 probes for this game

1. Trace **`apply_oracle_action_json` for `Fire`**: when nested `Move` is consumed relative to `_apply_attack` / `unit_pos` on the `Action`.  
2. Re-run from day **12** end state and verify whether **repair / end-turn supply** on the opponent envelope (idx 441) mutates only AWBW metadata or also must move units in-engine.  
3. Add a **Chebyshev-vs-Manhattan matrix** for these 44 IDs once positions match (regression guard).

Raw window JSON: `logs/phase7_drill_1618770_window.json`.

## Phase 8 recommendations (ordered by estimated row impact)

1. **Board / unit position drift before combat (~42 rows, Bucket A)**  
   Focus on **Fire nested Move**, **End-of-turn repair/supply lists**, **cargo load/unload**, and any path where the engine’s `get_unit_at` tile lags the replay’s firing tile. The 1618770 trace shows **last Move day 12 → Fire day 13** with `unit_pos` still on the old tile — that pattern should be the first automated regression target.

2. **Direct-fire metric mismatch (~41 rows with Manhattan `from→target` = 2)**  
   After positions are aligned, validate whether failures collapse when **diagonal adjacency** is treated as in-range for direct fire (AWBW Chebyshev-1). Coordinate oracle, `_apply_attack`, and `get_attack_targets` so Phase 6’s Manhattan correction does not reintroduce false `engine_bug` for legitimate diagonal strikes.

3. **Oracle attacker resolution / non-adjacent `from` (~2 rows, Bucket B)**  
   `games_id` **1628198** and **1633184**: `unit_pos == from` but `manhattan(from, target)` is **4** — replay violation is unlikely; investigate **wrong attacker unit** or **wrong tile attribution** in the direct-attacker resolver (post–Phase 6 Manhattan candidate list).

---

*Register fields used: `games_id`, `class`, `message` (attack error text), `actions_applied`, `approx_envelope_index`, `approx_day`, `approx_action_kind`. Fields `defect_locator` / `actor_action_index` are not present on these rows; flat action index **443** is derived from `actions_applied` for 1618770.*
