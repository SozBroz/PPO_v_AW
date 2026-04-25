# Phase 11D — Residual `engine_bug` triage (Phase 10Q baseline)

**Mode:** read-only (no engine, oracle, tests, or data edits)  
**Baseline:** `logs/desync_register_post_phase10q.jsonl`, `class == "engine_bug"` (10 rows after 741-game audit)  
**Cross-refs:** Phase 7 Bucket A/B (`docs/oracle_exception_audit/phase7_drift_triage.md`), Phase 10A B_COPTER (`phase10a_b_copter_pathing.md`), Phase 10D non–B_COPTER (`phase10d_non_b_copter_triage.md`), Phase 9 snapshot `logs/desync_register_post_phase9.jsonl`

## Executive summary

All ten rows are classifiable into the campaign taxonomy. **Six** show **`unit_pos` ≠ oracle `from`** (Manhattan \> 0) — **Phase 7 Bucket A / move-truncation family** (F1; three B_COPTER rows are also Phase 10A’s **three B_COPTER residuals**, F3). **Two** are **`Illegal move: … not reachable`** (F2) — `compute_reachable_costs` / movement-legality lane. **One** is **friendly fire** (F4). **One** is **Black Boat “attack” with drift 0** — unarmed attacker / envelope semantics (F5).  

**Phase 9 cross-reference:** eight games were already `engine_bug` in Phase 9; **two** were **`oracle_gap`** (“truncated path vs AWBW path end”) and are now first-divergence **`engine_bug`**, i.e. **unmasked** after later pipeline phases.  

**Verdict:** **WORK** — 10 bugs, all classified, fix lanes identified.

---

## Per-bug table

Manhattan **`Δ(unit_pos, from)`** uses coordinates parsed from each 10Q `message` (`unit_pos=(…)`, `from (…)`, `from (row, col)` patterns).

| # | games_id | zip path | Unit | Δ(`unit_pos`,`from`) | from → target (MD) | Family | Classification & fix lane |
|---|----------|----------|------|------------------------|----------------------|--------|---------------------------|
| 1 | 1605367 | `replays/amarriner_gl/1605367.zip` | Mech (Load) | *n/a* (no `unit_pos` in msg) | from (1,16) → to (2,14): **MD 3** | **F2** | **Illegal move / reachability** — terrain id=1, fuel 68. Base MECH `move_range=2` (`engine/unit.py`); **MD 3** end displacement plausibly exceeds one-turn reach or path cost. **Lane:** `compute_reachable_costs` / load-step application vs oracle path (Phase 10C / Lane L family). |
| 2 | 1622104 | `replays/amarriner_gl/1622104.zip` | MECH | **1** | (6,17) → (6,16): 1 | **F1** | **Bucket A** — engine one tile behind oracle stance. **Lane:** oracle position-snap / nested Fire commit (10B/10C). |
| 3 | 1625784 | `replays/amarriner_gl/1625784.zip` | B_COPTER | **3** | (8,2) → (8,1): 1 | **F1 + F3** | **Bucket A**; **Phase 10A cited example** for residual (“3-tile truncation upstream”). **Lane:** same as F1; B_COPTER called out in 10A as pathing/truncation, not damage table. |
| 4 | 1626642 | `replays/amarriner_gl/1626642.zip` | BLACK_BOAT | **0** | (1,3) → (2,3): 1 | **F5** | **Drift 0** — orthogonally adjacent target; engine still rejects strike. Phase 10D **Class F**: Black Boat has no direct-fire weapon in damage matrix → **`get_attack_targets` empty** / `_apply_attack` range error masks **oracle “Fire” vs repair/export** mismatch. **Lane:** oracle / replay classification (10B), not `compute_reachable_costs`. |
| 5 | 1630794 | `replays/amarriner_gl/1630794.zip` | Infantry (Load) | *n/a* | (2,7) → (1,10): **MD 4** | **F2** | **Illegal move** — terrain id=46. INF `move_range=3`; **MD 4** displacement ⇒ one-turn reachability very likely false without CO/path edge (needs in-viewer confirmation). **Lane:** F2 / movement + load envelope. |
| 6 | 1630983 | `replays/amarriner_gl/1630983.zip` | MECH | **2** | (13,22) → (13,23): 1 | **F1** | **Bucket A** — engine on (13,20) vs oracle (13,22). Phase 9 had a *different* coordinate snapshot for this gid (MECH drift at another stance); 10Q failure point updated but family unchanged. **Lane:** F1. |
| 7 | 1631494 | `replays/amarriner_gl/1631494.zip` | FIGHTER | **10** | (16,13) → (14,13): 2 | **F1** | **Large Bucket A** — Phase 10A example **`unit_pos=(15,4)` vs `from (16,13)`**. Phase 9 register showed **MEGA_TANK** at a different tile (failure advanced); 10Q first divergence is **FIGHTER** Fire — same **truncation / board-lag** family. **Lane:** F1. |
| 8 | 1634664 | `replays/amarriner_gl/1634664.zip` | INFANTRY (defender) | *see note* | friendly fire | **F4** | **`_apply_attack` friendly fire** (`engine/game.py` lines 635–638): attacker and defender same `player`. Either **wrong owner on attacker/defender** after envelope, **self-target** payload quirk, or **mis-associated unit** for Fire. **Lane:** owner index + envelope parsing; correlate with C# viewer for self-target legality. |
| 9 | 1635025 | `replays/amarriner_gl/1635025.zip` | B_COPTER | **6** | (14,19) → (15,19): 1 | **F1 + F3** | **Bucket A**; **10A residual cohort** (large air drift). **Lane:** F1 + B_COPTER emphasis. |
| 10 | 1635846 | `replays/amarriner_gl/1635846.zip` | B_COPTER | **4** | (8,5) → (8,4): 1 | **F1 + F3** | **Bucket A**; **10A residual cohort**. **Lane:** F1 + B_COPTER emphasis. |

**C# replay viewer:** For implementation work, open each `zip` in the upstream AWBW Replay Player with `--goto-*` locators from the row (`approx_envelope_index`, `approx_day`, etc.); see `.cursor/skills/desync-triage-viewer/SKILL.md`.

**Code anchors (read-only pointers):**

- `engine/game.py::_apply_attack` — friendly-fire guard (L635–638); range check uses `get_attack_targets` with `atk_from = action.move_pos or attacker.pos` (L641–646).
- `engine/action.py::compute_reachable_costs` — MP cap, terrain costs, CO movement bonuses (F2 triage).

---

## Family rollup

Counts below are **primary** labels; **F3 ⊆ F1** for the three B_COPTER rows (do not double-count toward “10 unique bugs”).

| Family | Count | Notes |
|--------|------:|-------|
| **F1** — Bucket A position drift (`Δ(unit_pos, from) > 0`) | **6** | Rows 2, 3, 6, 7, 9, 10 |
| **F2** — Move pathing / “not reachable” | **2** | Rows 1, 5 |
| **F3** — Phase 10A B_COPTER residual | **3** | Rows 3, 9, 10 (same three as F1’s B_COPTERs) |
| **F4** — Owner / friendly fire | **1** | Row 8 |
| **F5** — Other (oracle / unarmed attacker) | **1** | Row 4 (BLACK_BOAT) |

---

## Phase 10A cross-reference — three B_COPTER residuals

Phase 10A (`phase10a_b_copter_pathing.md`) reports **3** residual `engine_bug` rows in the **47 B_COPTER** targeted cohort, described as **move truncation / upstream drift** (same shape as Phase 10C ownership).

The **three B_COPTER** rows in the 10Q top-10 list are:

| games_id | Matches 10A residual narrative |
|----------|---------------------------------|
| **1625784** | **Yes** — explicit case study in 10A (“3-tile truncation upstream”). |
| **1635025** | **Yes** — large `unit_pos` vs `from` drift; same failure family. |
| **1635846** | **Yes** — same cohort and signature. |

**None** of the three are “new” B_COPTER species beyond 10A’s documented residual set — they **are** that set, still open at 10Q floor.

---

## Phase 9 cross-reference (63 → 10 `engine_bug`)

Compared `logs/desync_register_post_phase9.jsonl` for the same ten `games_id`:

| Segment | Count | games_id |
|---------|------:|------------|
| **Already `engine_bug` in Phase 9** | **8** | 1622104, 1625784, 1626642, 1630983, 1631494, 1634664, 1635025, 1635846 |
| **Were `oracle_gap` in Phase 9, now `engine_bug` in 10Q** | **2** | **1605367**, **1630794** (Phase 9 message: `Move: engine truncated path vs AWBW path end; upstream drift`) |

**Interpretation:**

- **Eight** are **long-standing** engine_bug surfaces (same campaign lane since Phase 9).
- **Two** are **class flips from oracle_gap:** earlier work surfaced **upstream drift** first; **10Q** now fails first on **`Illegal move: … not reachable`** — consistent with **unmasking** deeper movement/oracle application order after Phase 10A/10B/10C-style fixes removed masking failures earlier in the stream.
- **1631494** remains `engine_bug` but **failure unit/coords in the message changed** between Phase 9 (MEGA_TANK snapshot) and 10Q (**FIGHTER**) — replay first-divergence **moved forward**; still **F1** truncation family.

---

## Phase 11 fix priority (top 5)

| Rank | games_id | Rationale | Est. complexity |
|------|----------|-----------|-----------------|
| 1 | **1625784**, **1635025**, **1635846** (bundle) | **Largest coherent slice:** all **F1+F3** B_COPTER; matches **10A residual** list; single engine/oracle **nested Fire + move commit** fix likely clears all three. | **High** (touches hot path: oracle zip apply + air move parity); **high payoff** (3/10). |
| 2 | **1631494** | **Worst drift** (Δ=10); stress-tests any partial snap fix. Fix **or** validate after (1). | **High** (same lane as 1). |
| 3 | **1622104**, **1630983** | **Smaller MECH drifts** (1 and 2 tiles) — good **regression canaries** for Ground nested Fire once air path is understood. | **Medium**. |
| 4 | **1634664** | **Isolated F4** — small code surface if root cause is owner bit/envelope; **risk** if it reveals systemic mis-attribution. | **Low–medium** (investigation first in viewer). |
| 5 | **1605367**, **1630794** | **F2** pair — both were Phase 9 **oracle_gap**; blocked on **reachability** vs AWBW path. Depends on terrain/CO/L **and** load semantics. | **Medium–high** (may need terrain id→cost audit + transport rules). |

**Deferred (not “wrong priority” — different owner):** **1626642** (F5) — oracle/Black Boat semantics; fix alongside **10B** rather than engine movement core.

---

## Mandatory closure reminder (for implementers)

Per campaign rules, each replay should eventually close with **replay delete (if scuffed)**, **oracle fix**, or **engine fix** after deep triage — not left ambiguous in Phase 11 backlog alone.

---

*Phase 11D read-only triage complete.*
