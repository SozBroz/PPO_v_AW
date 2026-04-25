# Phase 10H — Re-audit Move-truncate + Build no-op residuals (post–10B + 10E)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** Read-only validation — no edits to `tools/oracle_zip_replay.py` or `engine/`.  
**Question:** After Phase **10B** (terminator snap generalization) and Phase **10E** (Lane K SUSPECT tightenings), how many of the **49** targeted post–Phase 9 `oracle_gap` rows flipped to `ok` or `engine_bug`?

---

## Methodology

1. **Baseline register:** `logs/desync_register_post_phase9.jsonl`.
2. **Sample definition (Lane J–aligned, exact message match):**
   - **Family A (Move-truncate):** `class == oracle_gap` and `message` equals exactly  
     `Move: engine truncated path vs AWBW path end; upstream drift` → **39** rows.
   - **Family B (Build no-op):** `class == oracle_gap` and `message` starts with  
     `Build no-op at tile` → **10** rows.
3. **Excluded from this sample:** Two other `oracle_gap` rows in the same register (e.g. AttackSeam no-ATTACK, mover-not-found) — not part of the 39+10 Lane J families for this lane.
4. **Re-audit:** `tools.desync_audit._audit_one` with keyword-only arguments  
   `games_id`, `zip_path`, `meta`, `map_pool`, `maps_dir`, `seed`  
   (`seed=CANONICAL_SEED` **1**, `MAP_POOL_DEFAULT`, `MAPS_DIR_DEFAULT`, catalog `data/amarriner_gl_std_catalog.json`).  
   Zip paths taken from each baseline row’s `zip_path`.
5. **Artifact:** per-row JSON lines in `logs/phase10h_residual_reaudit.jsonl` (includes `family`, `old_message`, `old_approx_action_kind`).

---

## Results summary

| Metric | Value |
|--------|------:|
| **Total gids re-audited** | **49** |
| **Move-truncate (Family A)** | 39 |
| **Build no-op (Family B)** | 10 |
| **Audit crashes / missing artifacts** | 0 |

---

## Per-family flip table

### Family A — Move-truncate (39 rows)

| Outcome | Count | % of family |
|--------|------:|------------:|
| **`ok`** | 1 | 2.6% |
| **`engine_bug`** | 2 | 5.1% |
| **`oracle_gap` (same message as baseline)** | 36 | 92.3% |
| **`oracle_gap` (different message)** | 0 | 0.0% |

**“Escaped” strict `oracle_gap` for this message ( `ok` + `engine_bug` ): 3 / 39 (7.7%).**

- **`ok`:** `games_id` **1634072** (baseline `approx_action_kind` **AttackSeam**).
- **`engine_bug`:** **1605367**, **1630794** — both baseline **Load**; first divergence is now `ValueError` **Illegal move: … is not reachable** (engine legality), not the truncated-path `UnsupportedOracleAction`.

### Family B — Build no-op (10 rows)

| Outcome | Count | % of family |
|--------|------:|------------:|
| **`ok`** | 0 | 0.0% |
| **`engine_bug`** | 0 | 0.0% |
| **`oracle_gap` (same message as baseline)** | 10 | 100.0% |
| **`oracle_gap` (different message)** | 0 | 0.0% |

**Escaped: 0 / 10 (0.0%).**

---

## Combined flip rate (49 rows)

| Outcome | Count | % of 49 |
|--------|------:|--------:|
| **`ok`** | 1 | 2.0% |
| **`engine_bug`** | 2 | 4.1% |
| **`ok` + `engine_bug`** | **3** | **6.1%** |
| **Still `oracle_gap` with identical message** | 46 | 93.9% |
| **Still `oracle_gap` but changed message** | 0 | 0.0% |

---

## Stuck rows — patterns for Phase 11

### Move-truncate (36 stuck)

All 36 retain the **exact** same user-facing string as post–Phase 9. There is no new `reason` text to rank; the actionable split is by **where** the register last saw an action (`old_approx_action_kind`):

| `approx_action_kind` (stuck only) | Count |
|----------------------------------|------:|
| Fire | 24 |
| Join | 5 |
| Move | 5 |
| Capt | 2 |

**Example gids (stuck, Fire-heavy):** 1619504, 1622140, 1624281, 1626181, 1626437 — same truncation message; dominated by nested-Fire / path-end drift (consistent with Phase 10B report: most Fire rows remained stuck).

### Build no-op (10 stuck)

Message text unchanged on all 10. Coarse refusal shape:

| Pattern (substring in message) | Count |
|-------------------------------|------:|
| `tile occupied` | 8 |
| `insufficient funds` | 2 |

**Example gids (distinct refusal flavors):** 1625178 (MEGA_TANK, tile occupied), 1627563 (INFANTRY, insufficient funds), 1632006 (ANTI_AIR, tile occupied), 1634961 (MECH, insufficient funds).

---

## Verdict vs Phase 10B / 10E intent

- **Phase 10B** targeted Move path-end / terminator reconciliation (Join, Load, Capt, nested Fire, AttackSeam reachability). On this **39-game Move-truncate** slice, the **measured** outcome matches the Phase 10B per-bucket audit already recorded: **1** replay clears to **`ok`**, **2** **Load** replays **surface `engine_bug`** (illegal reachability), and the majority remain the same **`oracle_gap`** truncation string — especially **Fire (24)** and **Join (5)**.
- **Phase 10E** edits were **orthogonal** (diagnostics / strictness on Power, Fire no-path messaging, AttackSeam `combatInfo` context, Unload parsing, `want_t` re-raise, etc.). On **Family B (Build no-op)** and on the **stuck Move-truncate** bulk, there is **no measurable flip** — **0 / 10** builds and **no** additional Move-truncate resolutions beyond what 10B already implied.

**Bottom line:** **10B+10E** moved **3 / 49** rows off the original **`oracle_gap` messages** for this sample (**6.1%**), all from **Family A**; **Family B** is unchanged. Phase 11 should still prioritize **nested Fire + Join** truncation and **build refusal / occupancy / funds** reconciliation.

---

## Artifacts

- `logs/phase10h_residual_reaudit.jsonl` — one JSON object per gid (49 lines).
- This report: `docs/oracle_exception_audit/phase10h_residual_reaudit.md`.
