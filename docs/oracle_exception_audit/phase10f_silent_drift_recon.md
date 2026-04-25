# Phase 10F — Replay-fidelity silent-drift recon

## Purpose

Post-Phase-9 register rows with `class: ok` mean the oracle drove the engine through every `p:` envelope without raising. That does **not** prove bitwise agreement with AWBW’s serialized PHP snapshots on each half-turn.

This recon compares engine `GameState` to **gzipped `awbwGame` lines** in each zip—same reference as `tools/replay_state_diff.py` and the C# AWBW Replay Player—using the same per-game RNG seed as `tools/desync_audit._audit_one` (`random.seed(_seed_for_game(CANONICAL_SEED, games_id))`).

## Sample

| Item | Value |
|------|--------|
| Source register | `logs/desync_register_post_phase9.jsonl` |
| Rows with `class == ok` | 627 |
| Sample size | 50 |
| Selection | `random.seed(1)`, `random.sample(ok_gids, 50)`, sorted ascending |
| Harness | `python tools/_phase10f_recon.py` → imports `replay_state_diff.run_zip` |

Artifacts: `tools/_phase10f_recon.py`, `logs/phase10f_silent_drift.jsonl`.

## Tally

| Metric | Count |
|--------|------:|
| First PHP snapshot mismatch (**drift**) | **39** |
| No mismatch before compare stopped (**clean**) | 11 |
| └─ Compared every envelope to PHP until zip end | **3** |
| └─ `Resign` / terminal before action stream exhausted (`oracle_error` set; no mismatch detected earlier) | 8 |

**Verdict:** In this sample, **`ok` does not imply Replay Player snapshot parity.** Most sampled games show funds and/or HP disagreement against PHP at the first failing step.

## Per-game drift table (`snapshot_diff_ok == false`)

| games_id | First mismatch envelope index | Primary `drift_kind` | Notes |
|----------|-------------------------------|----------------------|--------|
| 1609626 | 19 | funds | P0 funds |
| 1616284 | 29 | position | Tile set: PHP (0,12,7) vs engine (0,12,8) |
| 1619803 | 18 | funds | P1 funds + HP at one tile |
| 1620188 | 13 | funds | P0 funds |
| 1620450 | 24 | funds | P1 funds + multiple hp_bars |
| 1620579 | 18 | funds | Extreme P0/P1 vs PHP (likely pairing / timing — triage) |
| 1623193 | 19 | funds | P0 funds + hp |
| 1624670 | 15 | funds | P0 funds + hp |
| 1624953 | 18 | funds | P1 + hp |
| 1625290 | 19 | funds | Large bilateral funds delta |
| 1625681 | 20 | funds | P1 + hp |
| 1627328 | 22 | funds | P1 + hp |
| 1628095 | 18 | funds | P1 funds |
| 1628226 | 13 | funds | P0 + hp |
| 1628301 | 20 | funds | Large bilateral funds |
| 1628324 | 21 | funds | P0 + several hp (incl. engine *higher* max-ish bars) |
| 1628541 | 17 | funds | P0 + hp |
| 1628546 | 11 | funds | P0 + hp |
| 1628609 | 13 | funds | Large bilateral funds |
| 1630005 | 21 | funds | P0 + hp |
| 1630151 | 19 | funds | Large bilateral funds |
| 1631194 | 16 | funds | P1 + hp |
| 1631742 | 17 | funds | P0 + hp |
| 1631755 | 19 | funds | Large bilateral funds |
| 1631840 | 22 | funds | P1 funds |
| 1632233 | 12 | funds | Large bilateral funds |
| 1632355 | 14 | funds | P1 + hp |
| 1632441 | 18 | funds | P1 + hp |
| 1632495 | 18 | funds | P1 + hp |
| 1632662 | 22 | funds | P1 + hp |
| 1632707 | 18 | funds | P1 funds |
| 1632968 | 8 | hp | hp_bars first (no funds line before) |
| 1633242 | 15 | funds | Large funds + mixed hp |
| 1633894 | 17 | funds | P0 + hp |
| 1634030 | 18 | hp | hp_bars first (Md.Tank vs PHP snapshot) |
| 1634482 | 21 | structure | `Missiles` vs `Missile` naming — likely comparator alias gap, not gameplay |
| 1634522 | 16 | funds | P1 + hp |
| 1634889 | 17 | funds | P0 + hp |
| 1636157 | 23 | funds | Large funds + hp |

Envelope index is **0-based** `step_i` from `replay_state_diff` (same as after applying envelope `step_i`, compare to `frame[step_i+1]` when present).

## Games with no mismatch before stop (`snapshot_diff_ok == true`)

| games_id | Notes |
|----------|--------|
| 1625241 | Full zip compared (trailing pairing), no mismatch |
| 1629276 | Full zip compared (trailing), no mismatch |
| 1630555 | Full zip compared (trailing), no mismatch |
| 1624806, 1629456, 1630968, 1632881, 1633133, 1634245, 1634344, 1635366 | Terminal before stream exhausted — **no drift detected** in the prefix that was compared |

## Pattern analysis

Drift is **dominated by funds** appearing first in `compare_snapshot_to_engine` (funds are checked before units). Many rows also list **`hp_bars`** mismatches on the same step—consistent with **docs/desync_audit.md** Agent 8 findings: combat/resolution and economy drift compound. A single **pure tile-set** case (1616284) points at **movement / position** divergence (Bucket A–style). Two games show **HP-first** classification (1632968, 1634030) where funds already matched. Extremely large fund divergences (e.g. 1620579, 1625290, 1630151) may indicate **snapshot pairing edge cases** or **COP/income timing** worth isolating in Phase 11, not only oracle action mapping. The `Missiles` / `Missile` row is probably a **string alias** issue in the comparator, not a silent engine bug.

## Confidence on `ok` semantic

- **Oracle stream:** `ok` remains a reliable indicator that **`apply_oracle_action_json` completed** for every envelope (no exception).
- **AWBW Replay Player state:** **Do not** infer parity. In this sample, **39/50** showed **measurable snapshot drift** vs PHP; only **3/50** matched PHP through the final compared frame without early resignation.

## Recommended Phase 11 follow-ups

1. **Funds-first mismatches:** Trace income, repair charges, build costs, and COP-linked economy vs PHP snapshot timing (`replay_snapshot_compare.compare_funds`).
2. **HP / combat:** Where funds match but `hp_bars` differ, narrow to **luck vs oracle combat override** (see `oracle_state_sync`, `combatInfoVision` in `oracle_zip_replay`).
3. **Position-only (1616284):** Align with Phase 10A/10B movement/oracle path work.
4. **Comparator hygiene:** Add alias for PHP `'Missile'` vs engine `"Missiles"` if triage confirms no type error.
5. **Optional:** Batch `replay_state_diff` (or `_phase10f_recon` at higher N) over more than 50 `ok` games to quantify rates by map/CO tier.

---

*“In God we trust; all others bring data.”* — W. Edwards Deming (attributed; management-quality aphorism, late 20th c.)

*Deming: American statistician and management thinker known for quality-control and continuous improvement.*
