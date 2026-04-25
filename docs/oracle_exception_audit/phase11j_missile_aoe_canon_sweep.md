# Phase 11J-MISSILE-AOE-CANON-SWEEP — closeout

**Verdict: YELLOW.** Von Bolt Ex Machina is shipped to the AWBW-canon **13-tile Manhattan diamond** (Manhattan distance ≤ 2 from `missileCoords`, Tier 1–3 citations in code). The canonical **936-game** audit **improves** vs the wave-5 baseline. Sturm meteor + missile silo `Launch` were **implemented and PHP-verified**, then **reverted**: a combined ship drove the audit to **794 ok / 141 oracle_gap** (see §4), breaching the **≥3 regression** rollback rule. Those mechanics are **escalated** for a narrower follow-up lane.

---

## 1. AWBW canon (reference block for code comments)

**Tier 1 — AWBW CO Chart** `https://awbw.amarriner.com/co.php` (verbatim excerpts):

- Rachel SCOP “Covering Fire”: *“Three 2-range missiles deal 3 HP damage each.”*
- Von Bolt SCOP “Ex Machina”: *“A 2-range missile deals 3 HP damage and prevents all affected units from acting next turn.”*
- Sturm COP “Meteor Strike”: *“A 2-range missile deals 4 HP damage.”*
- Sturm SCOP “Meteor Strike II”: *“A 2-range missile deals 8 HP damage.”*

**Tier 2 — AWBW Fandom Wiki Interface Guide (Missile Silos):** blast radius 2 squares from center; damage in the **2-range diamond** (interpreted here as Manhattan ≤ 2, **13 tiles** — center + 4 orthogonal at d=1 + 4 diagonal at d=2 + 4 orthogonal at d=2).

**Tier 3 — PHP receipts in-repo:** e.g. Rachel gid 1622501 env 26 (`phase11j_rachel_funds_drift_ship.md`); Von Bolt gid 1622328 env 28 (`unitReplace`: seven enemies, all Manhattan ≤ 2 from center at **pre-envelope** positions; tile `(y=4,x=7)` is outside a 3×3 Chebyshev box); Sturm gid 1615143 env 25 (COP, Power-first) + gid 1635679 env 28/40 (SCOP); silo gid 1636411 `Launch` + frame deltas.

---

## 2. Per-mechanic table

| Mechanic | Before | After (shipped) | PHP / empirical | Tests | Audit note |
|----------|--------|-----------------|-----------------|-------|------------|
| **1. Von Bolt SCOP** | Oracle + docs assumed 9-tile Chebyshev box | **13-tile Manhattan diamond** in `tools/oracle_zip_replay.py` + comment refresh in `engine/game.py` | gid **1622328** env 28 | Updated `tests/test_co_vonbolt_ex_machina.py`, `tests/test_oracle_combat_damage_override_extended.py` | Contributes to **gap reduction** (§3) |
| **2. Sturm COP/SCOP** | No `co_id == 29` HP path | **Reverted** (was: oracle pin + `_apply_power_effects` 40/80 internal vs all units in AOE) | Diamond + 4/8 HP verified on listed gids | **Removed** (revert) | Full ship with silo regressed audit (§4); **do not ship** without isolated gate |
| **3. Missile silo** | `Launch` unsupported | **Reverted** (was: −30 internal in diamond, 111→112) | gid **1636411** | **Removed** (revert) | Same §4 |
| **4. Rachel SCOP** | Already 5-wide diamond | **No code change** (per order) | Already documented | `tests/test_co_rachel_funds_covering_fire_aoe.py` | N/A |

---

## 3. Gate — 936 `desync_audit` (seed 1)

| Register file | ok | oracle_gap | engine_bug |
|---------------|-----|------------|------------|
| Baseline `logs/desync_register_post_wave5_936_20260421_1335.jsonl` | 918 | 17 | 1 |
| **Post Von Bolt diamond only** `logs/desync_register_post_vonbolt_diamond_only_936.jsonl` | **924** | **11** | 1 |

**Net:** +6 ok, −6 oracle_gap, engine_bug unchanged.

Remaining **11** `oracle_gap` rows (sample classes): BUILD funds residual, mover-not-found, RECON vs Mega Tank chart gap — see register lines for gids **1617442, 1624082, 1626236, 1628722, 1628849, 1630341, 1632226, 1632825, 1634464, 1635679, 1635846**. Sturm gid **1635679** still gaps on **unmodeled meteor economy** until a safe Sturm ship lands.

**Aborted full ship register** (Sturm + silo, before revert): `logs/desync_register_post_missile_canon_936.jsonl` — **794 ok / 141 oracle_gap** — **do not use** as a release gate.

---

## 4. Why Sturm + silo were reverted

Order: *revert any mechanic edit that produces ≥3 regressions on the 936 audit.* The combined patch (Sturm oracle + engine + `Launch` handler) blew the gate (**−124 ok**). Without time to bisect on-CPU, both were reverted in full; **Von Bolt-only** re-run restored a **better-than-baseline** gate (§3).

**Counsel:** Re-introduce Sturm and silo under **separate PRs**, each with a **936 audit** before merge. Likely culprits to test first: **`Launch`** (terrain flip + blast ordering in multi-action envelopes) vs **Sturm** (COP envelope not Power-first — `unitReplace` can mix with other actions in the same envelope).

---

## 5. Pytest

- **657 passed**, 5 skipped, 2 xfailed, 3 xpassed — after revert.
- **1 failure:** `test_trace_182065_seam_validation.py::test_full_trace_replays_without_error` — `Illegal move` on Infantry `(9,8)→(11,7)`; **not touched** by this lane (pre-existing / parallel `game.py` drift). Not used as a missile gate.

---

## 6. Files touched (final shipped diff)

- `tools/oracle_zip_replay.py` — Von Bolt SCOP pin: Chebyshev 3×3 → Manhattan diamond; Sturm/`Launch` **removed** after rollback.
- `engine/game.py` — Von Bolt comment refresh + `_oracle_power_aoe_positions` field doc; Sturm `elif co.co_id == 29` **removed** after rollback.
- `tests/test_co_vonbolt_ex_machina.py`, `tests/test_oracle_combat_damage_override_extended.py` — diamond assertions.

---

*“In preparing for battle I have always found that plans are useless, but planning is indispensable.”* — Dwight D. Eisenhower, speech to the National Defense Executive Reserve Conference, 1957. *Eisenhower: U.S. President, Supreme Allied Commander Europe in WWII.*
