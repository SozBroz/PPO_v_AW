# Phase 11J State-Mismatch Full-Corpus Triage

**Status:** YELLOW — 1 strong mechanic ship candidate (Sonja D2D), 1 weaker
funds candidate (capture-tick), and a critical detector calibration finding
that re-frames the other ~787 rows. Bar for "≥3 ship-ready candidates with
Tier 1-2 citation" is partially met (2 mechanic candidates + 1 calibration).

**Mode:** Read-only audit. No code edits, no catalog edits, no ship attempts.
Reused `tools/desync_audit.py --enable-state-mismatch`; no new tools.

**Register produced:** `logs/desync_register_state_mismatch_936.jsonl`
(936 rows; full GL std + extras corpus). Audit log:
`logs/state_mismatch_936_audit.log` (~117 s wall, single-process).

---

## Headline tuple

```
state_mismatch_units:  790
state_mismatch_multi:    0   (none in 936 — 50-game multi rows closed by SASHA-WARBONDS)
state_mismatch_funds:    3
oracle_gap:              1   (build-tile-occupied; carryover, not state drift)
ok:                    142
total:                 936
```

The `0` for `state_mismatch_multi` is real. In the 50-game pre-WARBONDS sample,
multi rows arose because Sasha's bond payouts shifted funds *and* reduced
opponent HP simultaneously. Phase 11J-SASHA-WARBONDS-SHIP closed both axes; in
the 936-corpus rerun every surviving row mismatches on **exactly one axis**.

## Magnitude distribution (the headline finding)

Across **1,835 unit-HP drift datapoints** in 790 rows:

| Bucket (abs ∆ internal HP) | Count | % | Interpretation |
|---|---|---|---|
| **1-9 (sub-display)** | **1,832** | **99.84%** | Detector-precision noise: AWBW `combatInfo` records DISPLAY HP only (1-10), engine pins to display × 10 via `_oracle_combat_damage_override`; per-day PHP snapshot uses sub-display `hit_points` decimal (e.g. `9.4` = internal 94). Drift is the rounding remainder. |
| 10-19 (~1 display HP) | 3 | 0.16% | Real bug — all three are Sonja-bearing rows (see Candidate 1). |
| 20-29 (~2 display) | 0 | — | No repair-tick drifts (R1+R2+R3 funds-tight repair fix is closing them). |
| ≥30 (~3+ display) | 0 | — | No CO-power AOE residuals (Drake / Olaf / Hawke / VonBolt / Sasha SCOP all currently shipped). |

**Sign asymmetry:** 1,818 positive (engine HP > PHP HP) vs 17 negative (engine
HP < PHP HP). The 17 negatives are **100% Sonja-bearing** — a perfect
single-CO signature on the only direction-asymmetric subset of the corpus.

## Cluster breakdown

| Cluster | Definition | Rows | Ship-ready? |
|---|---|---|---|
| **C1 sub-display precision** | All drifts ≤9 internal HP | ~787 | NO (detector noise; see Recommendation §1) |
| **C2 combat damage drift small** | drift ≤9, no consistent CO signal | absorbed in C1 | NO |
| **C3 combat damage drift large (10-19)** | drift ≥10 — **all 3 are Sonja-bearing** | 3 (gids `1631943`, `1632283`, `1632968`) | YES — Candidate 1 |
| **C4 CO power AOE** | drift ≥30 OR clustered ≥4 same-envelope same-magnitude | 0 large; 115 "AOE-shape" rows but all sub-display | NO |
| **C5 funds drift** | `state_mismatch_funds` rows | 3 (gids `1618984`, `1621641`, `1631288`) | PARTIAL — Candidate 2 |
| **C6 detector retune (meta)** | Re-baseline `--state-mismatch-hp-tolerance` from 0 to 9 | reclassifies ~787 rows as `ok` | YES — Candidate 3 |

The "C1 = repair-related" hypothesis from the original brief is **falsified by
the data**: zero drifts in the 18-22 internal HP range. Phase 11J-FUNDS-SHIP
R2 closed the all-or-nothing repair edge case fully.

---

## Top-ranked ship candidates

### Candidate 1 — Sonja (CO 18) D2D combat-perception gap
- **Cluster:** C3 (all 3 mid-range HP drifts) + signature on C1 (Sonja owns
  100% of negative-direction drifts: 17 of 17)
- **gid evidence (≥3, hard):**
  - `1631943` — Adder vs Sonja (T2), env 18 day 10, drift on (1, 17, 15)
    Sonja's unit, engine 62 / php 52, **delta +10**
  - `1632283` — Sonja vs Jake (T2), env 13 day 7, drift on (0, 10, 0)
    Sonja's unit, engine 67 / php 57, **delta +10**
  - `1632968` — Sonja vs Kindle (T2), env 8 day 5, drift on (0, 9, 1)
    Sonja's unit, engine 67 / php 57, **delta +10**
  - **Plus 17 negative-delta drifts, all Sonja-bearing** (gids include
    `1627563`, `1628051`, `1628539`, `1629383`, `1630341`, `1631389`,
    `1632047`×3, `1634484`, `1634965`, `1634973`, `1634975`, `1634977`,
    `1635162`×3) — Sonja accounts for **17/17 = 100%** of the negative-delta
    drifts in the entire 936 corpus.
- **Suspected mechanic (two D2D pieces, both unimplemented in `engine/combat.py`):**
  1. **Hidden HP** — opponents see Sonja's units at 1 display HP less than
     actual; the damage formula `(200 - dv - dtr × hpd_bars) / 100`
     should consume the perceived (actual − 1) `hpd_bars` when the attacker
     is *not* Sonja. Engine uses real `defender.display_hp` unconditionally.
  2. **Counter ×1.5 (D2D)** — Sonja's counters deal 50% more damage. Engine
     handles only the SCOP "counter break" pre-attack-HP path
     (`combat.py:373-379`) and does not apply a 1.5× multiplier to her
     *D2D* counter damage.
- **AWBW citation (Tier 1):** `https://awbw.amarriner.com/co.php`, Sonja row:

  > Sonja: +1 vision, sees into woods/reefs. Units' hit points appear lower
  > to enemies. Counter attacks deal 50% more damage.

  Tier 2 mirror: `https://awbw.fandom.com/wiki/Sonja`.
- **Engine status:** `engine/combat.py:259` (`hpd_bars = defender.display_hp`)
  has **no Sonja branch**. `engine/co.py:166-168` exposes
  `sonja_counter_break` (SCOP only); no `sonja_counter_d2d_multiplier`.
  Greps for `co_id == 18` in `engine/combat.py` return only the SCOP path.
- **Ship complexity:** ~25-40 LOC across two files. Add a perceived-HP hook
  in `calculate_damage` and `damage_range` (one-liner for the perception
  delta, behind a `defender_co.co_id == 18 and not attacker_co.co_id == 18`
  guard — Sonja-vs-Sonja sees true HP), and a 1.5× scalar at the bottom of
  `calculate_counterattack` when `defender_co.co_id == 18`. Risk **MEDIUM** —
  combat ATK path touch.
- **Expected closure:** **3 mid-range gids** definitively (the only ≥10
  internal HP drifts in the corpus). Plus partial closure on the ~32
  Sonja-bearing sub-display rows that survive after detector retune.

### Candidate 2 — Capture-tick funds drift on End/Capt envelopes
- **Cluster:** C5 (all 3 `state_mismatch_funds` rows in the corpus)
- **gid evidence (≥3, exactly):**
  - `1618984` — Andy mirror (T3), day 3 envelope 5, **`Capt`** action, P0
    funds engine = $1,000 vs PHP = $9,000 (**−$8,000 in engine**, exactly 8 ×
    $1,000 income tick).
  - `1621641` — Jake (P0) vs Adder (T4), day 7 envelope 13, **`Capt`** action,
    P0 funds engine = $2,000 vs PHP = $12,000 (**−$10,000 in engine**, exactly
    10 × $1,000 income tick).
  - `1631288` — Adder (P0) vs Grimm (T2), day 4 envelope 7, **`End`** action,
    P0 funds engine = $12,000 vs PHP = $11,000 (**+$1,000 in engine**, exactly
    one extra income tick).
- **Suspected mechanic:** Per-day income on Capt-completion / End-of-turn
  ordering. The first two rows show engine **missing** an entire turn's
  property income at the moment of capture; the third shows engine **double-
  counting** one property's income. The rounded-to-$1,000 magnitudes and
  property-count-equality patterns rule out repair, build, and War Bonds —
  this is income-tick timing.
- **AWBW citation (Tier 1):** `https://awbw.amarriner.com/co.php` Capture
  rules — captured property begins generating income at the start of the
  capturer's *next* turn, not the day of capture.

  Tier 2: `https://awbw.fandom.com/wiki/Capturing` — same canon, with the
  edge case "if a property is captured on the same turn it would have
  generated income for the previous owner, the previous owner does not
  receive that day's income."
- **Engine status:** Plausible overlap with **L1-BUILD-FUNDS-SHIP** lane in
  flight. Greps in `engine/game.py` show `_grant_income`,
  `_resupply_on_properties`, and capture handlers; the income-tick ordering
  vs. `Capt` envelope replay is not obviously instrumented for the same-day
  edge.
- **⚠ Conflict flag:** **Possible overlap with the active L1-BUILD-FUNDS lane.**
  Recommend coordinating with that thread before claiming closure. If
  L1-BUILD-FUNDS is scoped to BUILD-side funds only, this candidate is
  orthogonal; if it covers Capt-side income too, fold into that lane.
- **Ship complexity:** ~10-15 LOC if it's a pure ordering swap. Risk **LOW**
  if isolated, **MEDIUM** if it interacts with End-of-turn income tick
  ordering.
- **Expected closure:** **3 funds gids** (the entire `state_mismatch_funds`
  class).

### Candidate 3 — Detector tolerance retune (meta-ship)
- **Cluster:** C6
- **What it ships:** Re-baseline state-mismatch with
  `--state-mismatch-hp-tolerance 9` (already a CLI flag at
  `tools/desync_audit.py:856-865`; no code change). Rerun and update
  `logs/desync_register_state_mismatch_936.jsonl`.
- **gid evidence:** Implicitly **~787 of 790** `state_mismatch_units` rows
  reclassified to `ok` (everything except the 3 Candidate-1 rows). Sample
  reclassifications: any row in the 50-game sample with sub-display drift —
  e.g. `1605367` (delta 6, 7, 8, 4), `1609533` (delta 7), `1609589`
  (delta 1, 2, 3, 9), `1611364` (delta 8), `1613840` (delta 4) — all become
  `ok` under tolerance 9.
- **Suspected mechanic:** N/A. This is calibration, not engine fix. The
  detector compares engine internal HP against PHP `hit_points × 10`, but
  the engine's combat oracle pins HP to AWBW's display-HP precision (`combat
  Info.units_hit_points × 10`, where the AWBW value is 1-10 display). The 0-9
  internal HP gap is the rounding remainder — present in *every* combat
  whose actual sub-display HP is mid-bucket. Sign skew (1818:17) is
  consistent: when PHP true value is 94 (display 10) the engine pins to 100,
  always overshooting.
- **AWBW citation (Tier 1):** N/A directly. Cite the existing tool docstring
  at `tools/desync_audit.py:855-865`:

  > "Maximum absolute internal-HP delta (engine.Unit.hp vs round
  > (php.hit_points*10)) absorbed silently by the state-mismatch hook.
  > Default 0 = EXACT (per design spec §4). Widen only for narrow luck-noise
  > experiments — wider values mask real combat bugs."

  The corpus evidence proves the design-spec assumption (display-HP-precise
  oracle = 0-tolerance comparable register) is invalidated by `combatInfo`'s
  display-only HP storage. Tolerance 9 = "absorbs everything below 1
  display HP" is the natural new floor. Real bugs ≥1 display HP (Candidate
  1) remain visible.
- **Engine status:** Tool already supports the flag — zero LOC. Risk **LOW**.
  Caveat documented in the docstring is *literally* what we found: tolerance
  10+ would mask real bugs (Candidate 1). Tolerance **9** is the safe
  ceiling that keeps Sonja-class signal visible.
- **Expected closure:** **~787 rows reclassified to `ok`**, leaving a clean
  per-day register of ~6 actionable rows (3 Sonja + 3 funds + a handful of
  Sonja sub-display residuals worth a follow-up drill).

---

## Recommendation

**P0 lane: Candidate 3 (detector retune) FIRST, then Candidate 1 (Sonja D2D).**

Order matters here. Without the retune, the Sonja ship's verification will
drown in 787 rows of unrelated sub-display drift — false negatives and
false positives become indistinguishable. Retune the detector to
`--state-mismatch-hp-tolerance 9` (zero LOC, just rerun + commit the
register), then ship Sonja D2D with a clean before/after register diff.

Candidate 2 (capture-tick funds) is **conditional on L1-BUILD-FUNDS-SHIP
scope review.** Wait for the L1 lane owner to confirm scope before opening
a parallel thread.

### Sequence

1. **C3 retune** — rerun `desync_audit.py --enable-state-mismatch
   --state-mismatch-hp-tolerance 9 --catalog data/amarriner_gl_std_catalog.json
   --catalog data/amarriner_gl_extras_catalog.json --register
   logs/desync_register_state_mismatch_t9_936.jsonl`. Document tolerance
   choice in `tools/desync_audit.py` docstring.
2. **C1 Sonja D2D** — implement perceived-HP hook + D2D counter ×1.5; verify
   against the 3 mid-range gids; rerun retuned audit; expect 3 gids close +
   net positive drift sign on Sonja-bearing rows reduces.
3. **C5 capture-tick funds** — coordinate with L1-BUILD-FUNDS owner; if
   orthogonal, drill `1618984` first (cleanest mirror Andy/Andy minimum-
   variable game).

### What this audit explicitly does NOT recommend

- **No CO-power AOE work** — zero drifts ≥30 internal HP. Every recently-
  shipped power AOE (Sasha War Bonds, VonBolt SCOP, Hawke COP, Olaf SCOP,
  Drake Tsunami/Typhoon) is closing cleanly at the snapshot level.
- **No repair-tick work** — zero drifts in the 18-22 range. R1+R2+R3 fix
  is fully closing repair drifts.
- **No Kindle, Drake, or Colin work** — none surfaces above the noise floor
  here. Refer to `phase11j_co_mechanics_survey.md` for the build-side
  Kindle ship candidate (still Rank 1 there, attacked from oracle_gap).

---

## Coordination

- **VONBOLT-SCOP-SHIP** — no overlap. VonBolt-bearing rows (90 drift units)
  are all sub-display.
- **L1-BUILD-FUNDS-SHIP** — **possible overlap on Candidate 2.** Flag, do
  not open parallel thread without scope check.
- **L2-BUILD-OCCUPIED** — no overlap. The single `oracle_gap` row in this
  register (`1632778`) is build-tile-occupied, classic L2 territory; not
  state-mismatch.
- **SASHA-MARKETCRASH** — no overlap. Already-shipped War Bonds explains
  the 50-sample → 936 disappearance of `state_mismatch_multi`.
- **F4-FRIENDLY-FIRE-WAVE2** — no overlap. No friendly-fire signal in any
  drift row.

---

## Verdict — YELLOW

- **1 strong mechanic candidate** (Sonja D2D hidden HP + counter ×1.5),
  ≥3 gid evidence + Tier 1 cite + clearly-missing engine code.
- **1 conditional mechanic candidate** (capture-tick funds), 3 gids + Tier 1
  cite, *blocked on L1-BUILD-FUNDS scope clarification*.
- **1 detector calibration** (Candidate 3 retune), ~787 row reclassification
  with zero LOC; non-mechanic but ship-ready and a prerequisite for the
  other two to be cleanly verifiable.

The state-mismatch finder is a real net positive — it surfaced the Sonja D2D
gap, three live funds bugs, and the detector-precision tax. But it is **not**
a 30-candidate goldmine. The default 0-tolerance setting overcounts
real-bug signal by ~260×. Retune first, ship Sonja second, coordinate on
funds third. No other actionable mechanic clusters in the residual.

---

## Appendix — register sources & process

- Audit command: `python tools/desync_audit.py --catalog
  data/amarriner_gl_std_catalog.json --catalog
  data/amarriner_gl_extras_catalog.json --enable-state-mismatch --register
  logs/desync_register_state_mismatch_936.jsonl`
- Wall time: ~117 s (single-process)
- Cluster analysis: ad-hoc inline `python -c` over the JSONL register
  (parses `state_mismatch.diff_summary.human_readable` lines with
  `at \(seat, x, y\) hp engine=N php_internal=M ... delta=K`). Not
  promoted to `tools/` — single-use, throwaway.
- CO id → name reference: `data/co_data.json`
- Combat code reviewed: `engine/combat.py` (lines 145-160, 205-277, 247-252,
  330-386), `engine/co.py:160-200`
- Tier 1 web canon spot-checked: `https://awbw.amarriner.com/co.php`
  (Sonja row), capture rules section.

*"Si vis pacem, para bellum."* (Latin, c. 4th–5th century AD)
*"If you want peace, prepare for war."* — Vegetius, *De Re Militari*
*Vegetius: late-Roman military writer; the line is the standard Western maxim on deterrence and readiness.*
