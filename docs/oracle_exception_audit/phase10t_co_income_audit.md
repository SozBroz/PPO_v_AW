# Phase 10T — Comprehensive CO income / treasury canonical audit

READ-ONLY recon for Phase 11. **No engine edits** in this phase.

## Executive return (briefing)

| Metric | Value |
|--------|-------|
| **Total COs audited** | **28** (all entries in `data/co_data.json`; roster uses IDs 1–30 with **4** and **6** absent) |
| **Engine gaps touching daily `_grant_income`** | **1** CO with a confirmed D2D rule not implemented: **Kindle (23)** — +50% income from owned **cities** per `data/co_data.json` and Phase 10N drill; **AWBW CO Chart** does not state a Kindle **income** line (see discrepancies). |
| **Broader treasury gaps** (income, build cost, COP/SCOP funds, repair) | **5** COs with at least one material gap vs primary canon: **Kindle**, **Hachi**, **Sasha** (SCOP only), **Colin** (COP only), **Rachel** (day repair). |
| **10F drift games with any “gap CO”** (seats 17, 19, 23, 28, 15) | **12 / 39** ≈ **31%** (union of games listing any of those IDs; see §4). |
| **10F drift games with Kindle (23)** | **8 / 39** ≈ **20.5%** |

**Top 5 Phase 11 fixes (ranked by `engine_gap × drift evidence` + rule centrality)** — recommendations only:

1. **Kindle (23)** — Implement D2D **+50% funds from owned cities only** in `GameState._grant_income` (and align “city” predicate with `PropertyState` / terrain; see Phase 10N). **Drift:** 8 games. **Note:** [AWBW CO Chart](https://awbw.amarriner.com/co.php) lists **urban attack** and powers, **not** city income; **prefer chart** for attack text, **flag income** as `co_data`/wiki/10N vs chart **discrepancy**.
2. **Sasha (19)** — D2D +100/property is implemented. **SCOP War Bonds:** CO chart — funds **50% of damage dealt** when attacking; **not** implemented in `_apply_attack` / treasury. **Drift:** 3 games include Sasha.
3. **Colin (15)** — D2D +100/property and **80%** build cost implemented. **COP Gold Rush:** multiply **current funds ×1.5** on activation — **not** found in `engine/game.py` power path (docstring claims “elsewhere”). **Drift:** 0 games with Colin in 10F sample.
4. **Hachi (17)** — [CO Chart](https://awbw.amarriner.com/co.php): **90%** unit cost D2D. Engine: **50%** only when building from `terrain.is_base` (`engine/action.py::_build_cost`). **Drift:** 0 games with Hachi in 10F sample.
5. **Rachel (28)** — CO chart: units repair **+1 additional HP** (liable for costs). Engine `_resupply_on_properties` uses fixed **+20** internal HP (`property_heal = 20`) with no Rachel branch. **Drift:** 1 game (1632355).

---

## Section 1 — Engine implementation summary (`GameState._grant_income`)

**Source:** `engine/game.py` lines **440–470** (see citation block below).

```440:470:D:\AWBW\engine\game.py
    def _grant_income(self, player: int) -> None:
        """
        Apply per-turn income to ``player``'s treasury using AWBW rules:
        1000g per owned income-property (HQ/base/city/airport/port), excluding
        comm towers and labs.

        CO modifiers applied here:
          * **Colin** (co_id 15) "Gold Rush" DTD — +100g per income-property.
          * **Sasha** (co_id 19) "Market Crash" DTD — +100g per income-property
            (mirrors Colin's per-property bonus; AWBW Sasha also gains funds on
            damage dealt, handled in combat — not here). Without this, mid- and
            late-game Sasha games consistently drift the engine treasury below
            AWBW's by ~100g × props × turn, surfacing as ``Build no-op
            (insufficient funds)`` at every Tank build (game ``1623012`` was
            the first traced case; the Sasha bucket dominates this cluster).

        SCOP / COP income multipliers (Colin's "Power of Money" funds ×1.5 of
        base income, Sasha's "Market Crash" funds drain on opponent) are
        modeled in ``_apply_power_effects`` / ``_apply_attack`` rather than
        here so the per-turn baseline stays clean.
        """
        n = self.count_income_properties(player)
        income = n * 1000

        co = self.co_states[player]
        if co.co_id == 15:  # Colin: base +100 per prop (DTD), COP: ×1.5 of base income
            income += n * 100
        elif co.co_id == 19:  # Sasha: +100g per income-property DTD (War Bonds)
            income += n * 100

        self.funds[player] = min(self.funds[player] + income, 999_999)
```

### 1.1 Base income

- **Per income property:** **1000** G.
- **Counted properties:** `count_income_properties(player)` — **excludes** comm towers and labs; **includes** income-producing properties (HQ, base, city, airport, port) per `engine/game.py` docstring and `PropertyState` usage.

### 1.2 CO-specific branches in `_grant_income`

| `co_id` | Name | Engine adjustment |
|--------:|------|---------------------|
| **15** | Colin | `+100 × n` income properties |
| **19** | Sasha | `+100 × n` income properties |
| *others* | — | **none** |

### 1.3 Weather

- **`state.weather`** is **not** read in `_grant_income`. [Weather (AWBW Wiki)](https://awbw.fandom.com/wiki/Weather) — confirm in Phase 11 if AWBW ever scales income by rain/snow; default Advance Wars behavior is **no** income weather modifier.

### 1.4 Spend-side (not `_grant_income`, but treasury)

**`engine/action.py::_build_cost`** (lines **697–709**):

- **Kanbei (3):** 120% of list cost.
- **Colin (15):** 80% of list cost.
- **Hachi (17):** **50%** of list cost **only** when `terrain.is_base` is true.

### 1.5 Commentary vs code

- `_grant_income` docstring attributes **Sasha** D2D to “War Bonds” naming; AWBW CO chart labels **+100/property** as D2D and **War Bonds** as the **SCOP** (damage-based funds). Engine implements **D2D +100** only.
- Docstring claims **Colin COP ×1.5** and **Sasha** damage funds are handled outside `_grant_income`; **grep** finds **no** `funds` multiply for Colin COP and **no** Sasha SCOP damage→funds in `_apply_attack` (Phase 11 should verify).

---

## Section 2 — Canon table (per CO)

**Primary source:** [AWBW CO Chart — `co.php`](https://awbw.amarriner.com/co.php) (retrieved 2026-04-21).  
**Secondary:** `data/co_data.json`, [AWBW Wiki Fandom — Category:Commanding Officers](https://awbw.fandom.com/wiki/Category:Commanding_Officers), per-CO pages where cited.

**Legend — rule class:** `NONE` | `INCOME%` | `UNIT_COST` | `PER_PROP_ADD` | `POWER_FUNDS` | `REPAIR_COST` | `SPECIAL`

| ID | Name | Income / treasury rule (canonical, CO chart–first) | Class | Notes / discrepancy |
|---:|------|------------------------------------------------------|-------|---------------------|
| 1 | Andy | No D2D income modifier | NONE | |
| 2 | Grit | No income modifier | NONE | |
| 3 | Kanbei | Units cost **+20%** (chart: “cost +20% more”) | UNIT_COST | Engine: `×1.2` in `_build_cost`. |
| 5 | Drake | No income modifier | NONE | |
| 7 | Max | No income modifier | NONE | |
| 8 | Sami | No income modifier | NONE | Capture speed ≠ treasury. |
| 9 | Olaf | No income modifier | NONE | |
| 10 | Eagle | No income modifier | NONE | |
| 11 | Adder | Chart: no D2D | NONE | `co_data.json` gives +1 movement D2D — **stats vs chart** (non-treasury). |
| 12 | Hawke | No D2D income | NONE | COP/SCOP HP drain/heal → repair costs (indirect). |
| 13 | Sensei | No D2D income | NONE | COP/SCOP spawns — **not** in `_grant_income`. |
| 14 | Jess | No D2D income | NONE | COP/SCOP resupply — indirect fuel/ammo. |
| 15 | Colin | **80%** unit cost; **−10%** attack (chart); **Gold Rush** COP: funds **×1.5**; **Power of Money** SCOP | UNIT_COST + POWER_FUNDS | CO chart **does not** list separate “+X per property” D2D line; **engine** adds **+100/property** in `_grant_income`. `co_data.json` +10% funds per property — **align with live AWBW**, not only chart one-liner. |
| 16 | Lash | No income modifier | NONE | |
| 17 | Hachi | **90%** unit cost D2D (chart); **Barter** / **Merchant Union** | UNIT_COST | `co_data.json`: 50% from cities only — **differs from chart** (prefer chart per mission). |
| 18 | Sonja | No income modifier | NONE | |
| 19 | Sasha | **+100 funds per income property** (chart); **Market Crash** COP; **War Bonds** SCOP: **50% of damage dealt** | PER_PROP_ADD + POWER_FUNDS | `co_data.json` D2D text wrong vs chart — **ignore JSON for D2D**. |
| 20 | Grimm | No income modifier | NONE | |
| 21 | Koal | No income modifier | NONE | |
| 22 | Jake | No income modifier | NONE | |
| 23 | Kindle | Chart: **+40% attack on urban** (HQ, base, city, airport, port, lab, com tower); **no D2D income line** | INCOME% (disputed) | **`co_data.json` + Phase 10N:** **+50% funds from owned cities** — **income rule not on CO chart**; **discrepancy**. |
| 24 | Nell | No income modifier | NONE | |
| 25 | Flak | No income modifier | NONE | |
| 26 | Jugger | No income modifier | NONE | |
| 27 | Javier | **Com tower** defense scaling; no income | SPECIAL | Towers excluded from income count. |
| 28 | Rachel | **+1 additional repair HP** (chart; liable for costs) | REPAIR_COST | Engine: no Rachel branch in `_resupply_on_properties`. |
| 29 | Sturm | No income modifier | NONE | |
| 30 | Von Bolt | No income modifier | NONE | |

---

## Section 3 — Engine vs canon delta (gaps)

| ID | Name | Topic | Engine | Canon (primary) | Match? | Priority |
|---:|------|-------|--------|-----------------|--------|----------|
| 23 | Kindle | D2D income | Flat `n×1000` (+0) | `co_data` + 10N: **+50% from cities**; CO chart **silent** | **N** (if `co_data`/wiki/live) | **HIGH** |
| 19 | Sasha | SCOP War Bonds | Not implemented | **50% of damage** as funds | **N** | **HIGH** |
| 15 | Colin | COP Gold Rush | Not found | **×1.5 current funds** | **N** | **MED** |
| 17 | Hachi | D2D build cost | 50% on `is_base` only | **90%** all builds (chart) | **N** | **HIGH** (rule) / **LOW** (10F drift) |
| 28 | Rachel | Day repair | Standard +2 display HP cap | **+1 extra** repair HP | **N** | **MED** |
| 15 | Colin | D2D attack | `atk_modifiers` empty in JSON | Chart: **−10%** ATK | **?** | **LOW** (combat) |
| 19 | Sasha | D2D | `+100×n` | Chart: **+100/property** | **Y** | — |
| 15 | Colin | D2D income | `+100×n` | Chart abbreviated; **data** +10% per property | **Likely Y** vs live | — |

**Weather × income:** no engine coupling; **OK** unless AWBW documents otherwise.

---

## Section 4 — Drift evidence (10F silent drift, `logs/phase10f_silent_drift.jsonl`)

**39** rows with `first_step_mismatch != null`.

### 4.1 Seat-agnostic CO appearance counts (sum of P0 + P1)

| `co_id` | Name | Appearances |
|--------:|------|------------:|
| 1 | Andy | 14 |
| 16 | Lash | 8 |
| 23 | Kindle | 9 |
| 14 | Jess | 6 |
| 7 | Max | 6 |
| 12 | Hawke | 5 |
| 8 | Sami | 5 |
| 22 | Jake | 4 |
| 10 | Eagle | 4 |
| 11 | Adder | 4 |
| 19 | Sasha | 3 |
| 20 | Grimm | 3 |
| 9 | Olaf | 3 |
| 5 | Drake | 1 |
| 18 | Sonja | 1 |
| 28 | Rachel | 1 |
| 30 | Von Bolt | 1 |

### 4.2 Games involving “gap COs” (17, 19, 23, 28, 15)

- **Kindle (23):** games **1628095, 1628324, 1628546, 1631755, 1632968, 1633242, 1633894, 1634522** → **8** games.
- **Sasha (19):** **1627328, 1632233, 1636157** → **3** games (rows with `first_step_mismatch` set; 1633133 has Sasha but `first_step_mismatch: null` — resign/truncation).
- **Rachel (28):** **1632355** → **1** game.
- **Hachi (17):** **0** games.
- **Colin (15):** **0** games.

**Union:** **12** distinct games (one game can count multiple gap COs; e.g. 1633133 has Von Bolt + Sasha).

**Interpretation:** High-frequency drift COs **Lash (16), Andy (1)** often have **no** D2D income special case — drift is **not** explained by `_grant_income` CO rules alone (aligns with Phase 10N taxonomy: spend slack, snapshot boundary, combat).

---

## Section 5 — Phase 11 fix priority (consolidated)

1. **Kindle** — `_grant_income` city income (+ reconcile CO chart vs `co_data` on whether income bonus exists in live AWBW).
2. **Sasha** — **War Bonds** SCOP: add `funds` on attack damage when `scop_active`.
3. **Colin** — **Gold Rush** COP: multiply `funds` by 1.5 on COP activation (and validate order vs AWBW).
4. **Hachi** — `_build_cost`: **90%** D2D per CO chart (replace or extend 50%-on-base heuristic).
5. **Rachel** — `_resupply_on_properties`: +1 display HP repair band with cost.

**Cross-cutting:** Oracle replay **build/repair** slack and **gzip vs `_grant_income` timing** (Phase 10N) remain **outside** pure CO income tables.

---

## Section 6 — Citations index

| # | Source | URL |
|---|--------|-----|
| C1 | AWBW CO Chart (primary rules text) | https://awbw.amarriner.com/co.php |
| C2 | AWBW Wiki — Category: Commanding Officers | https://awbw.fandom.com/wiki/Category:Commanding_Officers |
| C3 | AWBW Wiki — Kindle | https://awbw.fandom.com/wiki/Kindle |
| C4 | AWBW Wiki — Weather | https://awbw.fandom.com/wiki/Weather |
| C5 | Phase 10N context | `docs/oracle_exception_audit/phase10n_funds_drift_recon.md` |
| C6 | Engine `_grant_income` | `engine/game.py` |
| C7 | Engine build cost | `engine/action.py::_build_cost` |
| C8 | Drift register | `logs/phase10f_silent_drift.jsonl` |
