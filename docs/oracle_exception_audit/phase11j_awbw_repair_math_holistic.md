# Phase 11J — AWBW repair math (holistic audit)

**Role:** Phase-11J-AWBW-REPAIR-MATH-AUDITOR (read-heavy).  
**Scope:** Canon from listed primaries vs `engine/game.py::_resupply_on_properties`, helper `_property_day_repair_gold`, Black Boat path `_apply_repair` / `_black_boat_heal_cost`, and regression tests in `tests/test_co_funds_ordering_and_repair_canon.py`.  
**Out of scope:** Sasha War Bonds; capture income ordering (except where repair articles explicitly tie them — they do not).

---

## 1. Canon summary (with primary quotes)

### 1.1 Property day repair (start of turn on owned tile)

**Source — AWBW Fandom — Units (Repairing and Resupplying):**  
https://awbw.fandom.com/wiki/Units  

- **Heal amount (simplified prose):** *"Any unit on an appropriate resupplying property will be restored by **2HP** (**3HP** with Rachel) and have their fuel and ammo completely refilled at the start of their turn."*  
- **Internal vs visual / Black Boat scale:** *"As unit health is actually calculated in increments down to the hundredths and not the tens (see Damage Formula), **repairs in AWBW restore up to 20 health each in actuality (10 with the Black Boat)**. Since visual health is rounded up, this means that a unit at 10 may actually be slightly damaged …"*  
- **Cost basis (visual, not fractional internal):** *"**Repair costs will be deducted for each visual hitpoint recovered during repair, not actual.** As visual unit health is indeed rounded up, this means that repairs that restore less than 1 full HP … will not cost funds, and are free."*  
- **Fringe / increments (supports 2-display-HP steps and “can’t pay → no heal”):** Under **Transports**, *"If a unit is **not specifically at 9HP**, repair costs will be calculated **only in increments of 2HP**. This can create a fringe scenario where a unit that is at **8 or less** with **<20%** of the unit's full value available … **will not be repaired**, even if a **1HP repair is technically affordable**."*  

**Source — AWBW Fandom — Advance Wars Overview (Economy):**  
https://awbw.fandom.com/wiki/Advance_Wars_Overview  

- *"**Repairs are handled similarly**, with money being deducted depending on the base price of the unit — **if the repairs cannot be afforded, no repairs will take place**."*  

**Source — AWBW Fandom — Changes in AWBW (repair increments):**  
https://awbw.fandom.com/wiki/Changes_in_AWBW  

- **Retrieval note:** Direct `curl`/HTTP fetch from this audit environment returned a **Cloudflare “Just a moment…”** challenge page (no article body). The following sentence is **verbatim as embedded in-repo** at `engine/game.py` (Rachel docstring, Tier-2 cite): *"**Repairs will only take place in increments of exactly 20 hitpoints, or 2 full visual hitpoints.**"* Treat as **secondary until re-fetched live**; it matches the Units “2HP / fringe” narrative and the engine’s PHP-aligned display-step model.

**Source — AWBW CO chart (Rachel D2D):**  
https://awbw.amarriner.com/co.php  

- Rachel row: *"**Units repair +1 additional HP (note: liable for costs).**"*  
  (Interpreted with Units’ “3HP with Rachel” and 20-internal “actuality” line → **+30 internal HP / +3 display bars** for a full property-day step when HP allows; engine matches that bundle per Phase 11Y PHP drill notes in `game.py`.)

**Source — Advance Wars Wiki — Repairing (optional cross-check):**  
https://advancewars.fandom.com/wiki/Repairing  

- **Retrieval note:** `web_fetch` **timed out** here. `game.py` cites: *"**10% cost per 10% health, or 1HP.**"* as cost proportionality for vanilla repairs — consistent with AWBW Units’ “per visual hitpoint recovered” cost model when combined with the 10/20 internal HP tick structure.

### 1.2 Bullet rules (synthesized for this engine)

- **Cost per display “bar” (property day, non-Rachel):** Typically **10% of listed unit cost** per **+10 internal HP** (+1 display bar) when the repair only restores one bar; **20%** when a **+20 internal** (+2 bars) step applies. (Follows Units’ visual-cost rule + 20-internal “actuality” cap per tick; fringe cases below.)  
- **When +1 vs +2 bars (non-Rachel property day):**  
  - **Display 10** (`ceil(internal/10) == 10`, internal **91–99**): **no repair, 0g** — bar already full in UI (`TestR4DisplayCapRepairCanon`, `game.py` R4 commentary).  
  - **Display 9** (internal **81–90**): **+1 bar** (+10 internal), **10%** cost.  
  - **Display ≤ 8** but **not** in the special band: **+2 bars** (+20 internal), **20%** cost (subject to funds ≥ cost, R2).  
  - **Display 8 with internal 71–80 (“9 HP fringe” band in internal coordinates):** **First** property-day tick is **+1 bar** (+10 internal) at **10%**; if that tick lands the unit at **internal 90**, a **same-morning chained** second **+1 bar** at **10%** completes 90→100 (`game.py` chained block; `test_display_8_internal_80_one_bar_ten_percent`).  
- **Display-10 “no repair”:** Even if internal &lt; 100, **no property-day heal** — aligns with “repair … **not actual**” + bar-maxed UI semantics enforced by PHP parity tests.  
- **Rachel (co_id 28):** **+3 display bars** intent (**+30 internal**) per chart + Units “3HP with Rachel”; **costs liable** (30% of listed cost for a full three-bar step when applicable). **Display 10 (91–99)** still **skipped** like other COs (`TestR4DisplayCapRepairCanon::test_rachel_display_10_no_heal_matches_php`). For **display ≤ 9**, engine uses `display_step = min(3, 10 - display_hp)` with proportional cost — **PHP-empirical** path (see `game.py:2399-2439`).  
- **Black Boat (explicit REPAIR action):** Units wiki: *"**1HP**"* restored + resupply; *"**if the player cannot afford the cost … it will only be resupplied and no repairs will be given**."`* Engine: **+10 internal** per successful heal, **10%** cost (`max(1, listed//10)`), **resupply always** (`game.py:1680-1765`).  
- **Insufficient funds → no partial heal (property day):** Overview quote + Units fringe + Black Boat quote; engine **R2**: full computed step cost must fit in treasury or **entire heal skipped** (`game.py:2336-2342`, `tests/...::TestR2AllOrNothing`).  
- **Iteration when multiple units compete for funds (R3):** Sort **`(prop.col, prop.row)` ascending** — column-major-from-left (`game.py:2306-2312`). Tier-4 RPGHQ cite documented in existing audit docs; tests assert PHP-matched behaviour under straddle.

---

## 2. Engine mapping table (branch → canon)

| Mechanism | `game.py` lines | Canon rule (source) |
|-----------|-----------------|---------------------|
| Income before opponent heal | `_end_turn` (see test module + `game.py` comments ~612+) | User-confirmed + PHP corpus; not part of Fandom repair math but orders treasury for repair. |
| `property_heal = 30` if Rachel else `20` | 2284-2285 | Units: *"2HP (3HP with Rachel)"* + co.php Rachel *"+1 additional HP"* → +20/+30 **internal** baseline narrative. |
| Eligible tile / unit-class matrix | 2314-2328 | Units: land on city/base/HQ; air on airport; sea on port; not enemy-owned. |
| Labs / comm towers excluded | 2318-2334 | Implicit in property lists; matches engine `qualifies_heal` + `not prop.is_lab` / `not prop.is_comm_tower`. |
| **R3** sort `(col, row)` | 2306-2312 | Tier-4 column priority (see `phase11j_f2_koal_fu_oracle_funds.md`); tests: `TestR3SortOrder`. |
| **R2** full-step affordance gate | 2336-2342, 2440-2445 | Overview: *"if the repairs cannot be afforded, no repairs will take place"*; Units fringe / transport bullet on 2HP increments. |
| `display_hp = (unit.hp + 9) // 10` | 2412 | Units: visual HP **rounded up**; cost *"per visual hitpoint recovered"*. |
| Branch `display_hp >= 10` → `step = 0`, `cost = 0` | 2413-2417 | Bar-maxed UI: no paid “phantom” internal top-up (`TestR4…`). |
| Non-Rachel `display_step` (9 → 1; display 8 ∧ 71–80 → 1; else 2) | 2418-2428 | Units: *"not specifically at 9HP"* 2HP increments + fringe; display-9 single bar; 71–80 band one-tick-first + chain (below). |
| Non-Rachel `cost` / `step` from `display_step` | 2427-2428 | 10% / 20% of listed cost per the visual-bar accounting. |
| Rachel `display_step = min(3, 10 - display_hp)` | 2437-2439 | co.php + Units 3HP Rachel; PHP drill cited in docstring. |
| **Chained second +10** when morning started 71–80 and `unit.hp == 90` after first tick | 2446-2468 | Completes PHP **two single-bar ticks** for high display-8 internals (e.g. 80→90→100) without treating 71–80 as a single +20 purchase. |
| Resupply fuel/ammo even if heal skipped | 2470-2476 | Units Black Boat quote pattern mirrored for property: resupply even when repair denied. |
| `_property_day_repair_gold(internal_heal, unit_type)` | 100-110 | **Linear gold helper** (20% per +20 internal equivalent); still used by **tools/tests**, not the hot path inside `_resupply_on_properties` after R4 (cost computed inline there). |
| Black Boat `_black_boat_heal_cost` / `_apply_repair` | 1680-1765 | Units: **1HP** heal, **10 internal** = 1 display bar; **10** “actuality” vs property **20**; unaffordable → resupply only. |

---

## 3. Verdict

**GREEN (engine matches cited canon for property-day repair),** with one **documentation retrieval caveat**:

- **Tier-2 Fandom texts actually retrieved in this run:** **Units** (full article text) and **Advance Wars Overview** (Economy paragraph) both support **visual-based charging**, **20-internal “actuality”** framing, **Rachel +1 display HP on top of the normal property heal** (read with the 2→3 HP sentence), **no repair when unaffordable**, and the **9HP / 2HP increment fringe** that motivates **all-or-nothing** and **display-8 band** behaviour.  
- **CO chart** Rachel line matches engine’s **+30 internal / liable** interpretation.  
- **Changes in AWBW:** **YELLOW only for *live page confirmation*** — HTTP retrieval hit **Cloudflare**; the **20 HP / 2 visual HP** increment quote is taken from **`game.py`’s own citation block** until a browser-backed fetch re-validates. No contradiction found between that sentence and the implemented display logic.

**Not RED:** No line range contradicts the retrieved Units + Overview + co.php bundle given AWBW’s known **PHP oracle** refinements (display-10 skip, 71–80 band, Rachel display-10 skip) already locked by `TestR4DisplayCapRepairCanon` and Rachel tests.

---

## 4. Risk list (watch flanks)

1. **Chained tick guard (`unit.hp == 90` only):** Second +10 runs only when **morning internal** was **71–80** and after the first tick **exactly 90** (`game.py:2449-2452`). If PHP ever applied a different chain condition, this would be the first line to re-probe.  
2. **`display_hp = ceil(internal/10)` vs PHP `hit_points` float:** Engine stores **integer internal HP** (0–100). PHP uses fractional internals for combat; **rounding residue** can sit inside one display bar. `desync_audit.py` documents **post-envelope HP sync** and **End.updatedInfo.repaired** exclusions precisely because **pinning PHP floats → engine int** plus a second **day-start repair** pass caused **double repair / funds bleed** (`desync_audit.py` ~577–643, ~711–719). **Do not remove** those rails without explicit user approval — they are **comparator hygiene**, not a claim that `_resupply_on_properties` is wrong.  
3. **`_property_day_repair_gold` drift from live path:** The helper still describes “+20 internal” linearity; **property-day repair cost** in `_resupply_on_properties` is **inline** after R4/Rachel branches. Tools importing the helper for drift math must stay aligned with **display_step** semantics or they will mis-predict.  
4. **Black Boat vs Rachel interaction:** Black Boat path does **not** special-case Rachel in `_apply_repair` (only **+10 internal** per REPAIR). If AWBW PHP ever applied Rachel D2D to Black Boat **target** heals, that would be a separate audit (not claimed by Units excerpts above).  
5. **Wiki internal contradiction (informational):** Units says *"hidden damage is accounted for during repair"* while also driving **display-10 no-charge** in PHP/engine — resolved in practice by **“visual hitpoint recovered”** costing **zero** when no full display bar is recovered; engine **skips** heal entirely at 91–99. Wording is ambiguous; behaviour is **test- and PHP-locked**.

---

## 5. Test suite “PHP-matched” assertions (sanity anchor)

File: `tests/test_co_funds_ordering_and_repair_canon.py`

- **R1:** Income before opponent `_resupply_on_properties` (`TestR1IncomeBeforeRepair`).  
- **R2:** All-or-nothing full step (`TestR2AllOrNothing`).  
- **R3:** `(col, row)` sort (`TestR3SortOrder`).  
- **R4:** Display-cap repair canon for non-Rachel + Rachel display-10 (`TestR4DisplayCapRepairCanon`), including **display-8 internal 73** (700g), **80** (700g+700g chain), **display-9** at 10%, **display-10** no-op.

These encode **oracle parity** beyond wiki prose alone.

---

## 6. Changelog

| Date | Author | Note |
|------|--------|------|
| 2026-04-21 | Phase-11J-AWBW-REPAIR-MATH-AUDITOR | Initial holistic audit document. |
