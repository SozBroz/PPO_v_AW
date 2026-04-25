# Phase 11Y-CO-WAVE-2 — Sasha / Colin / Rachel treasury & repair gaps (read-only)

READ-ONLY recon: replay inventory, envelope drills, and fix lanes for three Phase 10T HIGH-priority gaps. **Engine / oracle / tests were not modified.** One-off helper: `tools/_phase11y_co_wave2.py` (outputs `logs/phase11y_co_wave2_raw.json`).

**Primary rules source:** [AWBW CO Chart — co.php](https://awbw.amarriner.com/co.php) (fetched 2026-04-21). **`data/co_data.json` is secondary** (Sasha D2D/JSON text and Rachel D2D text disagree with the chart; see §5).

---

## Section 1 — Replay inventory (merged GL std catalogs + on-disk zips)

**Catalogs merged:** `data/amarriner_gl_std_catalog.json` + `data/amarriner_gl_extras_catalog.json` (last-wins on duplicate `games_id`).

| Metric | Value |
|--------|------:|
| Unique `games_id` rows after merge | **995** |
| Zips under `replays/amarriner_gl/` matching catalog + **std** `map_id` pool (`data/gl_map_pool.json`, `type == "std"`) | **936** |

**Per-CO counts (catalog: P0 or P1 has this `co_id`; zip pool: same, with zip on disk):**

| CO | `co_id` | Catalog appearances | Zips in 936 std pool |
|----|--------:|----------------------:|----------------------:|
| Sasha | 19 | **61** | **59** |
| Colin | 15 | **0** | **0** |
| Rachel | 28 | **75** | **69** |

**Colin:** There is **no** Colin (15) row anywhere in the merged catalog JSON (grep for `"co_p0_id": 15` / `co_p1_id` is empty). Phase 10T already noted **0** Colin games in the 10F silent-drift sample; the broader GL scrape used here still has **no** Colin fixtures.

**Candidate `games_id` lists (~10–20 each) for follow-up drilling:**

- **Sasha (19):**  
  `1619791, 1620301, 1623012, 1624082, 1625167, 1626181, 1626284, 1626386, 1627093, 1627110, 1627324, 1627328, 1627453, 1627495, 1627530, 1627564, 1628211, 1628287, 1628953, 1629104`  
  (59 total; SCOP **War Bonds** appears in **5** distinct games in the 936 pool — see §2.)

- **Colin (15):** *none in catalog / pool* — recommend **scraping or hand-adding** Colin replays before COP **Gold Rush** can be replay-quantified.

- **Rachel (28):**  
  `1607045, 1620320, 1622501, 1623070, 1623772, 1624181, 1624648, 1624721, 1624764, 1625211, 1625905, 1626642, 1626658, 1626722, 1626991, 1627563, 1627594, 1628195, 1628276, 1628357`  
  (69 total.)

**Heuristic (powers):** envelope scan for `{"action":"Power", "coPower":"S"}` (SCOP) / `"Y"` (COP) plus `coName` / seat from catalog. **Rachel D2D repair** applies on every property-day heal for Rachel-owned units; drills in §4 use snapshot diff, not COP/SCOP detection.

---

## Section 2 — Sasha (19) SCOP **War Bonds** drill (5 replays)

**Chart text:** *“War Bonds — Receives funds equal to 50% of the damage dealt when attacking enemy units.”* ([co.php](https://awbw.amarriner.com/co.php))

**Envelope evidence:** In the 936 zip pool, **5** games contain at least one `Power` row with `coName == "Sasha"` and `coPower == "S"` (`powerName == "War Bonds"`). Drill uses the **first** such row per game.

**Implementation note:** SCOP can appear **mid-envelope** after other actions (e.g. game `1624082` env 21: `Fire`×… then `Power` at sub-index 18, then `Build`×3, `End`). Drills must advance the oracle stream through the **Power** row, then sum primary strike damage (`game_log` `type == "attack"`, `dmg` internal HP) on **Sasha** `Fire` rows while `scop_active`, until Sasha’s `End` (engine clears SCOP in `_end_turn`).

| `games_id` | SCOP env / sub | Primary damage in War Bonds window (internal HP) | Expected funds from War Bonds `floor(0.5 × dmg)` | Clean PHP vs engine at Sasha `End`? |
|------------|----------------|--------------------------------------------------|----------------------------------------------------|--------------------------------------|
| 1624082 | 21 / 18 | **0** (post-SCOP: Build×3, End only) | **0** | **Yes** — funds match at end-turn snapshot (`18800` both) |
| 1626284 | 24 / 0 | **480** | **240** | **No** — large pre-SCOP treasury drift; end-turn gap dominated by upstream state |
| 1628953 | 30 / 0 | **520** | **260** | **No** — pre-SCOP drift |
| 1634267 | 22 / 0 | **290** | **145** | **No** — pre-SCOP drift |
| 1634893 | 26 / 0 | **370** | **185** | **No** — pre-SCOP drift |

**Gap quantification (War Bonds–specific):** Among SCOP windows with combat after activation, expected extra treasury vs current engine (which credits **nothing**) is **~145–260 G** per activation in these samples (**mean ~207 G** over the four non-zero rows). This is a **lower bound on the rule**; longer games with heavier SCOP combat would scale linearly with damage.

**Verdict on replay evidence:** **GREEN** that War Bonds SCOP exists in-pool and damage-after-SCOP is measurable; **YELLOW** that only one drilled game (`1624082`) had clean funds at end-turn, and that game had **no** post-SCOP fire (0 G expected). Do **not** defer Sasha on “no replay” — defer **implementation order** only (§6–7).

---

## Section 3 — Colin (15) COP **Gold Rush** drill (target 5)

**Chart text:** *“Gold Rush — Funds are multiplied by 1.5x.”* ([co.php](https://awbw.amarriner.com/co.php))

**Finding:** **Zero** Colin games in the merged catalog / 936 zip intersection → **no** `Power` / `coName Colin` / `coPower Y` rows to drill.

**Expected check (when a zip exists):** At COP envelope index `k`, compare PHP `players[].funds` for Colin’s `players_id` in `frame[k]` vs `frame[k+1]`; expect `post == int(pre * 1.5)` (confirm AWBW rounding vs `int(pre * 1.5)` in a future micro-drill).

**Gap quantification:** **N/A** in this pool.

**Verdict:** **RED** for *empirical* replay proof in the 936 cohort; **GREEN** for *rule clarity* from the CO chart and `co_data.json` COP description (“Current fund total is multiplied by 1.5”).

---

## Section 4 — Rachel (28) D2D **+1 repair HP** drill (10 replays)

**Chart text:** *“Rachel — Units repair +1 additional HP (note: liable for costs).”* ([co.php](https://awbw.amarriner.com/co.php))

**Engine today:** `_resupply_on_properties` uses fixed `property_heal = 20` internal HP (+2 display bars) and `_property_day_repair_gold` (20% of unit cost per full +20 internal banding per code comments).

**AWBW expectation (for this campaign):** On property-day heal, Rachel should heal **+30** internal HP (+3 display bars) vs **+20** for other COs, with **proportional** gold (user briefing: **30%** of unit cost for a full “Rachel band” vs **20%** for standard — aligns with “+50% more heal ⇒ +50% more cost” for a full heal step).

**Procedure:** For each of **10** Rachel-bearing zips (first rows from sorted zip list), run the same stepping + `compare_snapshot_to_engine` loop as `tools/replay_state_diff.py` (no sync), and record the **first step** reporting a **funds-only** mismatch involving Rachel’s seat.

| `games_id` | Outcome |
|------------|---------|
| 1607045 | Oracle abort (move path drift) — no funds conclusion |
| 1620320 | No funds mismatch through full replay |
| 1622501 | First funds mismatch step **13** — `P0` **+200** engine vs PHP (Rachel seat P0) |
| 1623070 | No funds mismatch |
| 1623772 | Step **11** — **+100** engine vs PHP |
| 1624181 | Step **15** — **+1800** engine vs PHP |
| 1624648 | No funds mismatch |
| 1624721 | Step **17** — **+300** engine vs PHP |
| 1624764 | Oracle abort (move path drift) |
| 1625211 | Step **14** — **+300** engine vs PHP (Rachel P1) |

**Interpretation:** When the engine **overstates** funds vs PHP at the same snapshot, it is consistent with **under-charging** property-day repair (Rachel should pay for **+1** extra bar). Magnitudes vary with **which unit** healed and **partial** heal steps (not every step is a full +20/+30 band). **1624181** shows a **large** positive engine delta — likely compounded with other drift, not repair alone.

**Verdict:** **YELLOW** — pattern matches the missing +1 HP / extra cost hypothesis on several games, but **not** all Rachel zips show funds drift early, and two zips hit unrelated oracle move errors.

---

## Section 5 — Per-CO fix design (where / what / LOC / citation)

### Sasha (19) — SCOP War Bonds

| Field | Proposal |
|-------|-----------|
| **Where** | `engine/game.py::_apply_attack`, after primary damage is applied to the defender (post-`dmg` / `defender.hp` update, alongside `_charge_power`), **not** on counterattack-only damage unless chart says otherwise (chart: “when **attacking** enemy units” — scope **primary strike** first; counter is a separate flank for Phase 12 if AWBW counts it). |
| **What** | If `att_co.co_id == 19` and `att_co.scop_active`, `self.funds[attacker.player] = min(999_999, self.funds[attacker.player] + int(dmg * 0.5))` (or treasury field if refactored). |
| **LOC** | ~5–10 |
| **Citation** | AWBW CO Chart — Sasha / War Bonds ([co.php](https://awbw.amarriner.com/co.php)) |
| **`co_data.json`** | SCOP name matches; D2D JSON is **wrong** vs chart (ignore for D2D). |

**Test cases (3–5):**

1. SCOP active, one attack deals 50 internal HP → **+25** funds.
2. SCOP active, attack deals 0 (miss / null damage) → **+0** funds.
3. SCOP inactive / COP only → no War Bonds credit.
4. Funds cap **999_999** after credit.
5. (Optional) Seam / no-defender attacks excluded or **0** credit per AWBW.

### Colin (15) — COP Gold Rush

| Field | Proposal |
|-------|-----------|
| **Where** | `engine/game.py::_apply_power_effects(player, cop)` — **only** when `cop is True` and `co.co_id == 15`, or a one-line helper called from `_activate_power` **after** bar reset if you need ordering vs other effects. |
| **What** | `self.funds[player] = min(999_999, int(self.funds[player] * 1.5))` **once** at COP activation (not SCOP). |
| **LOC** | ~3–5 |
| **Citation** | AWBW CO Chart — Colin / Gold Rush ([co.php](https://awbw.amarriner.com/co.php)); `co_data.json` COP text matches. |

**Test cases (3–5):**

1. COP at **10 000** → **15 000** funds.
2. COP at **1** → **1** or **2** per `int(*1.5)` (document expected rounding).
3. SCOP **Power of Money** does **not** multiply funds.
4. Cap at **999_999**.
5. COP with **0** funds stays **0**.

### Rachel (28) — D2D property-day repair

| Field | Proposal |
|-------|-----------|
| **Where** | `engine/game.py::_resupply_on_properties` — property-day auto heal only (not ` _apply_repair` Black Boat path). |
| **What** | If `self.co_states[player].co_id == 28`, use effective internal heal **30** (still capped by `100 - unit.hp` and funds loop), with per-step cost derived from **`_property_day_repair_gold`** using the same fractional rule but scaled so a full +30 step costs **30%** of unit cost (either extend `_property_day_repair_gold(internal_heal, …)` with a Rachel flag or branch cost calculation). |
| **LOC** | ~5–10 |
| **Citation** | AWBW CO Chart — Rachel repair line ([co.php](https://awbw.amarriner.com/co.php)); `co_data.json` D2D text is **not** trusted (mentions luck only). |

**Test cases (3–5):**

1. Rachel **Tank** (7000) at **70** internal HP on owned base → heal +30 if funds allow; cost **2100** for full band (vs **1400** for +20).
2. Insufficient funds → partial heal steps until gold exhausted (same loop structure as today).
3. Non-Rachel CO → still +20 / 20% behavior.
4. Air on **airport** / naval on **port** qualify per existing terrain gates.
5. **No** heal on lab / comm tower (existing guards).

---

## Section 6 — Coordination (in-flight lanes)

- **11J-FIRE-DRIFT** — touches `engine/game.py::_apply_attack` (selection / combat / damage accounting). **Sasha War Bonds** also wants a **post-damage treasury hook** in `_apply_attack`. **Treat as merge conflict / sequencing risk** — land FIRE-DRIFT first, then add War Bonds in the stabilized block, or combine in one reviewed PR with two logical commits.
- **11J-F2-KOAL** — `engine/action.py::compute_reachable_costs` — **no conflict** with these three fixes.

---

## Section 7 — Recommended dispatch order

1. **Colin COP** — smallest, localized `_apply_power_effects` branch; **no** Colin zips in pool, so gate with **unit tests + synthetic state** until a Colin replay appears.
2. **Rachel D2D repair** — isolated `_resupply_on_properties`; can ship in parallel with Colin; replay cohort is **rich** (69 zips).
3. **Sasha War Bonds** — **after 11J-FIRE-DRIFT** to avoid fighting in `_apply_attack`; replay evidence exists (5 SCOP games).

---

## Section 8 — Verdict

**YELLOW**

- **Sasha:** Clear fix path and in-pool SCOP samples; treasury parity drills are **noisy** due to **pre-existing** engine/PHP drift on several games; one clean end-turn match had **zero** post-SCOP combat.
- **Colin:** Chart rule is clear; **no** Colin replays in merged GL catalog / 936 zip set → **no empirical ×1.5 boundary check** here.
- **Rachel:** Multiple games show **small–medium** engine-high funds deltas consistent with **under-spending** on repair; two games failed earlier on **unrelated** oracle move errors.

---

## Artifacts

- `tools/_phase11y_co_wave2.py` — inventory + drills (stdout JSON).
- `logs/phase11y_co_wave2_raw.json` — latest run snapshot.

---

## Code citations (current engine, read-only)

`_grant_income` (Colin/Sasha D2D +100/prop; **no** Colin COP ×1.5 / no Sasha damage credit):

```449:493:c:\Users\phili\AWBW\engine\game.py
    def _grant_income(self, player: int) -> None:
        ...
        co = self.co_states[player]
        if co.co_id == 15:  # Colin: base +100 per prop (DTD), COP: ×1.5 of base income
            income += n * 100
        elif co.co_id == 19:  # Sasha: +100g per income-property DTD (War Bonds)
            income += n * 100

        self.funds[player] = min(self.funds[player] + income, 999_999)
```

`_apply_power_effects` — Sasha **COP** drain present; **no** Colin Gold Rush branch:

```611:615:c:\Users\phili\AWBW\engine\game.py
        # Sasha COP: drain power bar of enemy CO
        elif co.co_id == 19 and cop:
            self.co_states[opponent].power_bar = max(
                0, self.co_states[opponent].power_bar - (self.count_properties(player) * 9000)
            )
```

`_apply_attack` — primary damage and `_charge_power`; **no** Sasha War Bonds funds increment:

```708:723:c:\Users\phili\AWBW\engine\game.py
        if override_dmg is not None:
            dmg = max(0, int(override_dmg))
        else:
            dmg = calculate_damage(
                attacker, defender,
                att_terrain, def_terrain,
                att_co, def_co,
            )
        if dmg is not None:
            defender.hp = max(0, defender.hp - dmg)
            self.losses_hp[defender.player] += dmg  # Track HP lost
            if defender.hp == 0:
                self.losses_units[defender.player] += 1  # Track unit destroyed
            self._charge_power(attacker.player, defender.player, dmg)
```

`_resupply_on_properties` — fixed `property_heal = 20`; **no** Rachel branch:

```1540:1590:c:\Users\phili\AWBW\engine\game.py
    def _resupply_on_properties(self, player: int) -> None:
        ...
        property_heal = 20  # +2 display HP
        for unit in self.units[player]:
            ...
            if (
                qualifies_heal
                ...
            ):
                desired = min(property_heal, 100 - unit.hp)
                funds   = self.funds[player]
                h       = desired
                while h > 0:
                    cost = _property_day_repair_gold(h, unit.unit_type)
                    ...
```

---

*“In preparing for battle I have always found that plans are useless, but planning is indispensable.”* — Dwight D. Eisenhower, speech to the National Defense Executive Reserve Conference, 1957  
*Eisenhower: Supreme Allied Commander Europe in World War II and 34th U.S. President.*
