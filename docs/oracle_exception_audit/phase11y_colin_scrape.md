# Phase 11Y-COLIN-SCRAPE — Colin replay scrape & Gold Rush evidence (INFRA WRITE)

**Date:** 2026-04-21  
**Lane:** Replay infrastructure (`replays/amarriner_gl/`, `data/amarriner_gl_extras_catalog.json`, Colin batch catalog).  
**Engine / oracle / `desync_audit.py` sources:** not modified (per campaign constraints).

---

## Section 0 — Canonical wiki citations for Colin's mechanics

> **Imperator note on the URL.** The directive cited `https://awbw.amarriner.com/wiki/` as the canonical wiki. **That path returns HTTP 404** on the mirror — there is **no self-hosted wiki at that URL**. The two AWBW-side canonical sources are:
>
> - **`https://awbw.amarriner.com/co.php`** — the on-mirror **CO Chart**, served by the same host as the game itself.
> - **`https://awbw.fandom.com/wiki/Colin`** — the **AWBW community wiki on Fandom** (this is the wiki the AWBW community routinely cites; it is AWBW-specific, not generic Advance Wars).
>
> A **third** wiki — `https://advancewars.fandom.com/wiki/Colin` — covers the **console DS / BHR** games. It **disagrees** with the two AWBW canonicals on Power of Money's coefficient (it lists 3.33 % per 1000, the DS console value). For AWBW the two AWBW canonicals override; the disagreement is documented in §0.5 below.

### 0.1 Day-to-day: −20 % unit cost, −10 % attack

> **`co.php` (verbatim):**  
> *"Colin | Unit cost is reduced to 80 % (20 % cheaper), but lose −10 % attack. …"*  
> Source: `https://awbw.amarriner.com/co.php`

> **`awbw.fandom.com/wiki/Colin` (verbatim, "Day-to-Day Abilities"):**  
> *"Units cost −20 % less to build and lose −10 % attack."*  
> Source: `https://awbw.fandom.com/wiki/Colin`

### 0.2 CO Power — Gold Rush — funds × 1.5

> **`co.php`:** *"Gold Rush — Funds are multiplied by 1.5x."*  
> **`awbw.fandom.com/wiki/Colin`** ("CO Power Gold Rush"): *"Funds are multiplied by 1.5x."*

Both AWBW canonicals **agree** on the multiplier and are **silent on rounding** for non-integer halves (e.g. 50 835 × 1.5 = 76 252.5). Empirical PHP payload disambiguation in §7.

### 0.3 Super CO Power — Power of Money — formula

> **`co.php` (verbatim, exact formula):**  
> *"Power of Money — Unit attack percentage increases by `(3 * Funds / 1000)%`."*  
> Source: `https://awbw.amarriner.com/co.php`

> **`awbw.fandom.com/wiki/Colin` (verbatim, "Super CO Power Power of Money"):**  
> *"All units gain 3 % attack per 1000 funds."*  
> Source: `https://awbw.fandom.com/wiki/Colin`

These two formulations are **algebraically identical**: `(3 × Funds / 1000) %` == `3 % per 1000 funds`. Worked example confirmed by the wiki strategy text:

> *"having 30 000 spare funds will let you hit twice as hard with Power of Money active."*  
> Source: `https://awbw.fandom.com/wiki/Colin` ("Strategy" section)

Check: `30 000 × 3 / 1000 = +90 %` attack. Combined with the universal `+10 %` SCOP attack/defense modifier (cited next), Colin's units in Power of Money at 30 000 funds attack at base + 90 % + 10 % = **+100 %** over D2D — i.e. "twice as hard." ✔

### 0.4 Universal COP / SCOP rider

> **`co.php` (verbatim footer):**  
> *"Note: All CO's get an additional +10 % attack and defense boost on COP and SCOP."*  
> Source: `https://awbw.amarriner.com/co.php`

This rider stacks with Colin's D2D −10 % during COP/SCOP (so during Gold Rush, Colin's units net 90 % × 1.10 ≈ 99 % attack vs baseline; during Power of Money the +10 % stacks with the funds-scaled bonus).

### 0.5 Disagreement with non-AWBW wiki — escalation note

The third wiki (`https://advancewars.fandom.com/wiki/Colin` — generic Advance Wars wiki, **not** AWBW-specific) lists **Power of Money as "3.33 % more firepower per 1000 War Funds"** under the **AW2: BHR** section. That value is the **console-DS** formula (10/3 % per 1000, derived from the DS-engine integer arithmetic). It **disagrees** with both AWBW canonicals.

**Resolution:** AWBW is a **separate engine** from the DS port. Both AWBW-host sources (`co.php` and `awbw.fandom.com`) say `3 × Funds / 1000`. We **adopt the AWBW value** for this engine. **No silent pick** — the disagreement is documented here for the Imperator's review. If later guidance reverses this (e.g. for parity with the console games), call it out and we re-baseline the SCOP coefficient.

---

## Section 1 — Search methodology + sources

1. **Why GL std catalog had zero Colin:** `data/gl_map_pool.json` places **Colin (`co_id` 15) in tier T0**, and **T0 is disabled** for Global League std rotation. Competitive GL std completed listings therefore skew to enabled tiers (T1–T4 CO sets) — consistent with Phase 11Y CO-WAVE-2 finding **0** Colin rows in the merged std/extras cohort intersecting the 936-zip std map pool.

2. **What worked — global completed listing:** Paginating  
   `https://awbw.amarriner.com/gamescompleted.php?start={1,51,101,…}`  
   and parsing **player row** CO portraits via  
   `id="do-game-player-row" … class='co_portrait' … ?v=<co_id>?v=`  
   (with a fallback to the legacy GL listing regex where that layout appears).  
   This yields real **1v1 games where a player selected Colin**, not `co_id=15` (that parameter surfaces *ban-list* portraits and is misleading for “plays Colin”).

3. **What did not work:**
   - **`gamescompleted.php?co_id=15`** — CO portraits in the **Bans & Limits** strip polluted parsing; semantically this is not “Colin as combatant.”
   - **`gamescompleted.php?players_co_id=15` / `playing_co_id=15` / …** — tested query variants behaved like **unfiltered** listings (no reliable Colin-only filter in those probes).
   - **`browsegames.php`** — 404 on the mirror tested.

4. **Tier / map pool:** These games are **not** restricted to GL std maps; scraped `map_id` values are mostly **outside** `gl_map_pool.json` `type == "std"`. That is expected for Colin-heavy casual / T0 / custom maps.

---

## Section 2 — Colin games found + downloaded

| Metric | Count |
|--------|------:|
| Unique Colin games identified (global crawl, first 15 hits) | **15** |
| Zips downloaded successfully | **15** |
| ReplayVersion 1 (no `a<games_id>` action stream) | **1** (`1637705.zip`, ~1.7 KB) |

**Download command:**  
`python tools/amarriner_download_replays.py --catalog data/amarriner_gl_colin_batch.json --allow-non-gl-std-maps --sleep 0.75`

**Output directory:** `replays/amarriner_gl/{games_id}.zip` (default; no separate `amarriner_gl_colin/` folder).

**`games_id` list:**  
`1636107, 1558571, 1637153, 1638360, 1637096, 1638136, 1636411, 1620117, 1628024, 1629555, 1637705, 1358720, 1636108, 1619141, 1637200`

---

## Section 3 — Normalization results

Per `awbw-replay-ingest`, `amarriner_download_replays.py` runs **`run_normalize_map_to_os_bm`** after each successful save.

**Outcome:** **warnings only** — for this batch, **none** of the `map_id` values had a local `data/maps/<map_id>.csv` in the repo. Normalization is a no-op until those CSVs exist (expected for arbitrary user maps). No CSVs were added in this lane.

---

## Section 4 — Catalog updates

| Artifact | Action |
|----------|--------|
| `data/amarriner_gl_colin_batch.json` | **Created** — 15 games, `co_p0_id` / `co_p1_id` from listing portrait scrape; titles from `game.php`; `map_id` from `game.php` / listing. |
| `data/amarriner_gl_extras_catalog.json` | **Updated** — **+15** rows merged (`n_games`: 195 → **210**), `meta.colin_batch_note` + `updated_at` set. |
| `data/amarriner_gl_std_catalog.json` | **Not touched** (per constraints). |

---

## Section 5 — Power envelope verification (Colin / COP)

Used `tools.oracle_zip_replay.parse_p_envelopes_from_zip` (same envelope source as oracle tooling). Scanned for `action == "Power"`, `coName == "Colin"`, `coPower == "Y"` (COP), `powerName == "Gold Rush"`.

| `games_id` | Colin Power rows | COP (`coPower == "Y"`) |
|-------------|-----------------:|------------------------:|
| 1636107 | 6 | 6 |
| 1558571 | 14 | 14 |
| 1637153 | 10 | 9 |
| 1638360 | 4 | 4 |
| 1637096 | 1 | 0 (SCOP only) |
| 1638136 | 1 | 0 (SCOP only) |
| 1636411 | 9 | 4 |
| 1620117 | 37 | 35 |
| 1628024 | 3 | 1 |
| 1629555 | 5 | 5 |
| 1637705 | 0 | 0 (RV1 — no `p:` stream) |
| 1358720 | 22 | 22 |
| 1636108 | 4 | 4 |
| 1619141 | 6 | 6 |
| 1637200 | 3 | 2 |

**Summary:** **12 / 14** actionable RV2 zips contain **≥1** Colin **COP** Gold Rush envelope (**≥5** requirement satisfied).  
Helper: `tools/_colin_envelope_scan.py` (re-run for verification).

---

## Section 6 — Desync audit on Colin pool

**Command:**  
`python tools/desync_audit.py --catalog data/amarriner_gl_colin_batch.json --register logs/desync_register_colin_pool.jsonl --seed 1`

**Result:**

```
zips_matched=0 filtered_out_by_map_pool=15 filtered_out_by_co=0
```

**Cause:** `desync_audit` only audits zips whose `map_id` is in the **GL std** map pool (`tools/gl_std_maps.gl_std_map_ids`). **All** Colin batch maps are non-std here, so **no register file was written** (tool exits after “no zips matched”).

**Interpretation:** This is **tool filtering**, not “Colin zips missing.” Extending audit to non-std maps would require a **future** `desync_audit` flag or a separate harness — **out of scope** for this infra lane (and `desync_audit.py` was not edited).

---

## Section 7 — Gold Rush drill (×1.5 treasury) — wiki vs PHP

**Wiki rule (citation):** "Funds are multiplied by 1.5x." — both `https://awbw.amarriner.com/co.php` ("Gold Rush — Funds are multiplied by 1.5x.") and `https://awbw.fandom.com/wiki/Colin` (CO Power: Gold Rush). Wiki is **silent on rounding** for non-integer halves.

### 7.1 Why naive `frame[k]` vs `frame[k+1]` is not a clean test

For `1636107`, first COP at envelope **22**: Colin `players_id` funds **23 800 → 26 100** between `frame[22]` and `frame[23]` — **not** `int(23 800 × 1.5) = 35 700`. PHP **trailing snapshots aggregate the whole half-turn**: income, COP fund replacement, **and all subsequent Build / Repair / Supply spending** in the same envelope. Boundary frames therefore cannot isolate Gold Rush by themselves.

### 7.2 Strict drill — PHP `frame[k]` (pre-envelope) vs server-authoritative COP payload

For every Colin COP envelope **with the `Power` action at sub-index 0** (i.e. nothing fires before COP within the envelope), pre-COP funds are exactly the snapshot value `frame[k].players[Colin].funds`. The COP `Power` JSON carries the post-COP authoritative value at `playerReplace.global[<players_id>].players_funds`. Both come from the same PHP server — both are oracle-grade.

Helper: `tools/_colin_gold_rush_drill_strict.py` (dispatches by `Power.playerID` so Colin-vs-Colin mirror games attribute funds to the correct seat).

**Result across 12 RV2 Colin zips, 22 sub=0 COP envelopes:**

| Stratum | Match `payload == round_half_up(pre × 1.5)` |
|---|---:|
| Envelopes whose `Power` JSON carries `playerReplace.players_funds` | **15 / 15** |
| Envelopes whose `Power` JSON omits `playerReplace.players_funds` (older format variant) | **0 / 7** payload available — not falsifiable from snapshot alone |

**15 / 15 perfect agreement** between wiki rule and PHP payload on the strata that carry it. The 7 non-payload cases (concentrated in `1620117` Colin-mirror fog and `1358720` / `1558571`) do **not contradict** the rule — those PHP frames either omit per-half-turn fund mutations or aggregate them into post-spend values, matching the snapshot-aggregation caveat in §7.1.

### 7.3 Wiki silence on rounding — empirically resolved

Three of the 15 payload cases land on the `.5` boundary and disambiguate AWBW's rounding convention:

| zip | env | pre | `int(pre × 1.5)` (floor) | `round_half_up(pre × 1.5)` | PHP payload |
|---|---:|---:|---:|---:|---:|
| `1637153` | 38 | 50 835 | 76 252 | 76 253 | **76 253** |
| `1637153` | 44 | 48 533 | 72 799 | 72 800 | **72 800** |
| `1619141` | 35 | 23 331 | 34 996 | 34 997 | **34 997** |

AWBW uses **round-half-up** (PHP's default `round()` mode) for `funds × 1.5`, **not** `int()` / floor. The wikis don't say this — the PHP payload does. Implementation must use `round_half_up`, otherwise it will silently desync on roughly **20 %** of COP fires (the .5 boundary class).

### 7.4 Sample of strict-drill output

```
1637153.zip env= 20 pre=44400 exp_round= 66600 payload= 66600 [OK round_half_up]
1637153.zip env= 24 pre=41360 exp_round= 62040 payload= 62040 [OK round_half_up]
1637153.zip env= 38 pre=50835 exp_round= 76253 payload= 76253 [OK round_half_up]   <-- .5 case
1619141.zip env= 17 pre=20420 exp_round= 30630 payload= 30630 [OK round_half_up]
1619141.zip env= 35 pre=23331 exp_round= 34997 payload= 34997 [OK round_half_up]   <-- .5 case
1629555.zip env= 33 pre=24000 exp_round= 36000 payload= 36000 [OK round_half_up]
```

**Conclusion:** AWBW canonical wikis (`co.php`, `awbw.fandom.com/wiki/Colin`) and PHP payloads **agree** on Gold Rush. Neither AWBW wiki specifies the rounding convention; the **PHP payload reveals round-half-up**. The non-AWBW wiki at `advancewars.fandom.com` is silent on Gold Rush rounding and irrelevant for AWBW seat behavior. **No agent-side disagreement to escalate** for Gold Rush itself; the rounding convention is a **wiki-silence resolution**, not a wiki-vs-PHP conflict.

---

## Section 8 — Verdict

**GREEN**

- **≥5** ingested RV2 zips with Colin **COP** Gold Rush envelopes (**12** zips).
- **Gold Rush rule** (wiki: "funds × 1.5") empirically confirmed against PHP payload on **15 / 15** sub=0 envelopes carrying `playerReplace.players_funds`.
- **Rounding convention** (wiki silent → PHP authoritative): **round-half-up**, confirmed on 3 boundary cases.
- **Wiki-vs-PHP conflict for AWBW Colin mechanics: none.**
- **Inter-wiki conflict** (AWBW canonical vs generic Advance Wars wiki on Power of Money 3 % vs 3.33 %): AWBW wins for AWBW seat; logged in §0.5 for Imperator review.

Caveat: **1** zip is RV1-only (`1637705`); **2** zips have Colin SCOP but no COP in the sample.

---

## Section 9 — Recommendation for Gold Rush implementation lane

1. **Implement Gold Rush** at COP activation for `co_id == 15`:
   - `funds_post = min(round_half_up(funds_pre * 1.5), 999_999)` — **NOT** `int(funds_pre * 1.5)`.
   - Anchored by `co.php` ("Funds are multiplied by 1.5x") + `awbw.fandom.com/wiki/Colin` + 15 / 15 PHP payload empirical agreement.
2. **Implement Power of Money** as **`+ (3 × funds / 1000) %` attack** (formula from `co.php`; equivalent to `awbw.fandom.com` "3 % per 1000"). Stacks on top of the universal +10 % SCOP attack/defense rider. **Funds are not consumed** by SCOP.
3. **Implement D2D** as 80 % unit cost, −10 % attack — both AWBW canonicals agree.
4. **Unit tests** must include a `.5` boundary case (e.g. `pre=50 835 → post=76 253`), at least one Colin-vs-Colin mirror dispatch case, and one SCOP attack-bonus case at `funds=30 000` (expect +90 % from PoM + 10 % universal SCOP rider).
5. **Desync regression:** Colin maps are non-std; until `desync_audit` supports optional non-std catalog runs, treat Colin replays as **oracle / manual** fixtures.

---

## Reproducibility helpers (repo)

| Script | Purpose |
|--------|---------|
| `tools/_colin_scan_global.py` | Paginate `gamescompleted.php?start=` and collect Colin games. |
| `tools/_colin_fetch_meta.py` | Fetch `game.php` titles / `map_id` (CO ids patched from scan in batch). |
| `tools/_colin_envelope_scan.py` | Colin `Power` / COP counts per zip. |
| `tools/_colin_gold_rush_drill.py` | Frame-pair experiment + documents `playerReplace` check (early version). |
| `tools/_colin_gold_rush_drill_strict.py` | **Strict drill** — sub=0 COP envelopes only, dispatches by `Power.playerID` for Colin-vs-Colin mirrors, `round_half_up` reference. **15 / 15 OK**. |

---

*“In the practical use of our intellect there is a certain fundamental something about which nothing that we are taught can say anything, nor may anything be said about it by anyone who has not been through what I have been through.”* — Carl von Clausewitz, *On War* (early 19th c.; common English translation)  
*Clausewitz: Prussian general and military theorist.*
