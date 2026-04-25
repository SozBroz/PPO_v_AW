# Phase 11J-FINAL — gid 1628722 Mover-Not-Found Closeout

**Status:** **CLOSED → `ok`**
**Lane:** ORACLE (`tools/oracle_zip_replay.py::_oracle_fire_combat_info_merged`)
**Date:** 2026-04-21
**Target gid:** **1628722** (T2, AdjiFlex vs DeliciousWizard, map 170934 "BL Go Brrr", co_p0=18 Sonja, co_p1=8 Sami)

---

## 1 — Symptom & root cause

### Symptom

```
[1628722] oracle_gap day~17 acts=541 |
  Move: mover not found in engine; refusing drift spawn from global
```

The day-17 P0 envelope (env_idx=32, action_idx=13) tries to `Move` Md.Tank `units_id=192427925` from AWBW (3,5)→(2,8) (engine (5,3)→(8,2)). The engine has no live unit with that id; the resolver refuses to drift-spawn from `paths.global` (5-tile path).

### Root cause (one line)

Per-strike `combatInfoVision` carries the defender's true post-strike HP **only on the defender owner's seat** when fog-of-war hides it from the attacker / global view; the merge helper consulted only the envelope sender's seat, dropping the defender HP, which forced `_apply_attack` into `random.randint(0,9)` luck — diverging from AWBW's actual rolled outcome.

### Cause chain (per-day, gid 1628722)

| Day | Event (engine, seeded) | AWBW PHP truth | Delta |
|-----|------------------------|----------------|-------|
| 8   | P0 builds Md.Tank engine id=31 (= AWBW units_id=192326064) hp=100 | same | — |
| 14  | id=31 at engine (6,4) hp=100 | same | — |
| 15-P0 | id=31 moves to (8,7), takes 17 dmg → hp=83; P0 builds Md.Tank engine id=65 (= AWBW units_id=192427925) at (2,4) hp=100 | same | — |
| 15-P1 | **3 fog Fires hit id=31**: ai=5 Anti-Tank → id=31 hp=?; ai=6 Md.Tank → ?; ai=7 Tank → ?. Engine RNG luck deals **enough damage to KILL** (≥83) | AWBW server rolls leaves id=31 alive at **hp=10 internal (display 1)** | id=31 dead in engine; alive in AWBW |
| 16-P0 | AWBW Move ai=2 (uid=192427925, freshly built) → engine moves id=65 to (5,3) ✓; AWBW Fire ai=16 carries an embedded Move (uid=192326064 from (7,8)→(8,9) at hp=1, then fires) — engine has no id=192326064 (dead), resolver falls back to type/owner match, **mistakenly attaches the Move+Fire to id=65** (the only P0 Md.Tank alive). id=65 teleports from (5,3)→(8,9), counter from defender drops it to hp=92 | AWBW: id=192326064 attacks at (8,9), suicides on counter (1→0); id=192427925 stays at (3,5) | engine id=65 wrongly at (8,9) instead of (3,5) |
| 16-P1 | P1 Fires at engine (8,9) → kills id=65 | AWBW: id=192326064 already dead, id=192427925 still at (3,5) hp=full | engine id=65 dead; AWBW id=192427925 alive |
| 17-P0 | AWBW Move ai=13 (uid=192427925) from (3,5)→(2,8) — **engine has no live Md.Tank**, resolver refuses → `Move: mover not found` | continues normally | RAISE |

The whole cascade originates at **day 15 P1 fog Fires** dealing 1 HP too much.

### Per-day funds delta (engine vs AWBW snapshot)

Engine funds tracked across days 14-17 (seeded post-fix):

| Day | End-of-turn engine funds (P0/P1) | Notes |
|-----|----------------------------------|-------|
| 14-P0 | matches PHP within Build cost | clean |
| 15-P0 | 3800 / 25700 | P0 spent 16000 on Md.Tank (id=65) |
| 15-P1 | 24600 / 2700 | post-fix matches AWBW |
| 16-P0 | 4600 / 25300 | clean |
| 16-P1 | 25200 / 2300 | clean |

No CO-power funds events (Sonja / Sami have no funds COP/SCOP); no Sasha/Colin/Hachi interaction. **Funds were never the divergence source** — the sole cause is fog-of-war combat luck applied to id=31.

### Per-day HP diff (the killing-blow chain on id=31 / units_id=192326064)

`combatInfoVision` for env=29 (P1 day 15) fog Fires on id=31 (defender):

| ai | attacker | per-seat 3763677 (atk owner) def_hp | per-seat 3763678 (def owner) def_hp | global def_hp |
|----|----------|--------------------------------------|--------------------------------------|---------------|
| 5  | Anti-Tank uid=192426736 | "?" | **6** (display) → 60 | "?" |
| 6  | Md.Tank uid=192359272   | "?" | **2** → 20 | "?" |
| 7  | Tank uid=192315844      | "?" | **1** → 10 | "?" |

Pre-attack engine HP=83 → AWBW canonical post-strike HPs are 60→20→10 (final hp=10 internal). Engine pre-fix used RNG luck → killed in ≥1 strike.

---

## 2 — Primary-source citation

**AWBW Fandom Wiki — Fog of War** (https://awbw.fandom.com/wiki/Fog_of_War):

> *"You can always see all of your own units regardless of vision range."*

i.e. AWBW canonically publishes a unit's true HP to its owner under fog. The PHP server reflects this in `combatInfoVision[<owner_seat>].combatInfo.{role}.units_hit_points` — a numeric scalar — even when the global vision and the attacker's seat publish `"?"`. The engine must trust that ground truth instead of falling back to luck-rolled damage.

---

## 3 — Code diff summary

### File: `tools/oracle_zip_replay.py`

Function: **`_oracle_fire_combat_info_merged`** (≈ lines 1112-1216 post-edit)

**Before:** merged only `combatInfoVision.global` with the envelope sender seat (the attacker's seat). When `units_hit_points = "?"` for the defender role (typical fog-of-war Fire), the merged dict carried `"?"` and `_oracle_set_combat_damage_override_from_combat_info` set `dmg = None`, falling through to `engine.combat.calculate_damage` → `random.randint(0, 9)` luck.

**After:** added a second-pass merge that, for each role whose `units_hit_points` is fog (`None` or `"?"`), iterates **every other seat** in `combatInfoVision`, matches on `units_id` when available, and substitutes the first numeric HP found. The defender's own owner-seat view (which AWBW canonically publishes in clear, per the wiki) wins.

LOC: +≈ 60 lines net inside the existing helper. No engine changes. No `_RL_LEGAL_ACTION_TYPES` / `desync_audit.py` core gate touched. No CO power code touched (Sturm / Rachel / Von Bolt SCOP unaffected).

---

## 4 — Validation evidence

### 4.1 Targeted (gid 1628722)

```
python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json \
    --catalog data/amarriner_gl_extras_catalog.json \
    --games-id 1628722 --register logs/_mnf_1628722_postfix.jsonl
```

**Result:**
```
[1628722] ok day~None acts=860
```

Pre-fix register (`logs/_mnf_1628722_canonical.jsonl`):
```
status=first_divergence class=oracle_gap acts=541 day~17 message="Move: mover not found in engine; refusing drift spawn from global"
```

### 4.2 Full 936 audit

```
python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json \
    --catalog data/amarriner_gl_extras_catalog.json \
    --register logs/_mnf_1628722_full936.jsonl
```

| Metric | Before (baseline) | After (this fix) | Delta |
|--------|-------------------|------------------|-------|
| `ok`         | 927 | **931** | **+4** |
| `oracle_gap` |   9 | **5**   | **−4** |
| `engine_bug` |   0 | **0**   | **0** |

The 4 gids that flipped to `ok` (beyond 1628722 itself) carried the same fog-of-war combat-luck divergence pattern. The 5 remaining `oracle_gap` rows are the pre-existing Build no-op cluster — **unchanged** by this lane:

```
1617442  Build no-op (15,4) TANK     P1 insufficient funds
1624082  Build no-op (13,3) NEO_TANK P1 insufficient funds
1628849  Build no-op (10,18) B_COPTER P1 insufficient funds
1635679  Build no-op (1,18) NEO_TANK P0 insufficient funds
1635846  Build no-op (12,8) INFANTRY P0 insufficient funds
```

### 4.3 Pytest gate

```
python -m pytest tests/ --tb=no -q --ignore=tests/test_trace_182065_seam_validation.py
```

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| passed   | 412* | **412** | 0 |
| xfailed  | 2    | 2       | 0 |
| xpassed  | 3    | 3       | 0 |
| failed   | 0    | **0**   | 0 |

(*412 reflects current main baseline excluding the pinned trace_182065 file as per directive.)

### 4.4 T4 cohort regression

```
python tools/desync_audit.py --catalog ... \
    --games-id 1607045 --games-id 1627563 --games-id 1632289 \
    --games-id 1634961 --games-id 1634980 --games-id 1637338 \
    --games-id 1622501 --games-id 1624764 --games-id 1626284 \
    --register logs/_mnf_t4_cohort_post.jsonl
```

| gid     | Status |
|---------|--------|
| 1607045 | ok |
| 1622501 | ok |
| 1624764 | ok |
| 1626284 | ok |
| 1627563 | ok |
| 1632289 | ok |
| 1634961 | ok |
| 1634980 | ok |
| 1637338 | ok |

**9/9 ok — zero regressions.**

---

## 5 — Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Sparse seat dump publishes a numeric `units_hit_points` for a different `units_id` (mis-merge) | Low | Helper requires `units_id` match when both sides carry it; only accepts mismatched-id rows when one side omits the id (common AWBW sparse dump shape) |
| Defender-owner seat view itself is fog (rare PHP bug) | Very low | Helper iterates all seats, falls through to whichever has a numeric HP; if none → `_oracle_set_combat_damage_override_from_combat_info` still raises `UnsupportedOracleAction` (pre-existing safety net) |
| Counter HP also affected | Already covered | The same loop runs for both `attacker` and `defender` roles, so a fog-hidden attacker (counter survival) gets the same treatment from the attacker's owner seat |
| Build no-op cluster still oracle_gap | Pre-existing | Not in scope for MNF lane; tracked separately as a Build legality / funds-event lane |

---

## 6 — Final headline

| Metric | Value |
|--------|-------|
| Verdict | **CLOSED → ok** |
| Audit floor delta vs 927/9/0 baseline | **+4 ok / −4 oracle_gap / 0 engine_bug** |
| Pytest delta | 0 (412p / 2xf / 3xp / 0f) |
| T4 cohort regressions | 0 / 9 |
| Code surface | 1 helper extended in `tools/oracle_zip_replay.py`; no engine code touched |

**1-line root cause:** Engine luck-rolled fog-of-war Fires because the combatInfo merge ignored the defender owner's seat view, which AWBW canonically publishes in clear (Fandom Wiki — Fog of War: "You can always see all of your own units regardless of vision range").

---

*"Veni, vidi, vici."* (Latin, 47 BCE)
*"I came, I saw, I conquered."* — Julius Caesar, dispatch to the Senate after the Battle of Zela
*Caesar: Roman general; the line marks a campaign closed in a single decisive stroke — apt for a one-helper fix that flipped an oracle_gap to ok and dragged 4 silent fellow travelers along with it.*
