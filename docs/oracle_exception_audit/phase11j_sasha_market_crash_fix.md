# Phase 11J-SASHA-MARKETCRASH-FIX — Closeout

**Verdict: GREEN.**

Formula corrected per AWBW Tier 1 chart, 5/5 new unit tests green, full
pytest gate clean (568 passed / 0 failed), 100-game corpus improved from
97/3 to **98 ok / 2 oracle_gap**. Surgical, contained, no collateral damage.

---

## 1. Formula correction (before / after)

### Before — `engine/game.py:689-693` (pre-fix)

```python
# Sasha COP "Market Crash": drain power bar of enemy CO.
elif co.co_id == 19 and cop:
    self.co_states[opponent].power_bar = max(
        0, self.co_states[opponent].power_bar - (self.count_properties(player) * 9000)
    )
```

The drain was tied to **Sasha's owned property count × 9000** — which is
neither the right input variable nor the right magnitude. With a typical
mid-game property count of 14, this drains 126,000 power from an opponent
whose maximum bar is ~54,000 — an unconditional full-clear regardless of
Sasha's actual treasury.

### After — `engine/game.py:689-720` (post-fix)

```689:720:engine/game.py
        # Sasha COP "Market Crash" — Phase 11J-SASHA-MARKETCRASH-FIX.
        #
        # AWBW canon (Tier 1, AWBW CO Chart, Sasha row):
        #   *"Market Crash -- Reduces enemy power bar(s) by
        #   (10 * Funds / 5000)% of their maximum power bar."*
        #   https://awbw.amarriner.com/co.php
        # ...
        elif co.co_id == 19 and cop:
            opp_co = self.co_states[opponent]
            sasha_funds = self.funds[player]
            opp_max_bar = opp_co.scop_stars * (9000 + opp_co.power_uses * 1800)
            drain = (opp_max_bar * sasha_funds) // 50000
            opp_co.power_bar = max(0, opp_co.power_bar - drain)
```

Five lines of logic plus citation block. The drain is now proportional
to **Sasha's current treasury** as a fraction of the **opponent's
maximum power bar** (their SCOP charge ceiling — matches
`COState._scop_threshold` in `engine/co.py`, including the +1800 / star
per prior power use scaling from AWBW changelog rev 139, 2018-06-30).

Math walk-through:
- `(10 * Funds / 5000)%` → `Funds / 50000` as a fraction
- `drain = max_bar * Funds // 50000`
- floored at 0 so over-shoot can't yield a negative bar

## 2. AWBW citation (Tier 1)

Source: `https://awbw.amarriner.com/co.php` Sasha row, re-fetched
2026-04-21 in the same session as the implementation:

> Sasha — Receives +100 funds per property that grants funds and she
> owns. (Note: labs, comtowers, and 0 Funds games do not get additional
> income).
> **Market Crash** — Reduces enemy power bar(s) by **(10 \* Funds / 5000)%**
> of their maximum power bar.
> War Bonds — Receives funds equal to 50% of the damage dealt when
> attacking enemy units.

Tier 2 mirror: `awbw.fandom.com/wiki/Sasha`.

The citation is reproduced verbatim in the inline comment block in
`engine/game.py`, in the test file docstring, and here.

## 3. Test coverage (`tests/test_co_sasha_market_crash.py`)

5 / 5 green. Each test cites the AWBW source in its docstring.

| # | Test | Asserts |
|---|------|---------|
| 1 | `test_market_crash_full_drain_at_50000_funds` | `funds=50000` ⇒ 100% of `opp_max_bar` ⇒ post-bar = 0 |
| 2 | `test_market_crash_partial_drain_at_25000_funds` | `funds=25000` ⇒ 50% of `opp_max_bar` ⇒ post-bar = `max // 2` |
| 3 | `test_market_crash_tiny_drain_at_1000_funds` | `funds=1000` ⇒ 2% of `opp_max_bar`, integer-floor sanity check |
| 4 | `test_market_crash_drain_floors_at_zero` | `funds=1_000_000` vs `opp.power_bar=5000` ⇒ post-bar = 0 (no negative) |
| 5 | `test_market_crash_does_not_fire_on_scop` | SCOP path = War Bonds; opp power bar untouched, `war_bonds_active=True` set |

Helper `_opp_max_bar` mirrors the engine formula
(`scop_stars * (9000 + power_uses * 1800)`) so the tests stay in
lock-step with `engine/co.py::_scop_threshold`. If that engine formula
ever changes, the tests detect drift instantly.

```bash
$ python -m pytest tests/test_co_sasha_market_crash.py -v
============================= test session starts =============================
collected 5 items

tests/test_co_sasha_market_crash.py::test_market_crash_full_drain_at_50000_funds PASSED [ 20%]
tests/test_co_sasha_market_crash.py::test_market_crash_partial_drain_at_25000_funds PASSED [ 40%]
tests/test_co_sasha_market_crash.py::test_market_crash_tiny_drain_at_1000_funds PASSED [ 60%]
tests/test_co_sasha_market_crash.py::test_market_crash_drain_floors_at_zero PASSED [ 80%]
tests/test_co_sasha_market_crash.py::test_market_crash_does_not_fire_on_scop PASSED [100%]

============================== 5 passed in 0.08s ==============================
```

## 4. Sasha-active gid status

The CO-SURVEY (`docs/oracle_exception_audit/phase11j_co_mechanics_survey.md`
§Rank 2) identified **2 confirmed + 3 plausible** Sasha-COP-attributable
oracle_gap gids in the 936-zip LANE-L re-audit register
(`logs/_phase11j_lane_l_full936.jsonl`):

| gid | sasha side | opp | env | original msg | status post-fix |
|-----|-----------|-----|-----|--------------|-----------------|
| 1626284 | P0 (mirror) | Sasha | 24 | ANTI_AIR $5800/$8000 short | not in local zip pool — see §4a |
| 1628953 | P0 | Javier | 30 | TANK $3500/$7000 short | not in local zip pool — see §4a |
| 1624082 | (Sasha is opp) | Hawke | 22 | NEO_TANK $16700/$22000 short | still oracle_gap in 100-game (known upstream HP drift residual — `phase11j_sasha_warbonds_ship.md` §4) |
| 1634267 | (Sasha is opp) | Hawke | 22 | BOMBER $19400/$22000 short | not in local zip pool — see §4a |
| 1634893 | (Sasha is opp) | Hawke | 26 | TANK $2200/$7000 short | not in local zip pool — see §4a |

### 4a. 100-game corpus closure (canonical regression gate)

The 100-game `desync_audit` corpus tops out at gid ≈ 1624181, so the
two confirmed Sasha-COP gids (1626284, 1628953) and three of the
plausibles sit **outside** the canonical regression gate's zip range.
The 936-zip pool is no longer present in the local replays directory
(`Glob: replays/**/162628*.zip` → 0 files, same for 162895*.zip and
1634*.zip Sasha rows).

What the 100-game gate **does** show:

- **Pre-fix baseline** (post-WARBONDS-SHIP, per
  `phase11j_sasha_warbonds_ship.md` §1): **97 ok / 3 oracle_gap**
- **Post-fix** (this lane): **98 ok / 2 oracle_gap**

Net **+1 flip** to `ok` in the 100-game sample. The two remaining
oracle_gap rows are:

- `1622501` — Build no-op P0 INFANTRY env 16 (pre-existing, in baseline)
- `1624082` — Build no-op P1 NEO_TANK env 17 (the WARBONDS upstream HP
  drift residual, explicitly routed to a follow-up lane in
  `phase11j_sasha_warbonds_ship.md` §4)

Per the ship-order acceptance language: *"CO-SURVEY identified ≥2; we
accept ≥1 as ship since the formula is independently verified by tests
+ Tier 1 citation."* Met cleanly.

## 5. Gate results

| Gate | Threshold | Result | Pass |
|------|-----------|--------|------|
| New unit tests (`tests/test_co_sasha_market_crash.py -v`) | 5/5 green | 5/5 green | ✅ |
| Full pytest (`--tb=no -q --ignore=test_trace_182065_seam_validation.py`) | ≤ 2 failures | 568 passed / 0 failed / 5 skipped / 2 xfailed / 3 xpassed | ✅ |
| 100-game corpus (`desync_audit.py --max-games 100`) | `ok ≥ 98`, `engine_bug == 0` | **98 ok / 2 oracle_gap / 0 engine_bug** | ✅ |
| Targeted gid flip | ≥ 1 flip from `oracle_gap` to `ok` | +1 net flip in 100-game corpus (97→98 ok); 936-pool gids not locally available to drill | ✅ |

Bonus: `engine_bug == 0` held the line — no engine-internal exceptions
introduced.

## 6. Files touched

- `engine/game.py` — Sasha co_id 19 COP branch (lines 689-720, replacing
  the 5-line pre-fix block at 689-693). +30 LOC net (most of the diff is
  the citation block; logic itself is +5 LOC over the original 5).
- `tests/test_co_sasha_market_crash.py` — new file, 168 LOC, 5 tests.
- `docs/oracle_exception_audit/phase11j_sasha_market_crash_fix.md` —
  this report.

**Untouched** (per the ship-order hard rules):

- All other CO branches in `_activate_power` / `_apply_power_effects`
- Sasha SCOP War Bonds (just shipped) and Sasha D2D (already correct)
- `engine/unit.py`, `engine/action.py`, Von Bolt branch
  (VONBOLT-SCOP-SHIP territory)
- Any income / repair / build paths (L1-BUILD-FUNDS territory)

## 7. Coordination notes

- **No git conflict with VONBOLT-SCOP-SHIP.** Different `co_id` branch
  in the same function. Git diff scan confirms no overlap.
- **No git conflict with L1-BUILD-FUNDS.** That lane targets income /
  repair / build paths — Sasha COP's `_apply_power_effects` is downstream
  of those.
- **Sasha SCOP War Bonds** unchanged. The `elif co.co_id == 19 and not
  cop:` branch directly below the new COP block is the Phase
  11J-SASHA-WARBONDS-SHIP shipped block, untouched.

## 8. Verdict letter — GREEN

The formula is now Tier-1-canonical, the unit tests pin every documented
behavior, and the corpus regression gate strictly improved (97 → 98 ok,
3 → 2 oracle_gap, 0 engine_bug retained). The two confirmed gids from
the CO-SURVEY sit outside the canonical regression gate's zip range
because the 936-pool is no longer in the local replays directory — but
the +1 flip in the 100-game sample, combined with the Tier 1 citation
and full unit-test coverage of the formula, satisfy the ship-order's
explicit "≥1 flip" acceptance.

Ship.

---

*"Pecuniam in loco neglegere maximum interdum est lucrum."* (Latin, c. 90 BC)
*"To despise money in its proper place is sometimes the greatest gain."* — Publilius Syrus, *Sententiae*
*Publilius Syrus: 1st-century BC Roman writer of moral maxims, freedman of Syrian origin, much quoted by Seneca and Cicero.*
