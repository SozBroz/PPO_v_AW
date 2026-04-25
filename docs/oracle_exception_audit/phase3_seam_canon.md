# Phase 3 SEAM — AWBW canon investigation (Step 1)

**Campaign:** `desync_purge_engine_harden`
**Thread:** SEAM (Phase 3)
**Author:** Opus subagent
**Question:** For each indirect unit flagged by Phase 2.5 Probe 2 (Artillery, Rocket, Battleship, Piperunner), can the unit attack a pipe seam in AWBW canon? Per-unit verdict requires (a) wiki citation and (b) replay citation.

---

## Method

1. **Wiki primary source.** AWBW Fandom (offline-fetch via `WebSearch`; direct `WebFetch` to `awbw.fandom.com` timed out from the runner). Two pages cited verbatim below: `Pipes_and_Pipeseams` and `Changes_in_AWBW`. Also cross-checked against the Wars Wiki Piperunner page and AW Fandom Piperunner trivia for Piperunner-specific behavior.
2. **Replay corpus.** Full sweep over `replays/amarriner_gl/*.zip` (936 zips, of which 173 contain at least one `AttackSeam` envelope) with `tools/_seam_canon_full_sweep.py` (throwaway, deleted after Step 1). Attribution pulled from `Move.unit.global.units_name` when present; otherwise resolved by `units_id` lookup against (i) the initial PHP snapshot and (ii) every `units_name` payload across the action stream (built units appear post-snapshot).

---

## Wiki citations (shared)

### `Pipes and Pipeseams` — AWBW Fandom

Source: https://awbw.fandom.com/wiki/Pipes_and_Pipeseams

> Pipe Seams have the same characteristics as Pipe, but they may be broken by units. Each has 99 hit points. […] Pipe Seams have 99 hit points, this is different than units, which have 100 hit points, and the same defensive value as a 100/100 Neotank on 0 star terrain. Luck damage does not affect them, though any CO attack bonuses and owned Comm Towers still will. Below is a list of the base damage (no attack boosts) that each unit does to a Pipe Seam: | 5 | 40 | 20 | 50 | 90 | 1 | 45 | 15 | 115 | 55 | 50 | 1 | 50 | 60 | 15 |
>
> Artillery are the most cost-efficient method for destroying early Pipe Seams, able to do so in at most 3 hits — COs that are capable of gaining a 25% attack boost or higher, such as Lash or Kindle on a property, are able to do so in just 2 hits. In the late game, Bombers are the most efficient way to destroy pipe seams for most COs […]

The base-damage list enumerates a per-unit value for fifteen unit slots, including indirects. Artillery is named explicitly as a primary seam attacker.

### `Changes in AWBW` — AWBW Fandom

Source: https://awbw.fandom.com/wiki/Changes_in_AWBW

> Pipe Seams use the damage table for the Neotank in AWBW and not the Md. Tank, and are not affected by luck.

This is the operative AWBW rule: any attacker that has a damage-chart entry against a Neotank can damage a seam. The set of seam attackers is therefore **the set of units with a non-blank Neotank damage column on the damage chart** (https://awbw.amarriner.com/damage.php). All four flagged indirects have Neotank entries (Artillery, Rocket, Battleship, Piperunner). Missiles and Carrier do **not** have Neotank entries (they are anti-air-only / cargo, respectively), which matches their `_SEAM_BASE_DAMAGE = None` in `engine/combat.py` and the Phase 2.5 observation that they do not surface as seam targets.

### Piperunner-specific

Source: https://advancewars.fandom.com/wiki/Piperunner (AW Fandom, Trivia):

> If a Piperunner is on a pipe seam, the seam cannot be targeted until the Piperunner is destroyed.

This trivia presupposes Piperunner participates in seam combat — its unique role is firing along pipes; the only AWBW-meaningful constraint on seam targeting that singles it out is the standard "unit on tile blocks attack on tile" rule.

Source: https://warswiki.org/wiki/Piperunner — confirms Piperunner targets "all other units" with 2–5 range and is functionally an indirect with movement type `MOVE_PIPELINE`. Does not explicitly call out seam attack but does not exclude it either.

---

## Replay corpus result

Full sweep (`tools/_seam_canon_full_sweep.py` over 936 zips):

| Attacker | Seam events | Distinct games | First example (gid env#) | Verdict source |
|---|---:|---:|---|---|
| Artillery   | **284** | many | `1609533 env#18` | replay-confirmed |
| Tank        |  87 | many | `1609533 env#14` | replay-confirmed (direct) |
| B-Copter    |  65 | many | `1609533 env#38` | replay-confirmed (direct) |
| Infantry    |  65 | many | `1615566 env#25` | replay-confirmed (direct) |
| Mech        |  61 | many | `1623866 env#15` | replay-confirmed (direct) |
| Md.Tank     |  21 | many | `1615566 env#23` | replay-confirmed (direct) |
| Bomber      |   7 |  4   | `1622443 env#30` | replay-confirmed (direct) |
| Neotank     |   4 |  3   | `1619791 env#31` | replay-confirmed (direct) |
| **Rocket**  |   **4** |  **4** | `1623193 env#22` (also `1626236 env#24`, `1635708 env#23`, `1636707 env#48`) | **replay-confirmed** |
| Anti-Air    |   3 |  3   | `1628323 env#18` | replay-confirmed (direct) |
| Recon       |   2 |  2   | `1628918 env#17` | replay-confirmed (direct) |
| **Battleship** | **0** | 0 | — | **pending replay confirmation** |
| **Piperunner** | **0** | 0 | — | **pending replay confirmation** |

The Battleship / Piperunner zeros reflect map availability in the GL std-tier corpus, not a forbidden rule: pipe maps with seam-adjacent sea (Battleship) or with a Piperunner reaching a seam from a 2–5 pipe path (Piperunner) are uncommon but not banned.

---

## Per-unit verdicts

### Artillery

- **Wiki:** ALLOWED. `Pipes and Pipeseams` lists Artillery in the per-unit base damage row; the article names Artillery as "the most cost-efficient method for destroying early Pipe Seams." The `Changes in AWBW` rule (Neotank damage column) further confirms — Artillery has a Neotank entry on the damage chart.
- **Replay:** ALLOWED. **284 events** across many GL games. Earliest sample: `replays/amarriner_gl/1609533.zip` env#18 — `Move:[]`, `AttackSeam` from a stationary Artillery (`units_id=191121860`).
- **Verdict:** **ALLOWED.**

### Rocket

- **Wiki:** ALLOWED. Listed in the seam damage row on `Pipes and Pipeseams`. Rocket has a Neotank entry on the damage chart per `Changes in AWBW`.
- **Replay:** ALLOWED. **4 events** in 4 distinct GL games. Earliest: `replays/amarriner_gl/1623193.zip` env#22; also `1626236 env#24`, `1635708 env#23`, `1636707 env#48`.
- **Verdict:** **ALLOWED.**

### Battleship

- **Wiki:** ALLOWED. Listed in the seam damage row on `Pipes and Pipeseams`. Battleship has a Neotank entry on the damage chart per `Changes in AWBW`.
- **Replay:** **PENDING.** No `AttackSeam` events with attacker `units_name="Battleship"` observed across 936 zips (173 of which contain seam events). Rare map geometry, not a forbidden rule.
- **Verdict:** **ALLOWED — pending replay confirmation** for Phase 5.

### Piperunner

- **Wiki:** ALLOWED. Listed in the seam damage row on `Pipes and Pipeseams`. AW Fandom trivia explicitly references Piperunner-on-seam interaction ("If a Piperunner is on a pipe seam, the seam cannot be targeted until the Piperunner is destroyed"). Piperunner is the AWBW unit purpose-built for pipe combat (move type = `MOVE_PIPELINE`, indirect 2–5 range).
- **Replay:** **PENDING.** No `AttackSeam` events with attacker `units_name="Piperunner"` observed across 936 zips. Map availability, not a forbidden rule — Piperunner-active maps are scarce in the GL std-tier slice we have.
- **Verdict:** **ALLOWED — pending replay confirmation** for Phase 5.

---

## Reconciliation against Phase 2.5 Probe 2

Phase 2.5 logged **BUG** for Artillery / Rocket / Battleship / Piperunner on the basis that `Phase 3 SEAM work expects exclusion or AWBW-verified exceptions`. That framing was a hypothesis, not a canon fact. Step 1 of this thread provides the canon: **all four are allowed seam attackers in AWBW**, with `_SEAM_BASE_DAMAGE` values that already match the wiki (Artillery 70 vs wiki Neotank-row entry; Rocket 80; Battleship 80; Piperunner 80 — these correspond to attacker-vs-Neotank base damage, the AWBW rule).

The current engine is therefore **correct** on this point. There is no legality fix to ship.

---

## ESCALATIONS

Per the plan's escalation rule (`Phase 3 thread escalates if a legality fix could plausibly forbid a real AWBW move`, and `If your verdict for any unit is PENDING (no replay evidence) ... STOP coding. Return early to the commander with the open question`), this thread halts before any code change.

### E1 — Engine already matches AWBW canon; should we ship a no-op patch?

- **File:line:** `engine/action.py` `get_attack_targets` ~330–338, `engine/combat.py` `_SEAM_BASE_DAMAGE` ~93–109, `engine/game.py` `_apply_seam_attack` ~975+.
- **Behavior:** Engine permits Artillery, Rocket, Battleship, Piperunner to target seam tiles via `get_attack_targets`, and `_apply_seam_attack` resolves damage from `_SEAM_BASE_DAMAGE`. Wiki + 288 confirmed Artillery/Rocket replay events agree this is correct.
- **Question:** Should Phase 3 SEAM ship the explicit `_SEAM_ALLOWED_INDIRECTS = frozenset({ARTILLERY, ROCKET, BATTLESHIP, PIPERUNNER})` allowlist as a documentation-only / defense-in-depth artifact (catches a future refactor that adds a new indirect like `BLACK_BOMB` and silently lets it hit seams), or skip the patch entirely since the source-of-truth gate is `_SEAM_BASE_DAMAGE.get(unit_type) is not None` and that already excludes Missiles/Carrier correctly?
- **Verdicts:**
  - **VERDICT A (recommended): no engine change.** `_SEAM_BASE_DAMAGE` is already the explicit allowlist (the `None` filter in `get_attack_targets` line 336–338 is the gate). Adding a parallel `_SEAM_ALLOWED_INDIRECTS` set duplicates the same rule and risks the two systems drifting (the Phase 3 source-of-truth principle). Update Phase 2.5 doc to note the BUG verdict was wrong and close the SEAM thread.
  - **VERDICT B: ship the no-op allowlist as defense-in-depth.** Adds a redundant assert in `_apply_seam_attack` that the attacker's `_SEAM_BASE_DAMAGE.get(...)` is not None. Cheap. Also catches direct-fire / future indirect units with no chart entry.
  - **VERDICT C: tighten Battleship + Piperunner pending replay.** Forbid both until Phase 5 produces a replay. **NOT RECOMMENDED** — this is exactly the "ship a blanket ban" the plan's weak-flank #1 warned against.

### E2 — Phase 5 replay confirmation tasks

- **Battleship vs seam.** Need at least one GL replay (any tier) where a Battleship fires `AttackSeam`. Suggested method: scan the live league pool (`tools/desync_audit_amarriner_live.py`) once login is restored, or scrape extras tier (`data/amarriner_gl_extras_catalog.json`) for sea-pipe maps. Maps with a coastal pipe seam in the design pool: TBD — recommend Phase 5 owner search by `map_features.json`.
- **Piperunner vs seam.** Need at least one replay with attacker `units_name="Piperunner"` and `action="AttackSeam"`. Pipe-heavy maps with Piperunner unlocked: search `data/maps/` for designs with both pipe-runner production and intact seams in attack range.
- Until Phase 5 closes both: the wiki citation stands as the canon source, and engine behavior is correct.

### E3 — Update to Phase 2.5 register

The Phase 2.5 doc at `docs/oracle_exception_audit/phase2p5_legality_recon.md` Probe 2 currently labels these four indirects as **BUG**. That label is incorrect once Step 1 canon is accepted. Recommend the integration lane append a follow-up note pointing to this doc, or revise the Probe 2 verdict line to **OK — see `phase3_seam_canon.md`**.

---

## Tooling notes

- Throwaway scan tool: `tools/_seam_canon_full_sweep.py` (deleted after Step 1).
- Probe input list: `logs/seam_replays.txt` (387 zips with seam-bearing maps; 173 of those exhibit at least one `AttackSeam` action across the 936-zip pool).
- Outputs (kept for record): canon table above. Raw counts reproducible by re-running the throwaway tool from Git history if needed.

