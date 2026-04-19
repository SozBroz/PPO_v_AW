---
name: AWBW engine parity (Black Boat + Pipe seams)
overview: Close two AWBW gameplay gaps in the Python engine — (1) Black Boat Repair with 1 HP, 10% cost, resupply rules, explicit no self-repair, and a dedicated REPAIR action; (2) Pipe seams with 99 HP, seam-attack damage rules, terrain flip to Broken Seam 115/116 when broken, and replay/export alignment — including an explicit check that seams are **targetable** (legal ATTACK targets without a unit on the tile). Also track **replay 170901** viewer issues (crash ~turn 29; errors opening several turns). Both areas currently diverge from AWBW; this plan merges their scope, references, and todos into one execution track.
todos:
  # --- Black Boat ---
  - id: bb-guard-self
    content: In Black Boat repair logic, skip `if adj is boat` (identity) so the boat can never be its own repair target; document adjacency-only already excluded self (defense-in-depth).
  - id: bb-hp-and-cost
    content: Apply +10 internal HP (1 bar) per eligible heal; heal cost from deployment cost (e.g. `max(1, UNIT_STATS[adj].cost // 10)` per AWBW); deduct `GameState.funds[boat.player]` only when HP increases; cap funds at 999_999.
  - id: bb-resupply-split
    content: Always refuel/rearm per wiki when repair resolves; if funds forbid heal or no HP heal needed, still resupply; handle full-HP / chip-damage edge cases with 0–100 HP + display_hp.
  - id: bb-repair-action
    content: Add `ActionType.REPAIR` with `target_pos` = orthogonally adjacent ally; remove mass auto-repair on WAIT for Black Boat; extend `get_legal_actions` ACTION stage.
  - id: bb-step-trace-export
    content: Wire `step` / `full_trace` for REPAIR; update `tools/export_awbw_replay_actions.py` and game_log shapes for viewer-compatible replays (see awbw-replay-system skill).
  - id: bb-tests
    content: Tests — no self-repair; 1 HP not 2; broke funds skip HP but resupply; full HP $0 heal still refuels.
  # --- Pipe seams ---
  - id: seam-targetable-check
    content: Explicit verification — pipe seams (113/114) must be **valid attack targets** (indirect and eligible direct fire) while intact; `get_attack_targets` / `get_legal_actions` must include seam coordinates **without** a defender unit on the tile; add regression test (minimal map + e.g. Artillery) that proves ATTACK can target a seam and that `step` resolves seam damage (not early-return on empty defender).
  - id: seam-hp-state
    content: Per-tile seam HP (99 for terrain 113/114) or building state; luck excluded from seam damage per wiki; Neo-on-0★ defense profile; CO/tower bonuses apply.
  - id: seam-attack-seam
    content: Allow targeting empty seam tiles; base damage from AWBW seam column (not unit-vs-unit table only); on cumulative damage >= 99 set terrain 113→115 or 114→116 and clear seam HP.
  - id: seam-legal-fov
    content: Extend `get_attack_targets` / `get_legal_actions` for seams; Piperunner traversal rules on broken vs intact seams; FoW/reveal if engine models fog.
  - id: seam-replay-export
    content: Align export with AWBW AttackSeam-style JSON (`third_party/.../AttackPipeUnitAction.cs` / AttackSeam) if replays must animate seam breaks.
  # --- Replay 170901 (viewer / export diagnostics) ---
  - id: replay-170901-crash-turn29
    content: "Investigate replay **170901**: desktop AWBW Replay Player **crashes** when advancing into **~turn 29** (exact turn unclear — window closes quickly). Reproduce with exported zip/trace; capture exception or native crash if possible; compare action stream vs `_rebuild_and_emit` / `full_trace` around that turn."
  - id: replay-170901-turn-open-errors
    content: "Same replay **170901**: on open, viewer reports **errors loading several turns** (partial playback). Diff `a<game_id>` envelopes for failing days; verify PHP snapshot lines and JSON action payloads; fix exporter or trace so rebuild matches upstream AWBW Replay Player format (reference parsers on GitHub — we do not vendor the C# viewer)."
---

# AWBW engine parity — Black Boat and Pipe seams

Single plan merging former **Black Boat AWBW parity** and **Pipe seam verification** work. Execute when you leave plan mode and order implementation.

## Scope summary

| Area | AWBW expectation | Engine today |
|------|------------------|--------------|
| **Black Boat** | [Repair command](https://awbw.fandom.com/wiki/Black_Boat): 1 HP, 10% target cost (heal skipped if unaffordable), resupply still; one chosen adjacent ally | Repair on every WAIT, +20 HP (2 bars), no funds, all adjacent allies |
| **Pipe seams** | [99 HP](https://awbw.fandom.com/wiki/Pipes_and_Pipeseams), special damage (no luck), break → Broken Seam (Plains-like); Piperunner cannot use broken seam | Terrain IDs exist; `_apply_attack` exits when no unit on target — **cannot attack empty seam** |

## Part A — Black Boat

### Self-repair

[`_black_boat_repair`](c:\Users\phili\AWBW\engine\game.py) only scans orthogonal **neighbors**, so the boat is never `adj` in normal play. Still add **`if adj is boat: continue`** for defense-in-depth.

### Rules to implement

- **1 HP** → internal **`+10`** (not current `+20`).
- **10%** of target’s **listed** deployment cost; align rounding with AWBW.
- **Resupply** fuel/ammo even when heal is skipped or unaffordable (per wiki).
- **`REPAIR`** action with **`target_pos`**, not mass heal on **`WAIT`**.

```mermaid
flowchart TB
  subgraph today [Current]
    W[WAIT]
    W --> BB[Heal all adjacent +20 HP]
  end
  subgraph target [Target]
    R[REPAIR]
    R --> one[One adjacent ally]
    R --> hp[+10 if affordable]
    R --> rs[Resupply]
  end
```

### Files

[`engine/game.py`](c:\Users\phili\AWBW\engine\game.py), [`engine/action.py`](c:\Users\phili\AWBW\engine\action.py), [`tools/export_awbw_replay_actions.py`](c:\Users\phili\AWBW\tools\export_awbw_replay_actions.py), new/extended `test_*.py`.

---

## Part B — Pipe seams

### AWBW reference

- **99 HP** per seam; defense as **100/100 Neotank on 0★**; **no luck**; **CO + Comm Tower** bonuses apply.
- Broken tile = **Broken Seam** — **115 / 116** in [`terrain.py`](c:\Users\phili\AWBW\engine\terrain.py) (`HPipe Rubble` / `VPipe Rubble`), Plains-like; **Piperunners** cannot traverse broken seam (per wiki).

### Gap

[`_apply_attack`](c:\Users\phili\AWBW\engine\game.py) returns early when `get_unit_at(target_pos)` is **None**, so **empty seams cannot be struck**. No seam HP state; [`damage_table.json`](c:\Users\phili\AWBW\data\damage_table.json) has no seam column.

### Implementation direction

- Store **remaining seam HP** (or damage tally) for tiles 113/114.
- **Targetability (acceptance check):** Seams must be **selectable** like unit targets — the legal-action pipeline must not require `get_unit_at(target)` for seam tiles; today [`_apply_attack`](c:\Users\phili\AWBW\engine\game.py) bails when `defender is None` — fixing that is part of **seam-targetable-check** + **seam-attack-seam**.
- **Attack seam** path (or extend attack): apply seam damage formula, then if **≥ 99** damage accumulated, rewrite **`map_data.terrain[r][c]`** to **115** or **116** (preserve H vs V).
- Encode or import **per-unit-type base damage vs seam** from [wiki table](https://awbw.fandom.com/wiki/Pipes_and_Pipeseams).

### Files

Same core trio as Part A (`game.py`, `action.py`, export tools), plus [`engine/map_loader.py`](c:\Users\phili\AWBW\engine\map_loader.py) / `MapData` if seam state lives on the map, [`data/`](c:\Users\phili\AWBW\data) for seam damage data if not hardcoded.

---

## Part C — Replay **170901** (bugs)

Symptoms reported:

1. **Crash near turn ~29** — Advancing the replay toward **turn 29** (approximate; hard to pin down) causes the viewer to **close immediately** / crash.
2. **Turn load errors on open** — Opening the replay shows **issues opening a few turns** (error state for those turns).

**Constraints:** Treat as **export / trace / action-stream** correctness unless proven viewer-only. Cross-check upstream AWBW Replay Player sources on GitHub; we do not keep a vendored C# tree in-repo.

**Suggested workflow:** Locate `170901` artifact under [`replays/`](c:\Users\phili\AWBW\replays) or regenerate from trace; use [`tools/export_awbw_replay_actions.py`](c:\Users\phili\AWBW\tools\export_awbw_replay_actions.py) rebuild path and [`tools/compare_awbw_replays.py`](c:\Users\phili\AWBW\tools\compare_awbw_replays.py) / [`deep_diff_replays.py`](c:\Users\phili\AWBW\tools\deep_diff_replays.py) if a reference zip exists; bisect which turn’s envelope first fails deserialization or state rebuild.

---

## Shared concerns

- **`full_trace` / replay:** Both features need correct action types and snapshot consistency ([`awbw-replay-system` skill](c:\Users\phili\AWBW\.cursor\skills\awbw-replay-system\SKILL.md)).
- **Execution order:** Either sub-feature can be implemented first; both touch **`game.py`** `step` / attack terminators — coordinate merges to avoid conflicting branches. **Replay 170901** can be debugged in parallel once artifacts are in-tree.

When ready to **execute**, exit plan mode and implement against the **todos** in the YAML frontmatter above.
