# Phase 3 SEAM — Closeout (VERDICT A)

**Campaign:** `desync_purge_engine_harden`  
**Status:** CLOSED — no engine change shipped.

---

## Verdict

**VERDICT A (approved):** The engine already matches AWBW canon for pipe-seam combat. **No code change was shipped** for SEAM in Phase 3.

**Single source of truth:** `_SEAM_BASE_DAMAGE` in `engine/combat.py` defines which unit types may damage seams; the seam-targeting branch in `get_attack_targets` (`engine/action.py`, ~~approximately lines 335–338~~) gates targets on `_SEAM_BASE_DAMAGE.get(unit_type)` (no parallel allowlist needed — duplicating the rule would risk drift).

**AMENDED IN PHASE 7 (2026-04-20):** Phase 6 removed the old line-335–338 Chebyshev pocket; seam targeting now lives inside the unified **Manhattan** range loop in `get_attack_targets` — approximately **`engine/action.py` lines 301–334**, with the no-defender seam branch at **lines 326–334** (`get_seam_base_damage`, terrain 113–116). Same behavioral contract; cites `logs/desync_regression_log.md` § **2026-04-20 — Phase 6: Manhattan correction (post-Phase-5 critical fix)** and the Phase 6 note on `phase3_seam.md` in that entry.


---

## Evidence (see primary write-up)

Full wiki citations, replay corpus counts, and per-unit verdicts (Artillery, Rocket, Battleship, Piperunner, direct-fire units) are documented in:

- **`docs/oracle_exception_audit/phase3_seam_canon.md`**

That document establishes: wiki + replay evidence support indirect seam attacks where the damage chart / `_SEAM_BASE_DAMAGE` permits; Missiles and Carrier correctly have no seam entry; Artillery and Rocket are replay-confirmed at scale; Battleship and Piperunner are wiki-allowed with **pending** GL replay confirmation due to map availability, not a forbidden rule.

---

## Mech “indirect on seam” is structurally impossible

`Mech` has `is_indirect=False` and `max_range=1` in unit stats. ~~The seam-targeting logic in `engine/action.py` (the Chebyshev-distance-1 loop around lines 335–338)~~

**AMENDED IN PHASE 6 (2026-04-20):** After the Phase 6 Manhattan fix, the seam-targeting loop is **Manhattan-distance-1**. The conclusion stands: Mech (`is_indirect=False`, `max_range=1`) can attack a seam ONLY at one of the four orthogonal neighbours, and only as a direct attack (move + adjacent fire), never as an "indirect" attack from 2+ tiles away. The original commander concern (indirect-Mech-on-seam) remains structurally impossible.

The seam-targeting branch still applies to **direct** adjacent attacks on the seam tile; Mech seam attacks are **direct**, not indirect. There is no separate “indirect seam” code path that incorrectly applies to Mech.

---

## Phase 5 follow-ups (replay confirmation only)

No engine change is pending for SEAM correctness. Optional **evidence** follow-ups when the Global League extras tier or live pool produces suitable maps:

1. **Battleship vs seam:** At least one `AttackSeam` replay with attacker Battleship — e.g. sea-pipe-adjacent map geometry (search GL extras / live pool for seam near sea).
2. **Piperunner vs seam:** At least one `AttackSeam` replay with attacker Piperunner — e.g. pipe-runner-active map with intact seam in 2–5 range along pipes.

Until then, wiki + `_SEAM_BASE_DAMAGE` alignment remains the canon basis; engine behavior is held as correct.

---

## Phase 8 Lane H — Replay confirmation (2026-04-20)

**NOT FOUND** — No `AttackSeam` envelope in the **local** replay corpus had an attacker resolved as **Battleship** or **Piperunner** (same stream as the oracle: `parse_p_envelopes_from_zip` on each zip; attacker typing from JSON `units_name` or `units_id` → name via turn-0 PHP snapshot + action-stream walk, per `phase3_seam_canon.md`).

| Metric | Value |
|--------|------:|
| Pools scanned | **3** (`replays/amarriner_gl`, `replays/amarriner_gl_current`, `replays/*.zip` loose root) |
| Zip files scanned | **955** (936 GL std + 0 `amarriner_gl_current` + 19 loose) |
| `AttackSeam` envelopes (all attacker types) | **692** |
| Battleship hits | **0** |
| Piperunner hits | **0** |

**Top 5 attacker types** among those envelopes (after `units_id` resolution): Artillery **289**, Infantry **98**, Mech **95**, Tank **87**, B-Copter **65**.

**Artifacts:** `tools/_phase8_lane_h_seam_scan.py`, `logs/phase8_lane_h_seam_scan.json`.

**Next steps:** `replays/amarriner_extras/` is not present on disk; the GL **std** catalog (`data/amarriner_gl_std_catalog.json`) was already known empty for these unit types. To close the loop: (1) target maps with **sea adjacent to pipe seams** (Battleship) and **active Piperunner lines of sight to seams** (2–5 range on pipes) — see map hints and unit citations in **`docs/oracle_exception_audit/phase3_seam_canon.md`**; (2) when policy allows downloads, expand the mirror to **GL extras tier** and/or **live-pool** exports (this lane did not fetch). Phase 5 engine verdict remains unchanged: wiki + `_SEAM_BASE_DAMAGE` alignment; no `engine_bug` audit was required because no candidate replay was found.
