# Phase 11 — `state_mismatch_investigate` implementation design

Executable spec for wiring **PHP snapshot ↔ engine `GameState` diff** into `tools/desync_audit.py`, turning silent drift (Phase 10F: **39/50** `ok` games mismatched PHP) into first-class register rows.

**Status:** Design only (no engine/oracle edits in this document).

---

## Section 1 — Snapshot inventory

**Source of truth:** first member of the `.zip` is a gzip blob; decompressed text is one PHP-serialized `awbwGame` line per turn/frame. Parsed with `tools/diff_replay_zips.py::load_replay` → `parse_php` per line. **Envelope stream** is separate: `tools/oracle_zip_replay.py::parse_p_envelopes_from_zip` reads `a<games_id>` gzip for `p:` lines. Pairing of frames to envelopes: `tools/replay_snapshot_compare.py::replay_snapshot_pairing` (**trailing** `N+1` frames / `N` envelopes, or **tight** `N`/`N`).

| Domain | In PHP gzip body (`frame` dict) | Extraction | Engine field(s) | Compare mode |
|--------|----------------------------------|------------|-----------------|--------------|
| **Funds** | `players[*].funds` (int), keyed with `players[*].id` | `compare_funds` pattern: map `id` → seat via `map_snapshot_player_ids_to_engine(frames[0], …)` | `GameState.funds[0|1]` | **EXACT** integer |
| **Unit list** | `units[*]` dict: `x`,`y` (col,row), `players_id`, `name`, `hit_points` (float = internal/10), `carried` (`Y` skips — onboard cargo), optional `id` | Build tile×owner keys excluding carried; match `replay_snapshot_compare.compare_units` | `GameState.units[seat]` alive units: `pos`, `player`, `unit_type`→name, `hp`→`display_hp` vs `ceil(hit_points)` bars | **Position / owner / type:** EXACT (with known **name aliases** only: Md.Tank ↔ Medium Tank, etc.). **HP:** EXACT on **display bars** after canonical decode (`ceil` of PHP float to bar, same as `_php_unit_bars`; internal HP via `round(hit_points*10)` if comparing raw HP — must match one convention everywhere) |
| **Unit fuel / ammo** | PHP unit rows typically include fuel/ammo fields (AWBW DB shape; not yet compared in `compare_snapshot_to_engine`) | Read from each `units` dict value when extending comparator | `Unit.fuel`, `Unit.ammo` | **EXACT** (recommended once parser fields confirmed on GL zips) |
| **Property ownership & capture** | `buildings[*]`: `terrain_id`, coordinates, `players_id`, capture-related fields (PHP schema; mirror C# Replay Player) | Iterate `frame["buildings"]` | `GameState.properties`: `owner`, `row`/`col`, `terrain_id`, `capture_points` (engine: 20 = neutral full / not capturing per `PropertyState` doc) | **EXACT** after defining **capture_points ↔ PHP** mapping (document equivalence; may need one-time validation pass on sample zips) |
| **CO identity** | `players[*].co_id`, `order`, `id` | Snapshot row + catalog cross-check already in `map_snapshot_player_ids_to_engine` | `GameState.co_states[seat].co_id` | **EXACT** (drift here is loader/catalog bug → existing `loader_error` path) |
| **CO power meter / active power** | PHP may expose star charge / power flags on player or CO rows (verify per export; engine has `power_bar`, `cop_active`, `scop_active`) | Field names from sample `awbwGame` dumps or C# deserializer | `COState.power_bar`, `cop_active`, `scop_active` | **EXACT** once field mapping verified; if PHP uses coarse stars only, map to engine bar scale or compare **rounded** tier — **±0 on integer bar** if both are integers; if one side is star count × constant, document conversion |
| **Day / active player** | `frame["day"]`, `frame["turn"]` (active PHP `players[].id`) | Top-level keys on parsed line | `GameState.turn`, `GameState.active_player` | **EXACT** at snapshot boundaries (misalignment often indicates **income timing / pairing** bug — Phase 10N **1628609**); treat mismatch as high-signal **meta** row or include in `diff_summary` |
| **Weather** | PHP weather fields on game object (confirm key names from exports) | Top-level | `GameState.weather`, `default_weather`, `co_weather_segments_remaining` | **EXACT** on discrete enum string; Phase 10N notes engine may ignore weather in income — extended diff can surface this |

**Already implemented in repo:** funds + units (tile, type, HP bars) in `compare_snapshot_to_engine`. **Phase 11 minimum viable:** reuse that + structured `diff_summary`. **Stretch:** buildings, fuel/ammo, CO meter, weather for tighter clusters.

---

## Section 2 — Diff cadence (recommendation)

| Option | When diff runs | Pros | Cons |
|--------|----------------|------|------|
| **A** | After every **action** inside envelopes | Maximum resolution | **5–10×** slower; snapshots do not exist per micro-action — still only meaningful when a **post-envelope** frame exists; mostly redundant with B |
| **B** | After each **`p:` envelope** completes (same as `replay_state_diff.run_zip` today) | Aligns 1:1 with gzip **frame index** `step_i+1`; natural AWBW half-turn grain; matches Phase 10N drill | **~1.5–3×** vs exception-only audit |
| **C** | Only after first divergence | Cheap | **Does not apply** — silent drift has **no** exception; C is the old “first exception” model |

**Recommendation: Option B (per-envelope, post-envelope snapshot).**  
Rationale: PHP lines are authored per **half-turn** boundary, not per JSON action. `replay_state_diff` and `_phase10n_drilldown` already use “after envelope `i` → compare `frames[i+1]` if present”. Option A adds CPU without a matching snapshot line. Optional micro-optimization: **skip diff** when `snap_i >= len(frames)` (tight export last line) — already implicit.

**Refinement (optional later):** “heavy” fields (full unit list) every envelope; **CO meter** only on envelopes whose action tail includes `Power` / `End` — **low priority** once baseline works.

---

## Section 3 — Classification taxonomy

**Precedence (single row per game):**

1. **Catalog / zip layout** → existing `catalog_incomplete`, `replay_no_action_stream`, `loader_error` as today.  
2. **During replay:** first raised exception → `engine_bug` | `oracle_gap` | `loader_error` (unchanged).  
3. **Replay completes without exception** and `--enable-state-mismatch` is on, frames/envelopes **aligned** (`replay_snapshot_pairing` not `None`): run snapshot diff at cadence B; **first** PHP mismatch → **state mismatch** row.  
4. Otherwise → `ok`.

**Primary `class` strings (proposed — all prefixed `state_mismatch_` for stable filtering):**

| `class` | When |
|---------|------|
| `state_mismatch_funds` | Only funds lines would fire (if funds checked first, use when **first** mismatch in compare order is funds-only) |
| `state_mismatch_units` | First mismatch includes unit tile / type / HP bars |
| `state_mismatch_properties` | First mismatch on buildings/capture (when implemented) |
| `state_mismatch_co` | First mismatch on CO meter / power flags (when implemented) |
| `state_mismatch_meta` | `day` / `turn` / `active_player` / weather disagree with PHP before or with unit/funds checks |
| `state_mismatch_multi` | Single step reports **multiple** mismatch families (e.g. funds + HP — common per Phase 10F) |
| `state_mismatch_investigate` | Reserved: comparator failure, unknown layout, or **pairing ambiguity** pending spec (use sparingly) |

**Interaction with existing classes:**

- **`engine_bug`:** Always wins if an exception fires **before** stream end. State diff is **not** consulted for that game in the same run (or optionally run only if harness catches exception — **not recommended**).  
- **`oracle_gap`:** `UnsupportedOracleAction` ends replay early; snapshot diff **skipped** (no “complete oracle path” guarantee).  
- **`ok`:** Only if no exception **and** (flag off **or** snapshot match through all compared frames).

**Implementation note:** `state_mismatch_multi` matches Phase 10F reality (funds + `hp_bars` same step). Triage can still **cluster** by `diff_summary.primary_cause` heuristics (funds-first vs HP-first).

---

## Section 4 — Per-field tolerance

| Field | Tolerance | Rationale (Phase 10F / 10N / `desync_audit.md`) |
|-------|-----------|---------------------------------------------------|
| **Funds** | **EXACT** | 10N: deltas like +200, +630 are **meaningful** economy bugs; any non-zero is actionable |
| **HP (display bars)** | **EXACT** after `ceil` rule | `desync_audit.md` + `replay_snapshot_compare._php_unit_bars`: `round` was **wrong**; remaining bar diffs are **true** drift, not noise |
| **HP (internal)** | **EXACT** if using `round(hit_points*10)` consistently | `oracle_state_sync._php_internal_hp`; ±5 **bars** would hide 50 internal HP — **reject** loose tolerance for audit |
| **Position (tile)** | **EXACT** | 10F **1616284** position-only case; movement bugs must surface |
| **Fuel / ammo** | **EXACT** | No evidence of intentional float; mismatch = supply/dive/hidden semantics bug |
| **Capture progress** | **EXACT** | Capture timer convergence is oracle-sensitive; small drift breaks Capt legality |
| **CO power meter** | **EXACT** on mapped integer | If mapping uncertain, defer sub-field to Phase 11b **after** scalar fields stable; do not use ±1 without evidence — would blur COP timing bugs |
| **Weather / day / active player** | **EXACT** | 10N **1628609**: boundary/meta skew is itself a **root cause** signal |

**Luck / RNG:** `desync_audit.md` notes luck noise in **unsynced** HP — without per-attack overrides, some HP drift is expected **in theory**, but Phase 10F sample shows **large** structural gaps; **do not** widen tolerance to “absorb” those — use `oracle_state_sync` only in **separate** `--sync` experiments, not in canonical mismatch classification.

---

## Section 5 — Output format (JSONL schema)

Extend each register row with optional snapshot-diff fields (backward compatible: consumers ignore unknown keys).

**Required when `class` starts with `state_mismatch_`:**

```json
{
  "games_id": 1628546,
  "map_id": 180298,
  "tier": "T1",
  "co_p0_id": 23,
  "co_p1_id": 7,
  "matchup": "Kindle vs Max",
  "zip_path": "replays/amarriner_gl/1628546.zip",
  "status": "snapshot_divergence",
  "class": "state_mismatch_multi",
  "exception_type": "",
  "message": "Funds P0: engine 9000 vs PHP 8800; hp_bars at (0,6,5); envelope 11 / day ~6",
  "approx_day": 6,
  "approx_action_kind": "End",
  "approx_envelope_index": 11,
  "envelopes_total": 120,
  "envelopes_applied": 120,
  "actions_applied": 842,
  "state_mismatch": {
    "first_mismatch_envelope": 11,
    "first_mismatch_frame_index": 12,
    "first_mismatch_day_php": 6,
    "pairing": "trailing",
    "diff_summary": {
      "axes": ["funds", "units_hp"],
      "funds_delta_by_seat": {"0": 200, "1": 0},
      "funds_engine_by_seat": {"0": 9000, "1": 10000},
      "funds_php_by_seat": {"0": 8800, "1": 10000},
      "unit_mismatch_count": 1,
      "human_readable": [
        "P0 funds engine=9000 php_snapshot=8800 (awbw_players_id=…)",
        "at (0,6,5) hp_bars engine=7 (hp=70) php_bars=6 php_id=…"
      ]
    }
  }
}
```

**Notes:**

- Keep top-level `approx_*` aligned with **last touched envelope** at mismatch (mirror current audit semantics).  
- `status`: propose `snapshot_divergence` vs `ok` / `first_divergence` for clarity in dashboards.  
- `human_readable` can be the first **K** lines from `compare_snapshot_to_engine` for drop-in triage.

---

## Section 6 — Performance budget

- **Baseline:** ~741-game audit ≈ **30–45 min** (user-reported).  
- **+ per-envelope diff (Option B):** **~1.5–3×** wall time — one `compare_snapshot_to_engine` per envelope (~O(units)) plus already-paid `apply_oracle_action_json`.  
- **Per-action diff (Option A):** **~5–10×** — unjustified without per-action PHP frames.

**Caching / shortcuts:**

- Reuse **already-loaded** `frames = load_replay(zip_path)` in `_audit_one` (today: load for mapping only; instrumented replay does not hold frames — **pass `frames` or `frame[snap_i]` into hook** to avoid double gzip).  
- **Early exit:** stop at **first** mismatch (same as `replay_state_diff` default).  
- **Skip state diff** when `replay_snapshot_pairing` is `None` → single `state_mismatch_investigate` or `loader_error`-family row with message `unsupported snapshot layout` (existing pattern in `replay_state_diff`).  
- **Optional:** `--state-mismatch-sample-rate` for CI (not in initial spec — prefer default OFF flag).

**Recommendation:** Ship **Option B** + **single load_replay per game** + **default OFF** `--enable-state-mismatch` for PR CI; **ON** for nightly / canonical desync campaign runs.

---

## Section 7 — Numbered implementation plan

1. **Extract snapshot parser to a reusable module** (e.g. `tools/php_awbw_snapshot.py`): `_Reader`, `parse_php`, `load_replay` from `diff_replay_zips.py`; re-export from `diff_replay_zips` for backward compat. **Files:** new ~180–220 LOC module; `diff_replay_zips.py` thin wrapper ~40 LOC.  
2. **Add `_diff_engine_vs_snapshot(state, php_frame, awbw_to_engine) -> DiffResult`** — structured dict/dataclass with axes + numeric deltas; internally call `compare_snapshot_to_engine` or refactor it to return structured output. **Files:** `replay_snapshot_compare.py` +50–120 LOC; or new `tools/state_mismatch_diff.py` ~150 LOC.  
3. **Hook cadence in `_run_replay_instrumented`** (or wrapper used by `_audit_one`): after each envelope loop iteration, if flag on and `snap_i < len(frames)`, call diff; on mismatch return a **sentinel** or raise `StateMismatchError` carrying metadata (cleaner than abusing `Exception` typing — use small custom exception class caught in `_audit_one`). **Files:** `desync_audit.py` ~80–120 LOC.  
4. **Add CLS constants:** `state_mismatch_funds`, `state_mismatch_units`, `state_mismatch_multi`, `state_mismatch_meta`, `state_mismatch_properties`, `state_mismatch_co`, keep `state_mismatch_investigate` as fallback (~10 lines + classifier).  
5. **Update `_classify` or parallel path in `_audit_one`:** map `StateMismatchError` → appropriate `state_mismatch_*` via heuristic on `diff_summary`. **Files:** `desync_audit.py` ~40–60 LOC.  
6. **`AuditRow.to_json` + dataclass:** add optional `state_mismatch: dict | None`; CLI `--enable-state-mismatch` (default **False**). **Files:** `desync_audit.py` ~30–50 LOC.  
7. **Tests:**  
   - Unit: `DiffResult` from known `php_frame` dict fixtures + minimal `GameState` mock or tiny map (~5–8 tests, ~150–250 LOC).  
   - Integration: one zip known to drift (e.g. **1628546** from Phase 10N) with flag ON → expect `state_mismatch_*` (~80–120 LOC).  
   - Regression: golden `ok` game from 10F clean list stays `ok` when flag OFF.

**Total estimate:** **~850–1150 LOC** touched/added across 4–6 files (excluding doc).

---

## Section 8 — Risk inventory

| Risk | Level | Mitigation |
|------|-------|------------|
| Audit slowdown breaks CI | **Med** | Default **OFF**; document ON for canonical runs only |
| False positives (RNG HP noise) | **Low–Med** | EXACT bar compare post-ceil fix; accept some HP rows until oracle combat override is universal — cluster as “luck-sensitive” if needed |
| Backward compat register format | **Low** | Additive keys; new `class` values only; old consumers ignore `state_mismatch` blob |
| Signal overload (hundreds of rows) | **High** | **Cluster** by `co_p0_id`×`co_p1_id`×`diff_summary.axes`×first mismatch envelope band; prioritize Kindle / income / build-slack (10N taxonomy) |
| Pairing / income boundary false “bugs” | **Med** | 10N **1628609**: may need **meta** row + manual tag before engine fix; document in triage playbook |
| Comparator alias gaps (e.g. Missile vs Missiles) | **Low** | Extend aliases; classify as `state_mismatch_investigate` until fixed |

---

## Section 9 — Expected new bug surface

**Phase 10F:** **39/50** (**78%**) of sampled **`class: ok`** games showed PHP snapshot drift; **3/50** full trailing compare clean; **8/50** resign early (no drift detected in prefix).

**Projection for ~936-game canonical run:**

- Assume **~620–700** games remain **`ok`** under exception-only audit (same order of magnitude as **627** `ok` rows cited in `phase10f_silent_drift_recon.md` on post-Phase-9 register — scale ± with map/oracle churn).  
- **0.78 × ~650 ≈ ~500–520** games may flip to **`state_mismatch_*`** when state diff is enabled and pairing is valid.  
- Games already **`engine_bug` / `oracle_gap`** unchanged (~200–300+ depending on register).  
- **Unsupported pairing** (`n_frames` vs `n_envelopes`): small fraction → `state_mismatch_investigate` or explicit `loader_error` message.

**Likely cluster families (from 10F / 10N / `desync_audit.md`):**

1. **Income / funds:** Kindle **+50% city** missing in `_grant_income`, Colin/Sasha edge cases, Sasha SCOP funds semantics.  
2. **Spend-side / oracle slack:** build or repair **no-op** in engine but zip advances (**1620188** pattern).  
3. **Income cadence vs gzip line:** **1628609**-shaped **boundary** (engine granted income PHP line does not yet reflect).

**Triage effort:** **~500** rows × **15–30 min** each coarse triage → **125–250 hours** if naïve; **~40–80 hours** with clustering + 10N-style drill scripts (automated `diff_summary` + income patch replay).

---

## Section 10 — Verdict

**YELLOW → GREEN (implementation-ready)**

- **GREEN:** Core paths exist (`load_replay`, `parse_p_envelopes_from_zip`, `compare_snapshot_to_engine`, pairing logic, Phase 10N drill proves funds instrumentation). An implementer can ship **cold** with **Option B** cadence, **default-OFF** flag, and **structured diff_summary**.  
- **YELLOW (non-blocking):** One cluster (**snapshot boundary / tight exports**) may mix **spec** questions with **engine** bugs — triage playbook should allow tagging `income_boundary_suspect` inside `diff_summary` without blocking the lever.

**Recommendation:** **Proceed to implementation lane** immediately after this design review; parallel **short** recon: dump PHP `day`/`turn` vs engine at first funds mismatch for **1628609** to lock boundary semantics (optional 1–2 hour spike, not a gate).

---

*Document version:* Phase 11STATE-MISMATCH-DESIGN, 2026-04-21.  
*Primary references:* `tools/desync_audit.py`, `tools/replay_snapshot_compare.py`, `tools/replay_state_diff.py`, `tools/_phase10n_drilldown.py`, `tools/_phase10f_recon.py`, `docs/oracle_exception_audit/phase10f_silent_drift_recon.md`, `docs/oracle_exception_audit/phase10n_funds_drift_recon.md`, `docs/desync_audit.md`.
