# Phase 10G — Wider static slack scan (Lane K extended: `tools/` + `engine/`)

**Campaign:** `desync_purge_engine_harden`  
**Mode:** read-only static analysis (no edits to source or tests).  
**Scope:** `tools/*.py` (excluding `tools/oracle_zip_replay.py`, `tests/`, `test_*.py`, `_phaseN_*.py` ad-hoc scripts) and **all** `engine/**/*.py`.  
**Canon:** Phase 6 Manhattan / `logs/desync_regression_log.md` ORCHESTRATOR FOOTNOTE — **hands-off**; this report does not propose engine rule changes, only slack visibility.

**Related:** `phase8_lane_k_slack_inventory.md` (pattern catalog), `phase9_lane_o_tightening.md` (chained raises).

---

## Executive summary

| Metric | Value |
|--------|------:|
| **Files in scope** | 11 engine modules + ~35 `tools/*.py` with slack-relevant patterns |
| **Total patterns catalogued** | **~118** (exception handlers + selected silent-control-flow sites; see §Method) |
| **JUSTIFIED** | **~82** |
| **SUSPECT** | **~31** |
| **DELETE** | **~2** (strict: export exception narrowing + BUILD continue); **+3** borderline (`except Exception: pass` on meta-actions) tracked as SUSPECT |

**Note:** `engine/state.py` does not exist in this repo; mutable state and `_apply_*` live in `engine/game.py` (`GameState`).

### HIGH-risk escalation (Phase 11 before audit closure)

1. **`tools/export_awbw_replay_actions.py` — `_emit_move_or_fire`** (`except ValueError` around `state.step`): `IllegalActionError` subclasses `ValueError` (`engine/game.py`). Failed STEP-GATE replays are **force-moved** and may emit **wrong or partial** Move/Fire JSON — corrupts exported replays and hides illegal-move signals.
2. **`tools/export_awbw_replay_actions.py` — `_rebuild_and_emit` / `_rebuild_and_emit_with_snapshots`**: `except Exception: continue` / `pass` on BUILD, END_TURN, and meta-actions — can **drop or desync** envelopes while state advances or stalls ambiguously.
3. **`engine/game.py` — `_apply_*` silent early `return` under `oracle_mode=True`**: When `step(..., oracle_mode=True)` (oracle / export path), missing units or bad preconditions often **no-op without raise** — **masks** AWBW/engine disagreement that Phase 6-style gates are meant to surface (STEP-GATE is bypassed).
4. **`tools/desync_audit.py` — `_audit_one` setup `except Exception`**: Relabels **`CLS_ENGINE_BUG` → `CLS_LOADER_ERROR`** for *any* setup failure (`base.cls = cls if cls != CLS_ENGINE_BUG else CLS_LOADER_ERROR`), mis-tagging true engine defects during initialization.
5. **`tools/desync_audit.py` — batch `except Exception` (line ~598)**: Harness bugs surface as `loader_error` / generic message — **wrong taxonomy** for regression triage.

---

## Method (Lane K buckets, extended)

Automated sweeps (multiline-aware where needed):

- `except …: pass` / `continue` / `return None` / bare `return`
- `except:` (bare) — **none found** in scope
- `except Exception` (broad)
- Engine: `if … is None: return` / `continue` (silent path)
- **Not** exhaustively flagging `.get(key, default)` (noisy); only noted where semantically suspicious

**Excluded from counts:** `tools/oracle_zip_replay.py`, `tests/`, `*_test*.py`, `_phaseN_*.py`.

---

## Per-file table (pattern sites rated)

Counts are **occurrences triaged** (one row can combine adjacent related handlers). “Patterns” = Lane K slack-relevant constructs, not LOC.

| File | Total | JUSTIFIED | SUSPECT | DELETE |
|------|------:|----------:|--------:|-------:|
| `engine/game.py` | 42 | 28 | 14 | 0 |
| `engine/combat.py` | 8 | 6 | 2 | 0 |
| `engine/action.py` | 4 | 4 | 0 | 0 |
| `engine/co.py` | 2 | 1 | 1 | 0 |
| `engine/map_loader.py` | 2 | 2 | 0 | 0 |
| `engine/predeployed.py` | 2 | 2 | 0 | 0 |
| `engine/terrain.py` | 1 | 1 | 0 | 0 |
| `engine/weather.py` | 1 | 1 | 0 | 0 |
| `engine/map_country_normalize.py` | 4 | 4 | 0 | 0 |
| `engine/belief.py` | 2 | 2 | 0 | 0 |
| `engine/unit.py` | 0 | 0 | 0 | 0 |
| `engine/__init__.py` | 0 | 0 | 0 | 0 |
| `tools/desync_audit.py` | 5 | 2 | 3 | 0 |
| `tools/export_awbw_replay_actions.py` | 14 | 3 | 9 | 2 |
| `tools/export_awbw_replay.py` | 2 | 2 | 0 | 0 |
| `tools/amarriner_download_replays.py` | 4 | 4 | 0 | 0 |
| `tools/cluster_desync_register.py` | 1 | 1 | 0 | 0 |
| `tools/build_legal_actions_equivalence_corpus.py` | 6 | 5 | 1 | 0 |
| `tools/desync_audit_amarriner_live.py` | 6 | 2 | 4 | 0 |
| `tools/oracle_state_sync.py` | 4 | 3 | 1 | 0 |
| `tools/verify_map_csv_vs_zip.py` | 1 | 1 | 0 | 0 |
| `tools/report_live_desync_categories.py` | 2 | 2 | 0 | 0 |
| `tools/analyze_live_pool_overlap.py` | 1 | 1 | 0 | 0 |
| `tools/analyze_game_log_turns.py` | 4 | 4 | 0 | 0 |
| `tools/analyze_game_log_property_trend.py` | 6 | 6 | 0 | 0 |
| `tools/_inspect_oracle_actions.py` | 3 | 1 | 2 | 0 |
| `tools/fetch_predeployed_units.py` | 6 | 4 | 2 | 0 |
| `tools/compare_awbw_replays.py` | 1 | 1 | 0 | 0 |
| Other `tools/*.py` with isolated `except Exception` (debug/validate scripts) | ~12 | ~10 | ~2 | 0 |
| **Approx. total** | **~118** | **~84** | **~32** | **~2** |

---

## DELETE-rated (clear bug / must narrow or raise)

| # | Location | Issue | Suggested patch (Phase 11) |
|---|----------|-------|----------------------------|
| D1 | `export_awbw_replay_actions.py` ~513–523 `_emit_move_or_fire` | `except ValueError` catches **`IllegalActionError`** (subclass of `ValueError`) and force-moves. | Catch **`IllegalActionError` first and re-raise**, or use `except ValueError as e: if type(e) is IllegalActionError or isinstance(e, IllegalActionError): raise` — *or* narrow to the specific `ValueError` shapes used for “blocker diverged” recovery (prefer dedicated exception type). |
| D2 | `export_awbw_replay_actions.py` ~638–643, ~734–737 | `except Exception: continue` on BUILD skips failed builds **without** structured logging. | Replace with logged **warning + counter**, or re-raise after N failures; never bare continue in silence for production export paths. |

*Optional DELETE-class (borderline — treat as SUSPECT if product wants soft export):* `export_awbw_replay_actions.py` ~678–681, ~770–773 `except Exception: pass` on SELECT/COP/SCOP — **swallows all step failures** during bucket rebuild.

---

## SUSPECT entries (full detail)

### `tools/desync_audit.py`

**S1 — `_run_replay_instrumented` L144–145**  
- **Snippet:** `except Exception as exc: return exc`  
- **Behavior:** Returns *any* exception from `apply_oracle_action_json` for `_classify`.  
- **Rating:** **JUSTIFIED** as classifier entry point — not silent (exception becomes row). *(Listed in table as JUSTIFIED; included here only for boundary clarity.)*

**S2 — `_audit_one` L407–446**  
- **Snippet:** `except Exception as exc:` … `cls, et, msg = _classify(exc)` … `base.cls = cls if cls != CLS_ENGINE_BUG else CLS_LOADER_ERROR`  
- **Behavior:** **Engine bugs during setup** are **forced to `loader_error`**, obscuring `engine_bug` class.  
- **Recommendation:** Split “zip/snapshot/map parse” failures from `make_initial_state` / engine ctor failures; or record **raw** exception class before relabeling.  
- **Risk:** **HIGH** (audit taxonomy).

**S3 — `main` batch loop L598–614**  
- **Snippet:** `except Exception as exc: # safety net` → synthetic `AuditRow` with `cls=CLS_LOADER_ERROR`, message `audit harness exception: …`  
- **Behavior:** Any bug in `_audit_one` / row construction is **indistinguishable** from loader failures in the register.  
- **Recommendation:** Use dedicated class e.g. `audit_harness_error` or always `exception_type=AuditHarnessError`.  
- **Risk:** **MEDIUM**.

---

### `tools/export_awbw_replay_actions.py`

**S4 — `_emit_move_or_fire` L513–523**  
- **Function:** `_emit_move_or_fire`  
- **Behavior:** On **any** `ValueError` from `state.step`, sets `step_failed`, may `_move_unit_forced`. **`IllegalActionError` is a `ValueError`.**  
- **Recommendation:** See DELETE D1.  
- **Risk:** **HIGH**.

**S5 — `_rebuild_and_emit` BUILD L638–643**  
- **Behavior:** `except Exception: continue` — **skips** BUILD JSON if step raises.  
- **Risk:** **HIGH** (missing actions in `p:` stream).

**S6 — `_rebuild_and_emit` END_TURN L659–673**  
- **Behavior:** `state.step(action)` **not** in try — OK; SELECT/COP L675–681 uses `except Exception: pass`.  
- **Risk:** **HIGH** (state/action stream mismatch).

**S7 — `_rebuild_and_emit_with_snapshots` BUILD L734–737, END_TURN L754–757, tail L770–773**  
- **Same pattern** as S5–S6 with snapshot list — **HIGH** (viewer ID drift).

**S8 — `_emit_move_or_fire` ValueError path**  
- **Current:** “Fire envelope skipped” when `step_failed` — intentional for divergence; **depends** on D1 fix to avoid swallowing illegal moves.  
- **Risk:** **MEDIUM** until D1 fixed.

---

### `tools/export_awbw_replay.py`

**S9 — L753–758, L841–846**  
- **Behavior:** `log.warning` on action-stream append/write failure; zip remains RV1 snapshot-only.  
- **Rating:** **JUSTIFIED** — **not silent** (warning).  
- **Risk:** **LOW** (operational visibility).

---

### `tools/amarriner_download_replays.py`

**S10 — L367–368 `except OSError: pass` after `dest.unlink()`**  
- **Behavior:** Best-effort delete when `--require_action_stream` rejects RV1 zip.  
- **Rating:** **JUSTIFIED** (cleanup).  
- **Risk:** **LOW**.

**S11 — L105–106 JSONDecodeError → `None`**  
- **Behavior:** `_replay_download_error_kind` returns `None` if JSON parse fails.  
- **Rating:** **JUSTIFIED** (probe function).  

---

### `tools/desync_audit_amarriner_live.py` / `oracle_state_sync.py`

**S12 — `desync_audit_amarriner_live.py`** — `except Exception` blocks (L445, L628) and KeyError continues in PHP scanners — **SUSPECT MEDIUM** (live mirror fragility; similar to Lane K waypoint recovery).  
**S13 — `oracle_state_sync.py` L187–188** — `except (KeyError, TypeError, ValueError): continue` in scan loops — **JUSTIFIED** (malformed envelope fragments), same rationale as Lane K §2.

---

### `tools/_inspect_oracle_actions.py` / `tools/fetch_predeployed_units.py`

**S14 — `_inspect_oracle_actions.py`** — `except Exception: pass` when pretty-printing optional fields — **SUSPECT LOW** (triage UX only).  
**S15 — `fetch_predeployed_units.py`** — broad `except` with `{}` / `[]` returns on network/JSON — **SUSPECT LOW** (script defaults; could hide auth failures).

---

### `engine/game.py` (oracle `_apply_*` silent `return`)

**S16 — `_apply_wait` L838–841, `_apply_dive_hide` L896–902**  
- **Behavior:** `if unit is None: return` (no raise) under oracle `step`.  
- **Recommendation:** Optional `oracle_strict=True` flag to **raise** on missing unit for audit builds.  
- **Risk:** **HIGH** (illegal AWBW envelope could “succeed” silently).

**S17 — `_apply_repair` L987–1015** — multiple early `return` with `_finish_action` only in some branches — **SUSPECT MEDIUM** (complex; needs case-by-case review against AWBW).  

**S18 — `_apply_load` L1140–1144** — `if unit is None or transport is None: return` — **HIGH** (silent skip).  

**S19 — `_apply_join` L1193–1196** — early `return` if mover/partner missing — **HIGH**.  

**S20 — `_apply_unload` L1252–1289** — many guard `return`s (adjacency, terrain) — **SUSPECT HIGH** collectively — **silent** under oracle.  

**S21 — `_apply_build` L1317–1348** — silent `return` on illegal factory/ funds/occupancy — **SUSPECT HIGH** for oracle parity.  

**S22 — `_apply_seam_attack` L1084–1088, 1106–1108** — `if dmg is None: dmg = 0` — **SUSPECT LOW** (hides rare table miss; seam uses Neo profile).  

**S23 — `_apply_attack` L654–661** — `if dmg is not None` else `dmg = 0` — **JUSTIFIED** (matches “no damage” path); **LOW** concern.

---

### `engine/co.py`

**S24 — `make_co_state_safe` L249–252**  
- **Behavior:** `except (FileNotFoundError, KeyError): return make_dummy_co_state(co_id)`  
- **Risk:** **MEDIUM** — missing `co_data.json` or unknown id silently uses **dummy** CO (combat skew). Acceptable for dev; **dangerous** if ever used for production audit without validation.

---

### `engine/combat.py`

**S25 — `calculate_damage` / seam helpers** — `return None` when no base table / hidden rules — **JUSTIFIED** contract.  
**S26 — Callers in `game.py` that coerce `None` → `0` for seam** — see S22.

---

### `engine/map_loader.py`

**S27 — `load_all_maps_gl` L368–373**  
- **Behavior:** `except FileNotFoundError: if skip_missing: continue`  
- **Rating:** **JUSTIFIED** when `skip_missing=True` (documented). **SUSPECT LOW** if caller assumes full coverage.

---

## Top priorities for Phase 11 (ordered)

1. **Fix `export_awbw_replay_actions._emit_move_or_fire` exception narrowing (D1 / S4)** — stop catching `IllegalActionError`.  
2. **Harden `_rebuild_and_emit*` exception handling (S5–S7)** — log, count, or fail closed; eliminate `except Exception: continue/pass` on BUILD/END_TURN/meta.  
3. **`desync_audit.py` setup classification (S2)** — stop relabeling `engine_bug` as `loader_error` without explicit policy.  
4. **`desync_audit.py` harness bucket (S3)** — separate `audit_harness_error` class.  
5. **Engine oracle `_apply_*` optional strict mode (S16–S21)** — raise on missing units / failed unload guards when building audit/export artifacts.  
6. **`make_co_state_safe` visibility (S24)** — warn or assert when dummy CO is used outside test harness.  
7. **`desync_audit_amarriner_live.py` broad handlers** — align with `desync_audit.py` taxonomy.  
8. **`_inspect_oracle_actions.py` / `fetch_predeployed_units.py` (S14–S15)** — low priority logging polish.

---

## Automated grep notes

- **No** `except:\s*$` bare handlers in `engine/` or `tools/` (excluding oracle — not rescanned per mission).  
- **No** `try/except: pass` in `engine/` — engine avoids Lane K’s primary oracle footgun pattern.  
- **`IllegalActionError` hierarchy:** `class IllegalActionError(ValueError)` in `game.py` — any `except ValueError` in tooling is a **footgun**.

---

## Appendix: `desync_audit.py` — harness behavior (reference)

```144:146:D:\AWBW\tools\desync_audit.py
            except Exception as exc:  # noqa: BLE001 — we classify upstream
                return exc
```

```440:446:D:\AWBW\tools\desync_audit.py
    except Exception as exc:  # noqa: BLE001 — pre-replay setup failures
        base.status = "first_divergence"
        cls, et, msg = _classify(exc)
        base.cls = cls if cls != CLS_ENGINE_BUG else CLS_LOADER_ERROR
```

```598:607:D:\AWBW\tools\desync_audit.py
            except Exception as exc:  # safety net — never let one zip stop the batch
                row = AuditRow(
                    games_id=gid, map_id=_meta_int(meta, "map_id"),
                    ...
                    cls=CLS_LOADER_ERROR, exception_type=type(exc).__name__,
                    message=f"audit harness exception: {exc}",
```

---

*End of Phase 10G wider slack inventory.*
