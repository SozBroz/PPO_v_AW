# Phase 10C — Move-truncate residual sub-shape classification

**Source:** `logs/desync_register_post_phase9.jsonl` — `class=oracle_gap` and message contains `engine truncated path`.

**Total rows:** 39

## Per-shape summary

| Sub-shape | Count | Lane L coverage | Representative `games_id` | Root cause (not-covered / partial) |
|---|---:|---|---|---|
| nested-Move + Fire (post-kill duplicate) | 22 | covered | 1619504 | Lane L added duplicate-Fire snap; residual means guard conditions still miss this row. |
| Move + Join | 5 | not_covered | 1607045 | Join needs partner HP merge and correct joinID; tail may be friendly-occupied with different seat/id than Lane L's snap allows. |
| plain Move + Wait | 5 | covered | 1628985 | Lane L targets plain Move + Wait; residual implies tail snap still blocked (stacking, seat drift, or occupant rules). |
| Move + Capt | 2 | not_covered | 1627557 | Capt terminator runs capture-progress state; forced tail snap can desync from partial capture / building owner semantics. |
| Move + Load | 2 | not_covered | 1605367 | Load boards onto transport; reachability + cargo/stack rules differ from plain Wait snap. |
| nested-Move + Fire (combat) | 2 | partial | 1630353 | Lane L mirrors post-ATTACK tail reconcile; residual may be indirect range, stance, or Bucket-A edge. |
| nested-Move + AttackSeam | 1 | not_covered | 1634072 | Uses _after_attack_seam / seam terminator, not plain _apply_attack tail. |

## Coverage status (aggregate)

| Status | Meaning | Shapes |
|---|---|---|
| covered | See mission table | nested-Move + Fire (post-kill duplicate), plain Move + Wait |
| partial | See mission table | nested-Move + Fire (combat) |
| not_covered | See mission table | Move + Capt, Move + Join, Move + Load, nested-Move + AttackSeam |
| unknown | See mission table | — |

## Per-row table

| games_id | env_idx | approx_action_kind | sub-shape | Lane L coverage | VAL window |
|---:|---:|---|---|---|---|
| 1605367 | 32 | Load | Move + Load | not_covered | no |
| 1607045 | 46 | Join | Move + Join | not_covered | yes |
| 1619504 | 29 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1620585 | 36 | Join | Move + Join | not_covered | no |
| 1622140 | 27 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1624281 | 26 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1626181 | 24 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1626437 | 25 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1626991 | 26 | Join | Move + Join | not_covered | yes |
| 1627557 | 32 | Capt | Move + Capt | not_covered | yes |
| 1627622 | 19 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1627696 | 22 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1628086 | 26 | Join | Move + Join | not_covered | no |
| 1628357 | 42 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1628722 | 31 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1628985 | 17 | Move | plain Move + Wait | covered | yes |
| 1629512 | 28 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1629722 | 33 | Move | plain Move + Wait | covered | yes |
| 1629757 | 21 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1630353 | 29 | Fire | nested-Move + Fire (combat) | partial | no |
| 1630748 | 15 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1630784 | 32 | Join | Move + Join | not_covered | yes |
| 1630794 | 37 | Load | Move + Load | not_covered | yes |
| 1631257 | 22 | Move | plain Move + Wait | covered | yes |
| 1631389 | 31 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1631767 | 62 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1631858 | 22 | Fire | nested-Move + Fire (combat) | partial | no |
| 1631943 | 26 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1632195 | 26 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1632283 | 29 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1632330 | 26 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1632447 | 29 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1632825 | 20 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1634072 | 22 | AttackSeam | nested-Move + AttackSeam | not_covered | no |
| 1634328 | 20 | Move | plain Move + Wait | covered | yes |
| 1634490 | 24 | Capt | Move + Capt | not_covered | no |
| 1634809 | 22 | Fire | nested-Move + Fire (post-kill duplicate) | covered | yes |
| 1634973 | 24 | Fire | nested-Move + Fire (post-kill duplicate) | covered | no |
| 1635119 | 48 | Move | plain Move + Wait | covered | yes |

## Method

- Failing action resolved via `actions_applied` offset into `approx_envelope_index` (matches `desync_audit` instrumented replay).
- `Fire` split: if merged defender `units_hit_points` ≤ 0 → **post-kill duplicate**; else **combat** (`_oracle_fire_combat_info_merged`).
- Top-level `Move` with only standard keys (`paths`, `unit`, …) → **plain Move + Wait**.
- `parse_failure` rows list error in JSON output.

## Artifact

Machine-readable: `logs/phase10c_classification.json`
