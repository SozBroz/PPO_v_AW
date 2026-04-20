# Desync bug tracker (clustered)

Auto-generated summary of `tools/desync_audit.py` register rows, grouped by **subtype**
(message-shape buckets). Official taxonomy remains `class` in each JSONL row; subtypes are for triage.

- **Source register:** `logs/desync_register_repair_unload_verify.jsonl`
- **Games in register:** 11

## Summary by `class`

| `class` | Count |
|---------|------:|
| `ok` | 9 |
| `oracle_gap` | 2 |

## Subtypes (grouped backlog)

| Subtype | Count | Typical cause | Where to fix |
|---------|------:|---------------|--------------|
| `oracle_fire` | 2 | Fire without full Move path, indirects, attacker resolution. | `tools/oracle_zip_replay.py` Fire |

### `ok` games

- **Count:** 9
- **games_id:** 1623866, 1624307, 1624764, 1625290, 1627323, 1630646, 1631178, 1634268, 1634889

### `oracle_fire` (2 games)

*Example message:* `Fire (no path): no attacker P1 (awbw id 192354923) at (4,18) [oracle_fire: strike_possible_in_engine=0 triage=drift_range_los_or_unmapped_co]`

**games_id:** 1630784, 1630794

## Machine-readable export

Run `python tools/cluster_desync_register.py --register logs/desync_register_repair_unload_verify.jsonl --json logs/desync_clusters.json` to emit `2` keys with sorted `games_id` lists.
