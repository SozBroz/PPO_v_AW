# Desync bug tracker (clustered)

Auto-generated summary of `tools/desync_audit.py` register rows, grouped by **subtype**
(message-shape buckets). Official taxonomy remains `class` in each JSONL row; subtypes are for triage.

- **Source register:** `logs/desync_register_20260427_d_awbw_main_full.jsonl`
- **Games in register:** 741

## Summary by `class`

| `class` | Count |
|---------|------:|
| `ok` | 741 |

## Subtypes (grouped backlog)

| Subtype | Count | Typical cause | Where to fix |
|---------|------:|---------------|--------------|

### `ok` games

- **Count:** 741
- **games_id (first 40):** 1605367, 1607045, 1609533, 1609589, 1609626, 1610091, 1611364, 1613840, 1614665, 1615143, 1615231, 1615566, 1615789, 1616284, 1617442, 1617897, 1618523, 1618770, 1618984, 1618986, 1619108, 1619117, 1619191, 1619454, 1619474, 1619504, 1619589, 1619695, 1619791, 1619803, 1619807, 1619894, 1620039, 1620188, 1620301, 1620320, 1620450, 1620558, 1620579, 1620585 … *(701 more)*

## Machine-readable export

Run `python tools/cluster_desync_register.py --register logs/desync_register_20260427_d_awbw_main_full.jsonl --json logs/desync_clusters.json` to emit `1` keys with sorted `games_id` lists.
