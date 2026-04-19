# Seat / opening measurement (`game_log.jsonl`)

Rows written by [`rl/env.py`](../rl/env.py) (`AWBWEnv._log_finished_game`) include **`opening_player`** (engine seat `0` or `1` who moves first per `make_initial_state` in [`engine/game.py`](../engine/game.py)) when `log_schema_version` is **≥ 1.5**. Slice win rate and length by opener to quantify a seat/tempo gap before encoder work ([`MASTERPLAN.md`](../MASTERPLAN.md) §9).

```python
import json
import pandas as pd

path = "data/game_log.jsonl"
rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
# Prefer rows that include opening_player (schema ≥ 1.5 from rl/env.py)
if "opening_player" in df.columns:
    df = df[df["opening_player"].notna()]
df["agent_won"] = df["winner"] == 0

g = df.groupby("opening_player")
print(g["agent_won"].agg(["mean", "count"]))
print(g["turns"].mean())
```

If your log mixes legacy rows without `opening_player`, filter on `log_schema_version` or `opening_player.notna()` before grouping.
