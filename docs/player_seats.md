# Engine player seats (red / blue)

In this repo, **engine player index** is the single source of truth for side, turn order, and learning:

| Seat | Index | Typical UI color | Turn order (symmetric start) |
|------|-------|------------------|------------------------------|
| Red / host | **0** | Trained agent, human in `/play/` | Moves **first** on day 1 when both sides have units |
| Blue / guest | **1** | Opponent (bot, self-play pool, etc.) | Moves **second** |

- **Training** ([`rl/env.py`](../rl/env.py)): the policy always chooses actions for **player 0**.
- **Human play** ([`server/play_human.py`](../server/play_human.py)): you are **player 0**; the bot is **player 1**.
- **Observations** ([`rl/encoder.py`](../rl/encoder.py)): “friendly” channels are always **P0’s** view.

Faction art (Orange Star, Black Hole, …) is mapped to 0/1 by [`engine/map_loader.py`](../engine/map_loader.py): default is **row-major scan** of the CSV — first distinct `country_id` on a property → player 0, second → player 1.

## Optional: force which AWBW country is P0 (red)

In [`data/gl_map_pool.json`](../data/gl_map_pool.json), a map entry may set:

```json
"p0_country_id": 1
```

`country_id` matches [`engine/terrain.py`](../engine/terrain.py) (`TerrainInfo.country_id`, e.g. 1 = Orange Star, 5 = Black Hole). That country’s properties and predeployed units are assigned to **player 0**; the other competitive country becomes **player 1**.

**Retrain** after changing `p0_country_id` for a map you train on — old weights assumed the previous seat/faction layout.

### Misery (`map_id` 123858)

This map sets **`"p0_country_id": 1`** so **Orange Star** (HQ bottom-right on the grid — red in the play UI) is **player 0** and moves first on symmetric starts; **Black Hole** (HQ upper-left — purple/black) is **player 1** with the starting infantry from [`data/maps/123858_units.json`](../data/maps/123858_units.json). Training ([`train.py`](../train.py) → [`rl/self_play.py`](../rl/self_play.py) → [`load_map`](../engine/map_loader.py)) and the [human vs bot UI](play_ui.md) both use the same pool entry.
