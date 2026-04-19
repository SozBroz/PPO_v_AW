"""Debug initial game state."""
from rl.env import AWBWEnv
from engine.action import get_legal_actions, ActionType

env = AWBWEnv()
obs, info = env.reset()
state = env.state

print(f"Map: {info['map_id']}, Tier: {info['tier']}")
print(f"\nInitial state:")
print(f"  P0 units: {len(state.units[0])}")
print(f"  P1 units: {len(state.units[1])}")
print(f"  P0 funds: ${state.funds[0]}")
print(f"  Total properties: {len(state.properties)}")

# Count properties by owner
neutral = sum(1 for p in state.properties if p.owner == -1)
p0_owned = sum(1 for p in state.properties if p.owner == 0)
p1_owned = sum(1 for p in state.properties if p.owner == 1)

print(f"  Neutral properties: {neutral}")
print(f"  P0 owned: {p0_owned}")
print(f"  P1 owned: {p1_owned}")

if p0_owned > 0:
    print(f"\nP0 owned properties:")
    from engine.terrain import get_terrain
    for p in state.properties:
        if p.owner == 0:
            terrain = get_terrain(state.map_data.terrain[p.row][p.col])
            print(f"  ({p.row},{p.col}): base={terrain.is_base}, airport={terrain.is_airport}, port={terrain.is_port}")

actions = get_legal_actions(state)
build_actions = [a for a in actions if a.action_type == ActionType.BUILD]
print(f"\nLegal actions: {len(actions)}")
print(f"BUILD actions: {len(build_actions)}")

print("\nDone!")

# Made with Bob
