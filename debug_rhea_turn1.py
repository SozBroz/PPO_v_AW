#!/usr/bin/env python3
"""Debug RHEA turn 1 decision making for base skipping issue."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from rl.env import AWBWEnv
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.value_net import load_value_checkpoint

def main():
    # Enable build punishment and phi reward shaping
    os.environ["AWBW_BUILD_PUNISHMENT"] = "1"
    os.environ["AWBW_REWARD_SHAPING"] = "phi"
    
    # Also set learner_seat via environment variable
    os.environ["AWBW_LEARNER_SEAT"] = "0"
    
    # Import here to ensure environment variables are set first
    import json
    from rl.env import AWBWEnv, POOL_PATH
    from rl.rhea import RheaConfig, RheaPlanner
    from rl.rhea_fitness import RheaFitness
    from rl.value_net import load_value_checkpoint
    
    print("=== RHEA TURN 1 DEBUGGING ===")
    print("Setting up environment...")
    
    # Load map pool and filter to specific map ID
    map_id = 171596  # Designed Desires map
    with open(POOL_PATH) as f:
        full_pool = json.load(f)
    
    # Filter to only include the specified map ID
    filtered_pool = [m for m in full_pool if m.get("map_id") == map_id]
    if not filtered_pool:
        raise ValueError(f"Map ID {map_id} not found in map pool")
    
    # Create environment
    env = AWBWEnv(
        map_pool=filtered_pool,
        opponent_policy="random",
        max_env_steps=1000,
        max_p1_microsteps=1000,
        co_p0=14,  # Andy
        co_p1=14,  # Andy
    )
    
    # Reset to get initial state
    obs, info = env.reset()
    
    print(f"\nInitial state created")
    print(f"Turn: {env.state.turn}")
    print(f"Active player: {env.state.active_player}")
    print(f"P0 properties: {env.state.count_properties(0)}")
    print(f"P1 properties: {env.state.count_properties(1)}")
    print(f"P0 funds: {env.state.funds[0]}")
    print(f"P1 funds: {env.state.funds[1]}")
    
    # Check if player has bases
    if hasattr(env, '_player_has_bases'):
        p0_has_bases = env._player_has_bases(env.state, 0)
        print(f"P0 has bases: {p0_has_bases}")
    
    # Check build punishment settings
    print(f"\nBuild punishment settings:")
    print(f"  _build_punishment: {getattr(env, '_build_punishment', 'NOT SET')}")
    if hasattr(env, '_phi_alpha'):
        print(f"  _phi_alpha: {env._phi_alpha}")
        print(f"  Potential punishment: {-6000.0 * env._phi_alpha}")
    
    # Load value network if available
    value_model = None
    try:
        checkpoint_path = project_root / "checkpoints" / "latest_awbw_net_scalpel.pth"
        if checkpoint_path.exists():
            print(f"\nLoading value network from {checkpoint_path}")
            value_model = load_value_checkpoint(checkpoint_path, device="cpu")
        else:
            print(f"\nValue checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load value model: {e}")
    
    # Create RHEA fitness
    fitness = RheaFitness(
        env_template=env,
        value_model=value_model,
        device="cpu",
        reward_weight=0.90,
        value_weight=0.10,
        illegal_gene_penalty=0.02,
    )
    
    # Create RHEA planner
    config = RheaConfig(
        population=16,  # Smaller for debugging
        generations=3,   # Fewer for debugging
        elite=2,
        mutation_rate=0.20,
        max_actions_per_turn=64,
        top_k_per_state=24,
        reward_weight=0.90,
        value_weight=0.10,
        log_initial_best=True,
        seed=42,
    )
    
    planner = RheaPlanner(
        fitness=fitness,
        config=config,
        dynamic_budget=False,
    )
    
    print(f"\nRunning RHEA for turn 1...")
    
    # Run RHEA on current state
    result = planner.choose_full_turn(env.state)
    
    print(f"\n=== RHEA RESULT ===")
    print(f"Score: {result.score:.6f}")
    print(f"Phi delta: {result.breakdown.phi_delta:.6f}")
    print(f"Value delta: {result.breakdown.value:.6f}")
    print(f"Illegal genes: {result.illegal_genes}")
    print(f"Number of actions: {len(result.actions)}")
    
    # Analyze the chosen actions
    print(f"\n=== CHOSEN ACTIONS ANALYSIS ===")
    build_action_taken = False
    end_turn_action = False
    
    for i, action in enumerate(result.actions):
        action_desc = str(action)
        if hasattr(action, 'action_type'):
            action_desc = f"{action.action_type.name}"
            if action.action_type.name == "BUILD":
                build_action_taken = True
                if hasattr(action, 'unit_type'):
                    action_desc = f"{action_desc} - {action.unit_type.name if hasattr(action.unit_type, 'name') else action.unit_type}"
            elif action.action_type.name == "END_TURN":
                end_turn_action = True
        
        print(f"  {i}: {action_desc}")
    
    print(f"\n=== BUILD CHECK ===")
    print(f"Build action taken: {build_action_taken}")
    print(f"End turn action: {end_turn_action}")
    
    if end_turn_action and not build_action_taken:
        print("WARNING: END_TURN without BUILD action!")
        if hasattr(env, '_player_has_bases') and env._player_has_bases(env.state, 0):
            print("  Player has bases - build punishment SHOULD be applied")
            if hasattr(env, '_build_punishment') and env._build_punishment > 0:
                print(f"  Build punishment is enabled: {env._build_punishment}")
            else:
                print(f"  Build punishment is NOT enabled: {getattr(env, '_build_punishment', 'NOT SET')}")
    elif build_action_taken:
        print("BUILD action was taken")
    
    # Now let's check what happens if we simulate ending turn without building
    print(f"\n=== SIMULATING END_TURN WITHOUT BUILD ===")
    
    # Create a copy of the state
    from engine.search_clone import clone_for_search
    test_state = clone_for_search(env.state)
    
    # Apply END_TURN action
    from engine.action import Action, ActionType
    end_action = Action(ActionType.END_TURN)
    
    try:
        test_state.step(end_action)
        print("END_TURN applied successfully")
        
        # Calculate fitness for this scenario
        test_breakdown = fitness.score(
            before=env.state,
            after=test_state,
            observer_seat=0,
            illegal_genes=0,
            actions=[end_action],
        )
        
        print(f"Score for END_TURN without BUILD: {test_breakdown.total:.6f}")
        print(f"  Phi delta: {test_breakdown.phi_delta:.6f}")
        print(f"  Value delta: {test_breakdown.value:.6f}")
        print(f"  Build punishment component: {test_breakdown.total - (0.9 * test_breakdown.phi_delta + 0.1 * test_breakdown.value):.6f}")
        
    except Exception as e:
        print(f"Error simulating END_TURN: {e}")

if __name__ == "__main__":
    main()