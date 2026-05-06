"""
Simple smoke test for Cython Rhea + Beam + MCTS.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("SMOKE TEST: Cython Rhea + Beam + MCTS")
print("=" * 60)

errors = []

# Test 1: Cython modules
print("\n--- 1. CYTHON MODULES ---")
try:
    from rl._rhea_cython import simulate_genome_cython
    print("  [OK] Rhea Cython imports")
except Exception as e:
    print(f"  [FAIL] Rhea Cython: {e}")
    errors.append("Rhea Cython import")

try:
    from rl._tactical_beam_cython import dynamic_budget_cython
    print("  [OK] Tactical Beam Cython imports")
except Exception as e:
    print(f"  [FAIL] Beam Cython: {e}")
    errors.append("Beam Cython import")

try:
    from rl._candidate_actions_cython import candidate_arrays_cython
    print("  [OK] Candidate Actions Cython imports")
except Exception as e:
    print(f"  [FAIL] Candidate Cython: {e}")
    errors.append("Candidate Cython import")

# Test 2: Rhea dynamic budget
print("\n--- 2. RHEA DYNAMIC BUDGET ---")
try:
    from rl.rhea import dynamic_rhea_budget
    pop, gen, max_acts = dynamic_rhea_budget(10, 3, 2, 5)
    print(f"  [OK] Rhea budget: pop={pop}, gen={gen}, max_acts={max_acts}")
    assert 12 <= pop <= 64, f"pop out of range: {pop}"
    assert 2 <= gen <= 7, f"gen out of range: {gen}"
    assert 48 <= max_acts <= 240, f"max_acts out of range: {max_acts}"
except Exception as e:
    print(f"  [FAIL] Rhea budget: {e}")
    errors.append("Rhea dynamic budget")

# Test 3: Beam dynamic budget (Cython)
print("\n--- 3. BEAM DYNAMIC BUDGET ---")
try:
    result = dynamic_budget_cython(
        owned_units=10,
        juicy=25,
        bucket_counts={'finish_capture': 3, 'strike': 5},
        cop_ready=False,
        scop_ready=False,
        min_width=8,
        max_width=48,
        min_depth=3,
        max_depth=14,
        min_expand=4,
        max_expand=24,
    )
    print(f"  [OK] Beam budget: width={result['width']}, depth={result['depth']}, expand={result['expand']}")
    assert 'width' in result
    assert 'depth' in result
    assert 'expand' in result
except Exception as e:
    print(f"  [FAIL] Beam budget: {e}")
    errors.append("Beam dynamic budget")

# Test 4: MCTS Config
print("\n--- 4. MCTS CONFIG ---")
try:
    from rl.mcts import MCTSConfig
    config = MCTSConfig(num_sims=16, c_puct=1.5)
    print(f"  [OK] MCTSConfig: num_sims={config.num_sims}")
    assert config.num_sims == 16
except Exception as e:
    print(f"  [FAIL] MCTS Config: {e}")
    errors.append("MCTS Config")

# Test 5: Rhea Planner with Beam
print("\n--- 5. RHEA + BEAM ---")
try:
    from rl.rhea import RheaPlanner, RheaConfig
    
    class MockFitness:
        def score(self, before, after, observer_seat, illegal_genes=0, actions=None):
            from rl.rhea_fitness import RheaFitnessBreakdown
            return RheaFitnessBreakdown(phi_delta=1.0, value=0.5, illegal_penalty=0.0, total=1.5)
        def phi(self, state, seat):
            return 0.5
    
    config = RheaConfig(
        population=16,
        generations=3,
        use_tactical_beam=True,
        tactial_beam_max_width=32,
    )
    planner = RheaPlanner(MockFitness(), config, dynamic_budget=False)
    print(f"  [OK] RheaPlanner with Beam: beam={planner.tactical_beam is not None}")
    assert planner.tactical_beam is not None
except Exception as e:
    print(f"  [FAIL] Rhea + Beam: {e}")
    errors.append("Rhea Planner with Beam")

# Summary
print("\n" + "="*60)
if not errors:
    print("ALL SMOKE TESTS PASSED!")
    print("="*60)
    sys.exit(0)
else:
    print(f"FAILED TESTS ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")
    print("="*60)
    sys.exit(1)
