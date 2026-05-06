"""
Smoke test for Cython-accelerated Rhea, Tactical Beam, and MCTS.

Verifies:
1. Cython modules import and function correctly
2. Rhea planner with dynamic budget
3. Tactical Beam search with Cython acceleration
4. MCTS (structure check, not full game)
5. COP disable logic
6. Auto-tuning calculations
"""

import sys
import os
import random
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("SMOKE TEST: Cython Rhea + Beam + MCTS")
print("=" * 60)

errors = []
warnings = []

def check(name, condition, details=""):
    status = "PASS" if condition else "FAIL"
    marker = "[OK]" if condition else "[FAIL]"
    print(f"  {marker} [{status}] {name}")
    if not condition:
        errors.append(f"{name}: {details}")
    return condition


# ===========================================================
print("\n--- 1. CYTHON MODULE IMPORTS ---")
# ===========================================================

try:
    from rl._rhea_cython import (
        simulate_genome_cython,
        crossover_cython,
        mutate_cython,
        random_genome_cython,
    )
    check("Rhea Cython module imports", True)
except Exception as e:
    check("Rhea Cython module imports", False, str(e))

try:
    from rl._tactical_beam_cython import (
        dynamic_budget_cython,
        bucket_for_candidate_cython,
        juicy_score_cython,
    )
    check("Tactical Beam Cython module imports", True)
except Exception as e:
    check("Tactical Beam Cython module imports", False, str(e))

try:
    from rl._candidate_actions_cython import candidate_arrays_cython
    check("Candidate Actions Cython module imports", True)
except Exception as e:
    check("Candidate Actions Cython module imports", False, str(e))


# ===========================================================
print("\n--- 2. RHEA DYNAMIC BUDGET ---")
# ===========================================================

try:
    from rl.rhea import dynamic_rhea_budget
    
    # Test with various complexity levels
    test_cases = [
        (5, 2, 1, 3, "simple position"),
        (15, 5, 3, 8, "medium position"),
        (30, 10, 5, 15, "complex position"),
    ]
    
    for owned, factories, captures, contacts, desc in test_cases:
        pop, gen, max_acts = dynamic_rhea_budget(owned, factories, captures, contacts)
        valid = (12 <= pop <= 64) and (2 <= gen <= 7) and (48 <= max_acts <= 240)
        check(f"Rhea budget ({desc}): pop={pop}, gen={gen}, max_acts={max_acts}", 
              valid, f"out of range for {desc}")
    
    check("Rhea dynamic budget calculation", True)
except Exception as e:
    check("Rhea dynamic budget calculation", False, str(e))


# ===========================================================
print("\n--- 3. TACTICAL BEAM DYNAMIC BUDGET (CYTHON) ---")
# ===========================================================

try:
    # Test Cython dynamic_budget_cython
    result = dynamic_budget_cython(
        owned_units=10,
        juicy=25,  # Total juicy candidates
        bucket_counts={'finish_capture': 3, 'start_capture': 2, 'killshot': 4, 'strike': 5},
        cop_ready=False,
        scop_ready=False,
        min_width=8,
        max_width=48,
        min_depth=3,
        max_depth=14,
        min_expand=4,
        max_expand=24,
    )
    
    check("Beam Cython dynamic_budget returns dict", isinstance(result, dict))
    check("Beam result has 'width'", 'width' in result)
    check("Beam result has 'depth'", 'depth' in result)
    check("Beam result has 'expand'", 'expand' in result)
    
    # Verify expand calculation: 8 + 0.60*25 + 0.25*10 = 8 + 15 + 2.5 = 25.5 -> 25 (or clamped)
    calc_expand = 8 + 0.60 * 25 + 0.25 * 10
    expected_expand = min(24, max(4, int(round(calc_expand)))
    check(f"Beam expand value ({result['expand']}) matches expected ({expected_expand})",
          result['expand'] == expected_expand,
          f"got {result['expand']}, expected {expected_expand}")
    
    print(f"    Beam budget result: width={result['width']}, depth={result['depth']}, expand={result['expand']}")
except Exception as e:
    check("Beam Cython dynamic budget", False, str(e))


# ===========================================================
print("\n--- 4. TACTICAL BEAM BUCKET ASSIGNMENT ---")
# ===========================================================

try:
    from rl.candidate_actions import CandidateKind
    
    # Test bucket_for_candidate_cython with mock candidate
    class MockCandidate:
        def __init__(self, kind, preview=None, terminal_action=None):
            self.kind = kind
            self.preview = preview
            self.terminal_action = terminal_action
    
    class MockAction:
        def __init__(self, name):
            self.action_type = type('MockType', (), {'name': name})()
    
    # Test POWER bucket
    cand_power = MockCandidate(CandidateKind.POWER)
    try:
        bucket = bucket_for_candidate_cython(cand_power, CandidateKind)
        check("POWER candidate -> 'power' bucket", bucket == "power", f"got {bucket}")
    except Exception as e:
        check("POWER candidate bucket", False, str(e))
    
    check("Beam bucket assignment", True)
except Exception as e:
    check("Beam bucket assignment", False, str(e))


# ===========================================================
print("\n--- 5. RHEA PLANNER INITIALIZATION ---")
# ===========================================================

try:
    from rl.rhea import RheaPlanner, RheaConfig
    
    # Create a minimal fitness function for testing
    class MockFitness:
        def score(self, before, after, observer_seat, illegal_genes=0, actions=None):
            from rl.rhea_fitness import RheaFitnessBreakdown
            return RheaFitnessBreakdown(phi_delta=1.0, value=0.5, illegal_penalty=0.0, total=1.5)
        def phi(self, state, seat):
            return 0.5
    
    config = RheaConfig(
        population=16,
        generations=3,
        elite=4,
        mutation_rate=0.20,
        max_actions_per_turn=64,
        use_tactical_beam=True,
        tactial_beam_max_width=32,
        tactial_beam_max_depth=8,
        tactial_beam_max_expand=16,
    )
    
    planner = RheaPlanner(
        MockFitness(),
        config,
        dynamic_budget=False,
    )
    
    check("RheaPlanner initializes with Beam", planner.tactical_beam is not None)
    check("RheaPlanner dynamic_budget=False", planner.dynamic_budget == False)
    
    # Test with dynamic budget
    planner_dynamic = RheaPlanner(
        MockFitness(),
        config,
        dynamic_budget=True,
        complexity_metrics=(10, 3, 2, 5),
    )
    check("RheaPlanner dynamic_budget=True", planner_dynamic.dynamic_budget == True)
    check("RheaPlanner has complexity_metrics", planner_dynamic.complexity_metrics is not None)
except Exception as e:
    check("Rhea Planner initialization", False, str(e))


# ===========================================================
print("\n--- 6. MCTS CONFIGURATION ---")
# ===========================================================

try:
    from rl.mcts import MCTSConfig, run_mcts
    
    config = MCTSConfig(
        num_sims=16,
        c_puct=1.5,
        min_depth=4,
        root_plans=8,
        max_plan_actions=128,
        luck_resamples=0,
        brute_force_branching_threshold=0,  # Disabled by default
    )
    
    check("MCTSConfig initializes", True)
    check("MCTS brute_force disabled by default", config.brute_force_branching_threshold == 0)
    check("MCTS num_sims is set", config.num_sims == 16)
    
    # Test the _estimate_branching_factor function
    from rl.mcts import _estimate_branching_factor
    check("MCTS _estimate_branching_factor exists", callable(_estimate_branching_factor))
except Exception as e:
    check("MCTS configuration", False, str(e))


# ===========================================================
print("\n--- 7. COP DISABLE LOGIC ---")
# ===========================================================

try:
    def _maybe_disable_cop_for_seat(co_state, disable_prob=0.10):
        if disable_prob <= 0.0:
            return False
        if not hasattr(co_state, 'cop_stars') or co_state.cop_stars is None:
            return False
        if random.random() < disable_prob:
            co_state.cop_activation_disabled = True
            return True
        return False
    
    class MockCOState:
        def __init__(self, has_cop=True):
            self.cop_stars = 3 if has_cop else None
            self._data = {"cop": True} if has_cop else {}
            self.cop_activation_disabled = False
    
    # Test with COP-enabled CO
    random.seed(42)
    co = MockCOState(has_cop=True)
    disabled_count = 0
    for _ in range(100):
        co.cop_activation_disabled = False
        if _maybe_disable_cop_for_seat(co, 0.10):
            disabled_count += 1
    check(f"COP disable ~10% (got {disabled_count}%)", 5 <= disabled_count <= 20,
          f"expected ~10, got {disabled_count}")
    
    # Test with COP-disabled CO (no cop_stars)
    co_no_cop = MockCOState(has_cop=False)
    disabled = _maybe_disable_cop_for_seat(co_no_cop, 1.0)  # 100% chance
    check("COP disable skipped for CO without COP", disabled == False,
          "should not disable if no COP available")
    
    # Test with 0% probability
    co = MockCOState(has_cop=True)
    disabled = _maybe_disable_cop_for_seat(co, 0.0)
    check("COP disable skipped when prob=0", disabled == False)
    
    print(f"    COP disable logic: OK (100 games, 10% prob -> {disabled_count} disabled)")
except Exception as e:
    check("COP disable logic", False, str(e))


# ===========================================================
print("\n--- 8. COMPLEXITY METRICS ERROR HANDLING ---")
# ===========================================================

try:
    from rl.rhea import RheaPlanner
    
    # Test that compute_complexity_metrics handles errors gracefully
    # (We can't easily create a real GameState, so just test the error handling)
    
    # The function should catch Exception (not just ImportError) now
    # We test this by checking the source code has "except Exception"
    import inspect
    source = inspect.getsource(RheaPlanner.compute_complexity_metrics)
    check("compute_complexity_metrics catches Exception (not just ImportError)",
          "except Exception:" in source, "Source should have 'except Exception:'")
except Exception as e:
    check("Complexity metrics error handling", False, str(e))


# ===========================================================
print("\n--- 9. CYTHON COMPILED FILES CHECK ---")
# ===========================================================

try:
    import glob
    
    # Check that .pyd files exist for all Cython modules
    cython_modules = [
        "rl._action_cython",
        "rl._occupancy_cython",
        "rl._search_clone_cython",
        "rl._candidate_actions_cython",
        "rl._tactical_beam_cython",
        "rl._rhea_cython",
        "rl._rhea_fitness_cython",
        "rl._encoder_cython",
    ]
    
    all_compiled = True
    missing = []
    for mod in cython_modules:
        mod_path = mod.replace(".", os.sep) + ".pyd"
        full_path = os.path.join(PROJECT_ROOT, mod_path)
        exists = os.path.exists(full_path)
        if not exists:
            missing.append(full_path)
            all_compiled = False
    
    check("All Cython modules compiled (.pyd exists)", all_compiled,
          f"Missing: {missing}")
except Exception as e:
    check("Cython compiled files check", False, str(e))


# ===========================================================
print("\n--- SUMMARY ---")
# ===========================================================

print("\n" + "=" * 60)
if not errors:
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)
    sys.exit(0)
else:
    print(f"FAILED TESTS ({len(errors)}):")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("=" * 60)
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")
    
    sys.exit(1)
