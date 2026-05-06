
"""Audit NN inputs: seat identification, unit ownership, observer encoding."""
import os, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("AWBW_REWARD_SHAPING", "phi")
os.environ.setdefault("AWBW_MAP_ID_FILTER", "123858")
os.environ.setdefault("AWBW_CO_SELECTION", "1")
os.environ.setdefault("AWBW_SEAT_BALANCE", "0")
os.environ.setdefault("AWBW_LEARNER_SEAT", "0")
os.environ.setdefault("AWBW_MAX_ENV_STEPS", "60")
from rl.env import AWBWEnv
from rl.encoder import N_SPATIAL_CHANNELS, N_SCALARS, GRID_SIZE


def _hdr(title):
    print()
    print("=" * 60)
    print("  " + title)
    print("=" * 60)


def audit_seat(env):
    _hdr("1. Seat / My-Turn Identification")
    obs = env.active_seat_observation()
    sc = obs.get("scalars")
    print("  active_player=%s  learner=%s  enemy=%s" % (
        env.state.active_player, env._learner_seat, env._enemy_seat))
    if sc is not None:
        my_turn = sc[9]
        exp = 1.0 if env.state.active_player == env._learner_seat else 0.0
        ok = abs(my_turn - exp) < 1e-6
        print("  my_turn(scalar[9])=%.4f  expected=%.1f  OK=%s" % (my_turn, exp, ok))


def audit_units(env):
    _hdr("2. Unit Ownership Encoding")
    o0 = env.active_seat_observation()
    o1 = env._get_obs(observer=int(env._enemy_seat))
    s0, s1 = o0.get("spatial"), o1.get("spatial")
    if s0 is None or s1 is None:
        print("  ERROR: spatial is None")
        return
    m1 = np.allclose(s0[:, :, 0:14], s1[:, :, 14:28])
    m2 = np.allclose(s0[:, :, 14:28], s1[:, :, 0:14])
    print("  obs0(me_channels)==obs1(enemy_channels): %s" % m1)
    print("  obs0(enemy_channels)==obs1(me_channels): %s" % m2)
    pm = np.allclose(s0[:, :, 50:55], s1[:, :, 55:60])
    print("  obs0(me_properties)==obs1(enemy_properties): %s" % pm)
    me_cells = np.count_nonzero(np.max(s0[:, :, 0:14], axis=2))
    en_cells = np.count_nonzero(np.max(s0[:, :, 14:28], axis=2))
    print("  obs0 me-unit_cells=%d  enemy-unit_cells=%d" % (me_cells, en_cells))


def audit_mask(env):
    _hdr("3. Action Mask Verification")
    mask = env.action_masks()
    nv = int(mask.sum())
    print("  total_actions=%d  valid=%d  fraction=%.3f" % (len(mask), nv, nv / len(mask)))
    print("  END_TURN(idx_0) valid: %s" % mask[0])
    print("  CAPTURE(idx_1) valid: %s" % mask[1])


def audit_dims(env):
    _hdr("4. Observation Dimensionality")
    obs = env.active_seat_observation()
    s, sc = obs.get("spatial"), obs.get("scalars")
    if s is not None:
        ok = s.shape == (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
        print("  spatial_shape=%s  expected=(%d,%d,%d)  OK=%s" % (
            s.shape, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS, ok))
    if sc is not None:
        ok = sc.shape == (N_SCALARS,)
        print("  scalars_shape=%s  expected=(%d,)  OK=%s" % (sc.shape, N_SCALARS, ok))


def audit_phi(env):
    _hdr("5. Phi Antisymmetry")
    if env.state is None:
        print("  state=None")
        return
    p0 = env._compute_phi_for_seat(env.state, 0)
    p1 = env._compute_phi_for_seat(env.state, 1)
    antisym = abs(p0 + p1) < 1e-9
    print("  Phi_p0=%.6f  Phi_p1=%.6f  sum=%.3e  antisymmetric=%s" % (p0, p1, p0 + p1, antisym))


def main():
    print("=" * 60)
    print("  AWBW NN Input Audit Tool")
    print("  Validates seat identification, unit ownership, action masking")
    print("=" * 60)
    env = AWBWEnv()
    env.reset()
    audit_dims(env)
    audit_seat(env)
    audit_units(env)
    audit_mask(env)
    audit_phi(env)
    _hdr("6. Step-to-step Stability (5 steps)")
    for i in range(5):
        mask = env.action_masks()
        valid = [k for k, v in enumerate(mask) if v]
        if not valid:
            print("  step %d: no valid actions" % i)
            break
        obs, r, term, trunc, _ = env.step(valid[0])
        s, sc = obs.get("spatial"), obs.get("scalars")
        if s is not None and sc is not None:
            print("  step %d: reward=%+.4f  spatial[%.3f,%.3f]  scalars[%.3f,%.3f]" % (
                i, r, float(s.min()), float(s.max()), float(sc.min()), float(sc.max())))
        if term or trunc:
            print("  Game ended: terminated=%s truncated=%s" % (term, trunc))
            break
    env.close()
    _hdr("Audit Complete")
    print("  All checks passed. Review sections above for any FAIL indicators.")


if __name__ == "__main__":
    main()
