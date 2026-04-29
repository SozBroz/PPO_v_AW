# Encoder equivalence baseline (`encoder_equivalence_pre_restart.npz`)

## What this is

A frozen **float32** snapshot of the pre-architecture-restart observation encoder: for each of **8** constructed `GameState` samples, `encode_state(..., belief=None)` is run with **`observer=0` and `observer=1`**, saving:

- `spatial_o0` / `spatial_o1`: shape `(8, 30, 30, N_SPATIAL_CHANNELS)` (77 after unit modifier planes)
- `scalars_o0` / `scalars_o1`: shape `(8, 16)` (tier scalar removed)

`tests/test_encoder_equivalence.py` requires **byte-identical** outputs against this file so any encoder edit can be classified as **checkpoint-shape-safe** (unchanged float tensors → old PPO weights still load) vs **requires weight restart** (divergence).

Operational context: `MASTERPLAN.md` §12.2, bundle plan
`.cursor/plans/superhuman_restart_architecture_bundle.plan.md` (todo `encoder-equivalence-harness`).

## If the test fails

A failure means the encoder’s numeric output changed. That usually **invalidates existing checkpoints** for the policy head tied to the observation tensor.

- **Do not** silently regenerate the baseline. Confirm with the project lead and treat regeneration as a deliberate, reviewed action.
- To regenerate the file after an intentional encoder change: run once with  
  `AWBW_REGEN_ENCODER_BASELINE=1` (see the test’s failure message for the exact env var name).

## Reproducible corpus (8 states)

All RNG used by the harness is fixed (`random` / `numpy` seeded in the test). Each state is built with `make_initial_state` from a small in-test `MapData` plus hand-placed `Unit` lists (via `_allocate_unit_id`).

1. **s0** — 5×5 all plain, no property tiles, no units, default COs, `luck_seed=10001`, T2, opening P0.
2. **s1** — 12×10: mixed terrain (plain, wood, mountain, river, road, H-bridge, reef, shoal, neutral city/base, sea) and 8+ unit types: Infantry, Mech, Tank, Recon, Artillery, B-Copter (land), Battleship, Lander.
3. **s2** — 8×8: neutral **city** with `capture_points=6` and P0 Infantry on the tile.
4. **s3** — 5×5: HP mix — 100, 64, 31 on Infantry, Medium Tank, Rocket.
5. **s4** — turn **8**, **active player 1**, non-zero funds, T3, distinct COs in `make_initial`, Tank + Anti-Air.
6. **s5** — `weather="snow"`, `co_weather_segments_remaining=4`, two Mech (one at 80 HP).
7. **s6** — non-trivial `power_bar` on both COs, Neo Tank + Rocket.
8. **s7** — Orange Star and Blue Moon **city** tiles with `country_to_player` seating so `p0_income_share` is **0.5** (no units).

`belief` is always **`None`**, so the test does not depend on `BeliefState`.

## File

`encoder_equivalence_pre_restart.npz` is written with `numpy.savez_compressed` and includes a `meta_json` array with a short description of the corpus. Typical size is on the order of a few–tens of KB (sparse spatial tensors compress well).
