# AWBW Fleet & Training — Consolidated Spec

**Purpose:** Single operator reference for how this repo's machines, filesystems, training stacks, and promotion workflows fit together. Synthesized from scattered docs and skills; when this file disagrees with code, **code wins** — file an issue and update this spec.

**Last compiled:** 2026-07-01

**Source of truth for tensor shapes:** `rl/encoder.py`, `rl/network.py`, `rl/value_net.py` (this doc mirrors them; if they drift, code wins).

### Contents

| § | Topic |
|---|--------|
| [1](#1-north-star) | North star (PPO endgame, RHEA bootstrap) |
| [2](#2-awbw--repo-basics) | AWBW & repo basics |
| [3](#3-neural-network-contract) | Neural network contract |
| [4](#4-observation-encoding-what-the-nn-sees) | Observation encoding |
| [5](#5-engine-qa--desync-oracle) | Engine QA & desync oracle |
| [6](#6-host-inventory) | Fleet hosts |
| [7](#7-filesystem-contract) | Filesystem / checkpoints |
| [8](#8-environment-variables) | Environment variables |
| [9](#9-transport-policy-samba-vs-ssh) | Samba vs SSH |
| [10](#10-ppo-fleet-operations) | PPO fleet ops |
| [11](#11-rhea-fleet-operations) | RHEA fleet ops |
| [12](#12-play-ui--behaviour-cloning) | Play UI & BC |
| [13](#13-logs--monitoring) | Logs & monitoring |
| [14](#14-replay-surfaces) | Replay surfaces (pointer) |
| [15](#15-strategic-phases-masterplan-summary) | MASTERPLAN phases |
| [16](#16-known-limitations--open-work) | Limitations |
| [17](#17-quick-reference-commands) | Commands |
| [18](#18-source-index) | Source index |

---

## 1. North star

| Fixed goal | Detail |
|------------|--------|
| **End state** | Superhuman Advance Wars by Web bot trained with **PPO** (`train.py`, MaskablePPO) |
| **Current campaign** | Bootstrap **100k–1M somewhat-competent full games** to warm-start PPO |
| **Active path** | **RHEA** with lookahead + value net → amateur-level self-play today |

**Means, not mission:** RHEA, MCTS, BC, opening books, fleet throughput.

| System | Role | Endgame? |
|--------|------|----------|
| `scripts/train_rhea_value_parallel.py` | Parallel RHEA actors + TD value learner | No |
| `scripts/rhea_remote_actor.py` | Fleet transition collectors | No |
| `train.py` + `rl/self_play.py` | Policy-gradient self-play | **Yes** |
| `scripts/train_bc.py` + `--bc-init` | Imitation warmstart into PPO | Bridge only |

**Corpus gap (standing fact):** Finished RHEA games today leave summary rows in `logs/game_log.jsonl` and turn-level value transitions in the RHEA replay buffer — **not** a durable (state, action) corpus for BC/PPO. Bridge work is open (`RheaStepTransition` in `rl/rhea_replay.py`).

**Sources:** `.cursor/skills/awbw-superhuman-ppo-north-star/SKILL.md`, `MASTERPLAN.md`

---

## 2. AWBW & repo basics

**Advance Wars by Web (AWBW)** is a browser-based, turn-based tactics game. This repo is a **Python reimplementation of the game engine** plus RL training, replay export, and a local play UI. The bot must respect the same legality and combat rules as live AWBW (ranked standard: full vision, standard tier bans).

### What the engine owns

| Module | Role |
|--------|------|
| `engine/game.py` | `GameState`, `step`, income, win/loss, `full_trace` |
| `engine/action.py` | Legal actions, SELECT → MOVE → ACTION stages |
| `engine/combat.py` | Damage, counterattacks |
| `engine/co.py` | CO charge, COP/SCOP |
| `engine/map_loader.py` | Maps from CSV + `data/gl_map_pool.json` |
| `engine/unit.py` | 27 `UnitType`s, stats, `unit_id` (monotonic, never reused) |

### Player seats (engine indices)

| Seat | Index | Typical color | Who uses it |
|------|-------|---------------|-------------|
| Red / host | **P0** | Orange Star (configurable) | Policy learner, human in `/play/` |
| Blue / guest | **P1** | Blue Moon / opponent | Bot, self-play pool, RHEA opponent |

Training historically learns **one policy for P0** with ego-centric observations (`observer=0`). Optional `AWBW_SEAT_BALANCE` generalizes rollout seat. Map entry `p0_country_id` in `data/gl_map_pool.json` forces which faction is P0 — **retrain** after changing it.

Informal “player 1 / player 2” (first vs second human) ≠ engine P0/P1 — see `.cursor/skills/awbw-seat-vocabulary/SKILL.md`.

### Repo layout (high level)

```
engine/     # Pure game logic
rl/         # Encoder, network, env, PPO self-play, RHEA, value net
tools/      # Replay export, desync audit, fleet utilities
scripts/    # Training launchers, eval, BC
server/     # Flask play UI (/play/)
data/       # Map pool, maps/*.csv, opening books
checkpoints/# PPO zips, value_rhea_latest.pt, pool/
```

### Legality & data quality gates

- **Engine ⊂ AWBW:** subset checks in `docs/engine_awbw_parity.md` + `tools/engine_awbw_legality_probe.py`
- **Desync oracle (AWBW zips → engine):** full stack in [§5 Engine QA](#5-engine-qa--desync-oracle) and `docs/desync_audit.md`
- **Regression:** `python -m pytest -q --tb=line` on every ship (unit tests — not full replay corpus)

**Deeper engine/replay reference:** `.cursor/skills/awbw-engine/SKILL.md`, `docs/player_seats.md`

---

## 3. Neural network contract

Two networks share the **same observation tensor** from `encode_state()`:

| Model | File | Output | Used by |
|-------|------|--------|---------|
| **AWBWNet** | `rl/network.py` | Policy logits `(B, 35_000)` + value `(B,)` | RHEA/MCTS advisor, scalpel bridges (**not** live PPO — see desync note below) |
| **AWBWValueNet** | `rl/value_net.py` | Value `(B,)` only | RHEA fitness (`value_rhea_latest.pt`) |

> **⚠ Action-representation desync (known — deliberately deferred, operator decision 2026-07-01):**
> Live PPO does **not** train AWBWNet's factored per-tile heads. `rl/self_play.py` /
> `rl/async_impala.py` build MaskablePPO with `AWBWCandidateFeaturesExtractor` (globally pooled
> 512-d trunk) plus an SB3 linear head over a `Discrete(4096)` **padded candidate-action** space
> (`rl/env.py`, `rl/candidate_actions.py`). Meanwhile BC demos are **flat-35k** rows with MOVE
> skipped (`scripts/train_bc.py`), and opening books / the game corpus use flat engine action
> indices. Three action contracts coexist:
>
> | Contract | Space | Consumers |
> |----------|-------|-----------|
> | Padded candidate rows | `Discrete(4096)` | Live PPO (self_play, async_impala) |
> | Factored flat assembly | 35_000 | AWBWNet (RHEA/MCTS advisor, scalpel) |
> | Flat engine action indices | 35_000 indices, MOVE collapsed | BC demos, opening books, corpus rows |
>
> **Decision:** leave these out of sync until the RHEA bootstrap line is confirmed good enough.
> Then align corpus → BC → PPO on one representation **before** large-scale corpus data feeds PPO.
> Do not "fix" this in passing.

### Architecture (shipped restart stack)

```
Input:
  spatial  (B, 30, 30, 79)   # N_SPATIAL_CHANNELS
  scalars  (B, 20)           # N_SCALARS

Stem:     Conv2d(79 → 128, 3×3) → ReLU
Trunk:    10 × ResBlock128 (depthwise-separable 3×3 + GroupNorm, stride 1)
          → activations stay 30×30×128 (no global pool on policy path)

Scalar fusion:
  Linear(20 → 16) → broadcast to (B, 16, 30, 30) → concat → (B, 144, 30, 30)

Policy head (factored, per-tile 1×1 convs on 144 ch):
  conv_select, conv_move, conv_attack, conv_repair, conv_build (27 types)
  + linear_scalar_policy(144 → 16) for END/COP/SCOP, capture/wait/load/join/dive, unload
  → scatter-assemble flat vector (B, 35_000), mask illegal with -inf

Value head:
  adaptive_avg_pool2d(144 → 1×1) → Linear(144 → 256) → ReLU → Linear(256 → 1)
```

**~6M parameters** (trunk + factored heads; see `docs/restart_arch/compute_budget.md`).

**SB3 path (live PPO):** `AWBWCandidateFeaturesExtractor` in the same file — same stem/trunk/fusion, pooled to `features_dim=512` for the candidate-action MaskablePPO head (see desync note above). The older `AWBWFeaturesExtractor` (256-d) remains as a legacy factory in `rl/ppo.py`.

**Checkpoint migration:** `rl/ckpt_compat.py` expands legacy stems (62/70/77-channel zips) into the current 79-channel layout; optimizer state is dropped when shapes change.

### Action space (flat index)

`ACTION_SPACE_SIZE = 35_000` (`rl/network.py`, must match `rl/env.py`).

| Range | Meaning |
|-------|---------|
| 0–2 | END_TURN, ACTIVATE_COP, ACTIVATE_SCOP |
| 3–902 | SELECT_UNIT (SELECT stage) |
| 903–1800 | ATTACK targets |
| 1818–2717 | MOVE destination (`_MOVE_OFFSET`; MOVE-stage SELECT_UNIT) |
| 3500–3799 | REPAIR tile |
| 10000+ | BUILD (tile × 27 unit types) |
| + unload slots, capture/wait/load/join/dive-hide scalars | See `docs/restart_arch/action_space_inventory.md` |

MOVE-stage actions encode **destination tile** at `1818 + r*30 + c` (not a separate move head).

**Design deep-dive:** `docs/restart_arch/MASTER_SPEC.md`, `spatial_head_spec.md`, `move_encoding_redesign.md`

---

## 4. Observation encoding (what the NN sees)

`encode_state(state, *, observer=0, belief=None)` in `rl/encoder.py` returns:

| Tensor | Shape | dtype |
|--------|-------|-------|
| `spatial` | `(30, 30, 79)` | float32 |
| `scalars` | `(20,)` | float32 |

Maps are padded to **30×30** (`GRID_SIZE`). All “me / enemy” channels are **ego-centric**: `observer` = engine seat of the policy being trained; enemy = `1 - observer`.

### Spatial channels (79 total)

| Ch | Name | Count | Description |
|----|------|------:|-------------|
| 0–13 | `units_me` | 14 | Unit-type one-hot (me) |
| 14–27 | `units_enemy` | 14 | Unit-type one-hot (enemy) |
| 28–29 | `hp_lo`, `hp_hi` | 2 | HP belief interval / 100 (see below) |
| 30–44 | `terrain` | 15 | One-hot terrain category |
| 45–49 | `property_neutral` | 5 | Neutral property types |
| 50–54 | `property_me` | 5 | Owned property types (me) |
| 55–59 | `property_enemy` | 5 | Owned property types (enemy) |
| 60–61 | `capture` | 2 | Capture progress chips (me / enemy) |
| 62 | `neutral_income` | 1 | Contestable neutral income tiles |
| 63–68 | `influence` | 6 | Threat / reach / capture-ETA planes (`engine/threat.py`) |
| 69 | `defense_stars` | 1 | Terrain defense / 4 (map-static cache) |
| 70–71 | `co_tile_attack_bonus` | 2 | CO tile attack bonus (me / enemy) |
| 72–78 | `unit_modifiers` | 7 | Per-occupied-cell move/atk/def/luck/indirect norms |

Channel index map is also `CHANNEL_GROUPS` in `rl/encoder.py` (used by `encode_state_components()` for NN audit UI).

### Scalar features (20 total)

| Idx | Label | Normalization / meaning |
|-----|-------|-------------------------|
| 0 | `funds_me` | / 50_000 |
| 1 | `funds_enemy` | / 50_000 |
| 2 | `power_bar_me` | CO power built / 50_000 |
| 3 | `cop_stars_me` | COP cost in stars (normalized) |
| 4 | `scop_stars_me` | SCOP cost in stars (normalized) |
| 5–6 | `cop_active_me`, `scop_active_me` | 0/1 |
| 7–11 | enemy CO mirrors | same layout |
| 12 | `turn_norm` | `turn / max_turns` |
| 13 | `my_turn` | 1.0 iff `active_player == observer` |
| 14–15 | `co_id_me`, `co_id_enemy` | / 30 |
| 16–18 | `weather_rain`, `weather_snow`, `weather_turns_norm` | binary + segments/2 |
| 19 | `me_income_share` | contestable income tiles owned by observer |

### HP belief

For **runtime bot play**, pass `belief: BeliefState` so enemy HP uses `(hp_min, hp_max)` intervals (`docs/hp_belief.md`). With `belief=None` (debug / legacy), both HP channels show exact `unit.hp/100` for all units — **do not use for production opponents** (leaks exact enemy HP).

### Env observation dict

`AWBWEnv` (`rl/env.py`) exposes `observation_space`:

- `spatial`, `scalars` — as above
- `action_mask` — `(35_000,)` bool legal mask

Reward shaping default: `AWBW_REWARD_SHAPING=phi` (Φ army/property/capture terms in learner frame).

**Encoder regression gate:** `tests/test_encoder_equivalence.py` vs `tests/fixtures/encoder_equivalence_pre_restart.npz` — intentional encoder changes require fixture regen per `tests/fixtures/encoder_equivalence_README.md`.

---

## 5. Engine QA & desync oracle

How we find engine bugs, oracle mapper gaps, and silent state drift vs real AWBW games. **Full detail:** [`docs/desync_audit.md`](desync_audit.md). **Triage runbook:** [`.cursor/skills/desync-triage-viewer/SKILL.md`](../.cursor/skills/desync-triage-viewer/SKILL.md).

### Two validation directions

| Direction | Question | Primary tools |
|-----------|----------|---------------|
| **AWBW → engine** | Can our engine replay every action in a downloaded GL zip? | `desync_audit.py`, `oracle_zip_replay.py`, `replay_state_diff.py` |
| **Engine → AWBW** | Does our engine ever offer moves AWBW would reject? | `engine_awbw_legality_probe.py`, `tests/test_engine_awbw_subset.py` |

There is **no live AWBW API** to query rules. Ground truth is: downloaded replay zips, PHP snapshot bytes in those zips, the **C# AWBW Replay Player**, and AWBW wiki / Amarriner charts.

### Tool catalog

| Tool | What it checks | Output |
|------|----------------|--------|
| **`tools/desync_audit.py`** | Batch-replay GL std zips through `oracle_zip_replay`; stops at **first exception** per game | `logs/desync_register.jsonl` |
| **`tools/oracle_zip_replay.py`** | Maps `p:` JSON (Move/Build/Fire/End/…) → `GameState.step` | Used by audit + diff tools (not usually run alone) |
| **`tools/replay_state_diff.py`** | After each envelope, diff engine vs **PHP `awbwGame` snapshot** in zip (Replay Player contract) | JSONL register; optional `--sync` |
| **`tools/oracle_state_sync.py`** | Snap engine HP/funds/units to PHP between envelopes (luck-drift mitigation) | Called by `replay_state_diff --sync` |
| **`tools/replay_snapshot_compare.py`** | Low-level funds/units/bar compare helpers | Used by `replay_state_diff` |
| **`tools/run_desync_cluster.py`** | Audit + cluster register into subtype buckets | `logs/desync_clusters.json`, `docs/desync_bug_tracker.md` |
| **`tools/cluster_desync_register.py`** | Cluster existing register by `class` / message prefix | JSON or markdown |
| **`tools/debug_desync_failure.py`** | Print engine state at first failure for one `games_id` | stdout |
| **`tools/engine_awbw_legality_probe.py`** | Every `get_legal_actions` output vs AWBW **documented invariants** | Exit ≠0 on violation |
| **`tools/desync_audit_amarriner_live.py`** | Same oracle path for **in-progress** live games (no zip) | `logs/desync_register_amarriner_live.jsonl` |
| **`server/routes/nn_audit.py`** (`/nn-audit/`) | Visualize encoder channels/scalars for a board state | Debug UI only — **not** parity QA |
| **`pytest`** | Unit/regression tests (combat formula, oracle subsets, encoder bytes, …) | CI gate — not a full replay corpus |

**Human comparator:** C# **AWBW Replay Player** (`AWBW_REPLAY_PLAYER_EXE` or `third_party/AWBW-Replay-Player/...`). **Not** the Flask `/replay/` UI (engine JSONL only).

### Primary pipeline (offline GL replays)

```
data/amarriner_gl_std_catalog.json
    → replays/amarriner_gl/{games_id}.zip   (tools/amarriner_download_replays.py)
    → tools/desync_audit.py
    → logs/desync_register.jsonl
    → tools/cluster_desync_register.py / run_desync_cluster.py
    → docs/desync_bug_tracker.md
```

**Smoke (CI / PR):** `python tools/run_desync_cluster.py --pr-smoke` or `--max-games N` — **not** the full ~800-game sweep on every push.

**Snapshot parity (stronger than default audit):**

```powershell
python tools/replay_state_diff.py --games-id 1623065
python tools/replay_state_diff.py --games-id 1610091 --sync   # HP/funds snap mode
python tools/desync_audit.py --games-id 272176 --enable-state-mismatch
```

Default `desync_audit` **`class: ok`** means the oracle stream completed without raising — it does **not** guarantee PHP snapshot agreement. Cross-check with `replay_state_diff` when you care about Replay Player parity.

### Register taxonomy (`desync_register.jsonl`)

| `class` | Meaning |
|---------|---------|
| `ok` | All envelopes applied (includes normal `Resign` endings) |
| `oracle_gap` | Action shape/kind not mapped in `oracle_zip_replay.py` |
| `engine_bug` | Mapped action caused engine exception — investigate rules |
| `loader_error` | Zip/catalog/CO mapping failed before replay |
| `replay_no_action_stream` | ReplayVersion 1 only — no `a<games_id>` `p:` stream |
| `catalog_incomplete` | Missing CO ids in catalog row |

Subtype clustering (`oracle_fire`, `oracle_build`, `oracle_move_no_unit`, …): `tools/cluster_desync_register.py`. Per-game drill-down: `docs/desync_subgroup_debug.md`.

### Engine ⊂ AWBW legality probe

[`docs/engine_awbw_parity.md`](engine_awbw_parity.md) tracks **permissive** bugs (engine allows more than AWBW — dangerous for RL) vs **restrictive** shaping (mask drops actions AWBW allows — usually OK).

```powershell
python tools/engine_awbw_legality_probe.py random --map-id 123858 --turns 60 --seed 0
python tools/engine_awbw_legality_probe.py zip replays/amarriner_gl/1628539.zip --map-id 171596 --co0 1 --co1 2 --tier T2
```

### Triage workflow (one replay at a time)

1. Pick row from `desync_register.jsonl` (or `logs/desync_triage_state.json` cursor — see desync-triage skill).
2. Launch C# Replay Player on the same zip with `--goto-day` / `--goto-envelope`.
3. Fix **oracle** (`tools/oracle_zip_replay.py`), **engine** (`engine/`), or **delete scuffed zip** — mandatory closure per skill.
4. Re-run `desync_audit --games-id <id>` until `ok` or all blockers documented.

### What this QA stack **does** check

- GL **std** map pool games with complete catalog CO ids and a `p:` action stream
- Oracle mapper coverage for mapped action kinds (Move, Build, Fire, End, Capt, Load, Join, Repair, Power, …)
- First engine exception along the recorded action sequence
- Optional per-envelope **funds / units / HP bars** vs PHP snapshots (`replay_state_diff`, `--enable-state-mismatch`)
- Documented **legality invariants** on `get_legal_actions` output (probe + subset tests)
- Combat **formula** vs `data/damage_table.json` at zero-luck baseline (`tests/test_combat_formula_baseline.py`)
- Encoder **byte stability** vs frozen fixture (`tests/test_encoder_equivalence.py`)
- Our **export** zip round-trip for zips we generate (`test_oracle_zip_replay.py`, `WORKING_REPLAY.zip`)

### What it **does not** check (explicit gaps)

**Scope / inputs**

- **Fog of War**, Flares, stealth visibility — engine is full-obs; no fog QA path
- **Non–GL-std** maps in zips (audit skips unless map is in `gl_map_pool.json` std rotation)
- **ReplayVersion 1** snapshot-only zips (no `p:` stream) — audit skips; delete or re-fetch
- **Incomplete catalog rows** (missing `co_p0_id` / `co_p1_id`) — `catalog_incomplete`, not replayed
- **Live AWBW** mid-game state — only via separate `desync_audit_amarriner_live.py` (not default pipeline)
- **Flask `/replay/`** JSONL — training debugger only; not compared to site zips
- **Our exporter → site** for arbitrary training games — only spot-checked via generated zips / oracle round-trip
- **PPO / RHEA policy quality** — separate competency bar; not engine parity
- **NN encoding correctness vs AWBW** — we encode *our* `GameState`; no oracle compares tensors to AWBW

**Combat & luck**

- **Per-attack luck roll** matching AWBW’s GL seed during free replay — `calculate_damage` defaults to `random.randint(0,9)` unless oracle pins `combatInfo` / overrides
- Exhaustive **CO power** parity (Market Crash, War Bonds SCOP, all COP edge timing) — many paths still drift; tracked as state mismatch / oracle_gap clusters
- **Oozium** custom movement (open A2 in parity doc)
- **Black Bomb** detonation targeting, Oozium auto-attack/consume — not fully modelled

**State sync limits (`oracle_state_sync` / `--sync`)**

- Does **not** auto-sync: CO power bars/active flags, fuel, ammo, capture progress, weather, property ownership (by design — orthogonal signals)
- Does **not** spawn PHP-only units (reports `php_only` — avoids masking oracle Load bugs)
- `ok` under `--sync` means no out-of-range HP deltas + no hard abort — cumulative `php_only`/`engine_only` counts may still be nonzero (luck divergence)

**Legality probe limits**

- Does **not** call AWBW — checks **documented invariants**, not exhaustive rules
- Does **not** detect **restrictive** mask gaps (engine ⊂ AWBW safe direction) unless they cause probe false positives
- Does **not** validate **state drift** after legal actions (use desync register / snapshot diff)

**Default `desync_audit` without flags**

- Does **not** compare engine to PHP snapshots each envelope
- Does **not** prove **export** or **human demo** rows match AWBW
- Stops at **first** failure — later envelopes in the same zip are not exercised until that failure is fixed

**CI default**

- **`pytest`** only — full desync corpus is manual / scheduled (`/.github/workflows/desync-smoke.yml`), not every PR

When a row is `ok` but play “looks wrong”, run **`replay_state_diff`** and open the zip in the **C# viewer** — silent gold/HP drift is a known class documented in `desync_audit.md`.

---

## 6. Host inventory

### Naming clarification

| Name | Reality |
|------|---------|
| **Main** | Canonical git + Samba publisher host |
| **workhorse1** | **Same physical machine as Main** (`192.168.0.160`). Fleet ID used for RHEA learner role, pool paths, and `fleet/workhorse1/` metadata — not a second box. |
| **pc-b** | Operator desk aux — active PPO solo training (`start_solo_training.py`) |
| **.122** | Windows aux (`C:\Users\sshuser\AWBW` + `Z:\` when mapped) |

### Host table

| Host | IP / access | Local repo | Samba mount | `AWBW_MACHINE_ID` | Primary workload |
|------|-------------|------------|-------------|-------------------|------------------|
| **Main / workhorse1** | `sshuser@192.168.0.160` | `D:/awbw` | *(publisher)* `\\192.168.0.160\AWBW` | `workhorse1` (RHEA), varies | Samba share, git canonical, RHEA value learner |
| **pc-b** (this dev box) | local | `D:/awbw` or `D:/AWBW` | `Z:\` when mapped | `pc-b` | PPO solo training + fleet orchestrator |
| **.122 aux** | `sshuser@192.168.0.122` | `C:\Users\sshuser\AWBW` | `Z:\` (map in Console/RDP) | operator-assigned | Remote RHEA actor / aux training |

**Aux hierarchy:** Flat — no tiers among aux machines. Disambiguate host when scripting.

**Path casing:** Windows is case-insensitive; docs use `D:/awbw`, `D:/AWBW`, and `Z:\` interchangeably for the same tree on a given host.

### Git sync (code parity)

Standard ship workflow:

1. Edit on aux dev clone
2. `python -m pytest -q --tb=line` (CI-equivalent)
3. Commit + `git push`
4. `ssh sshuser@192.168.0.160 "D: && cd D:\awbw && git pull"`

Both clones should track `origin/main` at the same commit. Working trees differ in untracked runtime artifacts (checkpoints, fleet logs, scratch scripts).

**Source:** `.cursor/skills/awbw-regression-then-ship-main/SKILL.md`

---

## 7. Filesystem contract

### Samba share

Main publishes **`D:/awbw`** as SMB share **`\\192.168.0.160\AWBW`**.

| Platform | Mount | `AWBW_SHARED_ROOT` |
|----------|-------|-------------------|
| Windows aux | `Z:\` | `Z:\` (default in `rl/fleet_env.py`) |
| Linux aux | `/mnt/awbw` | `/mnt/awbw` |

**Caveat:** `net use Z: \\192.168.0.160\AWBW` from an OpenSSH session often fails with **error 1312** — map `Z:` in an interactive Console or RDP session.

### Checkpoint tree

All paths relative to repo root or `AWBW_SHARED_ROOT/checkpoints/`:

```
checkpoints/
  latest.zip                    # Shared fleet policy line (PPO)
  checkpoint_*.zip              # Managed PPO snapshots (prune targets)
  promoted/
    best.zip                    # Eval-promoted line
    candidate_*.zip
  bc/
    bc_warmstart_*.zip          # BC warmstart for --bc-init
  pool/
    <MACHINE_ID>/
      latest.zip                # Per-aux export snapshot
      checkpoint_*.zip          # Per-aux managed snapshots
  value_rhea_latest.pt          # RHEA value net (Samba-synced)
```

### Fleet metadata tree

```
fleet/
  <MACHINE_ID>/
    status.json                 # Trainer heartbeat
    probe.json                  # Hardware probe (probe_machine_caps)
    proposed_args.json          # Orchestrator-proposed train.py args
    applied_args.json           # Args hash actually running
    operator_train_args_override.json  # Operator-forced flags
    train.pid                   # Solo bootstrap PID file
    train_launch_cmd.json       # Last launched train command
    reload_request.json         # Hot weight reload (rollout boundary)
    eval/*.json                 # Symmetric eval verdicts
    mcts_health.json            # MCTS gate state
    transitions/*.jsonl         # RHEA remote actor batches
    transitions/*.jsonl.done    # Ingested by learner
```

### Aux write policy

Auxiliary processes may write only under (`rl/fleet_env.py`):

- `checkpoints/promoted/`
- `checkpoints/bc/`
- `checkpoints/pool/<MACHINE_ID>/`
- `fleet/<MACHINE_ID>/`
- `replays/`

Pool aux **must** set `AWBW_MACHINE_ID` and use `--checkpoint-dir` under `checkpoints/pool/<ID>/`.

---

## 8. Environment variables

| Variable | Values | Purpose |
|----------|--------|---------|
| `AWBW_MACHINE_ROLE` | `main` (default) \| `auxiliary` | Fleet identity |
| `AWBW_MACHINE_ID` | e.g. `pc-b`, `workhorse2` | Pool subdir + fleet metadata |
| `AWBW_SHARED_ROOT` | `Z:\` or `/mnt/awbw` | Samba mount (required on aux) |
| `AWBW_CHECKPOINT_DIR` | path under repo or share | Pool trainer override |
| `AWBW_TORCH_COMPILE` | `1` to enable | Policy `torch.compile` (off on Windows by default) |
| `AWBW_FPS_DIAG` | `1` | Per-rollout throughput JSONL |
| `AWBW_TRACK_PER_WORKER_TIMES` | `1` / `0` / unset | Per-subprocess step timing |
| `AWBW_LOG_REPLAY_FRAMES` | `1` | Heavy per-step frames in `game_log.jsonl` |
| `AWBW_SEAT_BALANCE` | optional | Learner on either seat |
| `AWBW_REWARD_SHAPING` | `phi` (default) | Φ vs legacy level rewards |

**Validation:** `validate_fleet_at_startup()` in `rl/fleet_env.py` — aux must see `checkpoints/` and `data/` under shared root.

**Source:** `rl/fleet_env.py`, `README.md`

---

## 9. Transport policy: Samba vs SSH

| Subsystem | Transport | Rationale |
|-----------|-----------|-----------|
| Shared repo tree (`checkpoints/`, `data/`, `fleet/`) | **Samba** | One canonical tree; aux reads/writes pool + metadata |
| RHEA `value_rhea_latest.pt` + transition JSONL | **Samba** | Small/medium files; batch-friendly; observable with `dir Z:\fleet\*\transitions\` |
| Bulk PPO `checkpoint_*.zip` cross-host | **SSH/rsync preferred** | Samba poor for multi-hundred-MB zips (latency, partial-write visibility) |
| Git code sync | **Git + SSH pull on Main** | Not Samba |

**Implemented SSH sync:** `tools/fleet_scp_sync.py` → `tools/sync_checkpoint_zips_fleet.ps1` — syncs `checkpoint_*.zip` only, **never `latest.zip`**.

**Designed but not fleet-wide today:** Hub-and-spoke rsync via pc-b for silent weight fan-out (`docs/multi_machine_weight_sync_design.md`).

---

## 10. PPO fleet operations

### Main (sovereign)

```powershell
python train.py
```

Uses local `checkpoints/` only. No share required. Optional: `--pool-from-fleet`, `--load-promoted`, `--bc-init`.

### Auxiliary pool training

```powershell
$env:AWBW_MACHINE_ROLE = 'auxiliary'
$env:AWBW_MACHINE_ID = 'pc-b'
$env:AWBW_SHARED_ROOT = 'Z:\'
python train.py --checkpoint-dir Z:\checkpoints\pool\pc-b\
```

With `--pool-from-fleet`, opponent merge includes top-level `checkpoint_*.zip` **and** all `pool/*/checkpoint_*.zip` under the shared checkpoints root.

### Solo walk-away (Tier 1) — pc-b

```powershell
python scripts/start_solo_training.py --machine-id pc-b --auto-apply
```

**Boot sequence:**

1. `tools/probe_machine_caps.py` → `fleet/<id>/probe.json`
2. `tools/propose_train_args.py` → `fleet/<id>/proposed_args.json`
3. `train.py` with proposed `--n-envs` / `--n-steps` / `--batch-size`
4. `scripts/fleet_orchestrator.py --shared-root . --pools <id> --apply`

**Orchestrator tick** (default dry-run off when `--apply`):

- Curate pool zips (K/M/D keeper sets)
- Symmetric eval: pool `latest.zip` vs shared `latest.zip`
- Promote winner → shared `latest.zip` (if thresholds met)
- Issue `reload_request.json` for chronic laggards
- MCTS health gate (read-only on host unless `--enable-mcts-here`)
- Audit log → `logs/fleet_orchestrator.jsonl`

**Tier 1 limitations:** No curriculum advisor from competence metrics; orchestrator does not auto-respawn crashed children.

**Source:** `docs/SOLO_TRAINING.md`, `docs/play_ui.md`, `README.md`

### Promotion gate (mandatory for promotion decisions)

Never promote from mtime/size alone. Run symmetric head-to-head:

```powershell
python scripts/symmetric_checkpoint_eval.py `
  --candidate "Z:\checkpoints\pool\pc-b\latest.zip" `
  --baseline "Z:\checkpoints\latest.zip" `
  --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 `
  --games-first-seat 4 --games-second-seat 3 --seed 0 `
  --max-env-steps 0 `
  --max-days 150 `
  --json-out logs/promotion_symmetric_pc-b_vs_shared.json
```

| Flag | Why |
|------|-----|
| `--max-env-steps 0` | Default 100 truncates early → no decisive wins |
| `--max-days 150` | Raises calendar tiebreak above engine `MAX_TURNS` (100) |

Read `promotion_heuristic_ok` in JSON. If `total_decided == 0`, eval is invalid — fix caps and rerun.

**Eval daemon:** `scripts/fleet_eval_daemon.py` — symmetric eval loop, verdict JSON → `fleet/<id>/eval/`.

**Manual promote:** `scripts/promote.py` (or orchestrator auto-promote).

**Source:** `.cursor/skills/awbw-pool-latest-vs-shared-latest/SKILL.md`

### Hot weight reload

At rollout boundaries, `rl/self_play.py` reads `fleet/<machine_id>/reload_request.json`, loads `target_zip` via `MaskablePPO.set_parameters(...)`, acks by renaming the request file. Use for injecting stronger weights without full restart.

### Deferred

`--shared-training` / MASTERPLAN §10 async weight sync via SQLite — **not implemented**.

---

## 11. RHEA fleet operations

RHEA is the **bootstrap** path: evolves full-turn plans; value net guides fitness. **Not PPO.**

### Architecture

```
workhorse1 (192.168.0.160)
  train_rhea_value_parallel.py  ← learner, owns replay buffer
    reads  checkpoints/value_rhea_latest.pt
    polls  fleet/*/transitions/*.jsonl
    writes checkpoints/value_rhea_latest.pt

remote actors (any Samba-mounted host)
  rhea_remote_actor.py
    reads  value_rhea_latest.pt (refresh every ~120s)
    writes fleet/<machine_id>/transitions/*.jsonl
```

### Learner (workhorse1) — live production command

**How it is launched today (verified 2026-07-01):**

| Piece | Path / name |
|-------|-------------|
| Scheduled Task | `AWBW_workhorse1_rhea_learner_v9` (Status: **Running**) |
| Detach wrapper | `fleet/workhorse1/_run_rhea_learner_wrapper.cmd` (writes `fleet/workhorse1/train.pid`) |
| Runner | `fleet/workhorse1/run_learner_v9_book.cmd` |
| Start/stop script | `fleet/workhorse1/start_rhea_learner_detached.ps1` (`-Stop` to kill) |
| Log | `logs/train_learner_v9_book.log` (stdout+stderr appended) |
| CWD | `D:\awbw` |

**Exact command** (from `fleet/workhorse1/run_learner_v9_book.cmd` — this is what workhorse1 runs right now):

```batch
cd /d D:\awbw
python -m scripts.train_rhea_value_parallel ^
  --map-id 171596 ^
  --co-p0 14,8,28,7 ^
  --co-p1 14,8,28,7 ^
  --max-days 30 ^
  --rhea-autotune ^
  --save-every-transitions 1000 ^
  --reward-weight 0.6 ^
  --value-weight 0.4 ^
  --value-lr 1e-4 ^
  --replay-size 50000 ^
  --min-replay-before-train 1000 ^
  --updates-per-turn 1 ^
  --gamma-turn 0.9925 ^
  --target-update-interval 1000 ^
  --grad-clip 1.0 ^
  --device cuda ^
  --n-envs 9 ^
  --gpu-actors 3 ^
  --phi-capture-phase-weighting ^
  --phi-safe-neutral-opening-mult 1.50 ^
  --phi-safe-neutral-early-mid-mult 1.30 ^
  --phi-safe-neutral-mid-mult 1.15 ^
  --phi-safe-neutral-late-mult 1 ^
  --phi-safe-neutral-endgame-mult 0.50 ^
  --phi-contested-neutral-opening-mult 1.25 ^
  --phi-contested-neutral-mid-mult 1.00 ^
  --phi-contested-neutral-late-mult 0.90 ^
  --phi-capture-opening-end-day 5 ^
  --phi-capture-early-mid-end-day 8 ^
  --phi-capture-mid-end-day 12 ^
  --phi-capture-late-end-day 15 ^
  --dual-gradient-hist-prob 0.2 ^
  --dual-gradient-self-play ^
  --pairwise-zero-sum-reward ^
  --machine-id learner ^
  --checkpoint D:/awbw/checkpoints/value_rhea_latest.pt ^
  --push-gradients ^
  --buy-mode exhaustive ^
  --rhea-tactical-beam-max-width 96 ^
  --rhea-tactical-beam-max-depth 28 ^
  --rhea-pv-max-followup-pairs 4 ^
  --rhea-pv-inner-budget-scale 0.25 ^
  --rhea-adaptive-hard-turn-wall-s 900 ^
  --rhea-adaptive-extend ^
  --rhea-adaptive-max-extra-generations 20 ^
  --rhea-adaptive-patience-generations 4 ^
  --rhea-adaptive-min-improvement 0.0003 ^
  --capture-completion-bonus 0.03 ^
  --capture-progress-bonus 0.02 ^
  --neutral-income-gap-weight 0.04 ^
  --blunder-exposure-weight 0.01 ^
  --hq-defense-weight 0.01 ^
  --capture-interrupt-bonus 0.01 ^
  --buy-air-context-penalty 0.05 ^
  --opening-book-path D:/awbw/data/designed_desires_opening_book.jsonl ^
  --opening-book-prob 1.0
```

**Matchup:** map `171596`, four-CO Jess-style pairing `14,8,28,7` both sides, 30-day cap, opening book always on (`--opening-book-prob 1.0`).

**Mode flags on this run:** `--push-gradients` (actors compute gradients locally; learner aggregates from `data/gradients/`), `--machine-id learner`, 9 subprocess actors (`--n-envs 9`, `--gpu-actors 3`). No remote `fleet/*/transitions` actors (`remote_ingested: 0` in heartbeats).

### workhorse1 v9+book — logs & artifacts (current run)

All paths under **`D:\awbw`** on workhorse1 unless noted. **One-shot snapshot:** `python scripts/monitor_workhorse1_rhea.py` (SSH).

#### Primary logs (check these first)

| Path | ~Size (2026-07-01) | What to look for |
|------|-------------------|------------------|
| **`logs/train_learner_v9_book.log`** | ~6 MB, **active** | **Richest source.** All stdout/stderr from the learner + 9 actors: `rhea_turn_timing` (per-turn `search_s`, `pv_wall_s`, `adaptive_*`, `buy_mode`), `rhea_buy_exhaustive`, `gradients_applied` / `gradient_poll_error`, `heartbeat`, `queue_timeout`, `game_done`, `actor_dead`. Tail: `Get-Content D:\awbw\logs\train_learner_v9_book.log -Tail 40` |
| **`logs/games_log.jsonl`** | ~16 MB, **active** | Structured **learner-process** events (same events as above that the main loop writes to file). Key `event` values below. |
| **`logs/game_log.jsonl`** | ~9 MB, **active** | **Per-finished-game** rows from `AWBWEnv.finalize_rhea_episode` → `_log_finished_game` (`log_schema_version` ~1.18): winner, turns/days, CO/map/tier, `evolved_gain`, Φ breakdown fields, opponent type, seat/tempo fields. Use for win-rate / competency analysis — **not** the same file as `games_log.jsonl`. |
| **`logs/slow_games.jsonl`** | ~5 MB, **active** | Subset of finished games where `episode_wall_s ≥ 60` (default `AWBW_SLOW_GAME_WALL_S`) and/or `invalid_action_count > 0`. Expect **most** RHEA episodes here (turns often take 10+ minutes with current search budget). Fields: `episode_wall_s`, `p0_env_steps`, `max_p1_microsteps`, `reasons`. |

#### `games_log.jsonl` event types (current run)

| `event` | Meaning |
|---------|---------|
| `heartbeat` | Every ~60s: `transitions` (gradient steps under `--push-gradients`), `games_done`, `replay` (buffer size; often 0 in push-gradients mode), `actors_alive`, `remote_ingested`, `uptime_minutes`, optional `last_value_loss` |
| `queue_timeout` | Main queue had no message for 30s — **normal** while actors grind a single RHEA turn (`search_s` 600–900s+). Check `seconds_since_last_transition` and `actors_alive` (should be >0). Not an error by itself. |
| `game_done` | One actor finished a full game; includes actor payload + `total_games_done` |
| `actor_dead` | Subprocess actor crashed — investigate `error` field |
| `transition` | Per-transition training log (mainly when **not** using `--push-gradients`) |
| `gradients_applied` | **Stdout only** (`train_learner_v9_book.log`) — learner applied a batch from `data/gradients/` |

Win-rate tooling: `tools/analyze_hist_winrate.py` reads **`games_log.jsonl`** (not `game_log.jsonl`).

#### Checkpoints & config (not logs, but same run)

| Path | Purpose |
|------|---------|
| `checkpoints/value_rhea_latest.pt` | Live value net — actors refresh from this |
| `checkpoints/value_rhea_<timestamp>.pt` | Timestamped backups every `--save-every-transitions 1000` |
| `checkpoints/hparams_parallel.json` | Frozen CLI args written at learner startup |
| `data/gradients/*.json` | Gradient inbox (`--push-gradients`); learner polls, applies, deletes |
| `D:\awbw\.tmp\awbw_rhea_session_games_*.sqlite` | Ephemeral game counter (deleted on clean exit) |
| `fleet/workhorse1/train.pid` | Scheduled-task wrapper PID (stale after crash) |

#### Stdout-only events (grep `train_learner_v9_book.log`)

These do **not** land in `games_log.jsonl`:

- `rhea_turn_timing` — per-turn search/replay/PV timing and action counts
- `rhea_buy_exhaustive` — buy candidate enumeration stats
- `gradients_applied`, `gradients_pushed`, `gradient_poll_error`, `gradient_delete_error`
- Day/turn progress lines from actors (`day=… score=… phi=… gain=…`)

#### On disk but **not** this run

| Path | Why ignore now |
|------|----------------|
| `logs/nn_train.jsonl` | **PPO / MaskablePPO** learner scalars only — stale on workhorse1 (last write ~2026-05) |
| `logs/fps_diag.jsonl`, `logs/fleet_orchestrator.jsonl`, `logs/start_solo_training*.log` | **pc-b PPO** solo bootstrap — not started by v9 book learner |
| `logs/train_learner_v9.log` | Prior learner launcher (superseded by `train_learner_v9_book.log`) |
| `logs/train_negamax_*.log` | Old negamax experiments |
| `fleet/*/transitions/*.jsonl` | Remote RHEA actors — none feeding this learner (`remote_ingested: 0`) |
| `logs/co_matchup_session_totals.csv`, `logs/game_chunk_rollups.csv` | Legacy/manual rollups in `logs/` — not written by current learner |

#### Quick reads (PowerShell on workhorse1)

```powershell
# Snapshot script (from dev box)
python scripts/monitor_workhorse1_rhea.py

# Heartbeats + timeouts
Get-Content D:\awbw\logs\games_log.jsonl -Tail 15

# Last finished game summary
Get-Content D:\awbw\logs\game_log.jsonl -Tail 1

# Slow / long episodes
Get-Content D:\awbw\logs\slow_games.jsonl -Tail 5

# Gradient apply lines
Select-String -Path D:\awbw\logs\train_learner_v9_book.log -Pattern gradients_applied | Select-Object -Last 5

# Per-turn search cost
Select-String -Path D:\awbw\logs\train_learner_v9_book.log -Pattern rhea_turn_timing | Select-Object -Last 5
```

For replay zips see [C# replay export](#c-awbw-replay-player--generating-zip-replays) below.

### Remote actor (any aux with Z: mapped)

```powershell
python -m scripts.rhea_remote_actor `
  --shared-root Z:\ `
  --machine-id workhorse2 `
  --checkpoint Z:/checkpoints/value_rhea_latest.pt `
  --transition-batch-size 100 `
  --n-envs 8 `
  --device cuda
```

**Transition flow:**

1. Actor writes `Z:/fleet/<machine_id>/transitions/*.jsonl`
2. Learner polls every `--poll-remote-transitions-interval` (default 60s)
3. Learner ingests via `add_batch()`, renames to `*.jsonl.done`
4. Learner trains, saves `value_rhea_latest.pt`
5. Actors refresh weights periodically

**Why filesystem not Redis/gRPC:** Samba already works; simple failure model; observable batches.

**Source:** `docs/multi_machine_rhea_training.md`, `docs/rhea_value_hparams.md`

### C# AWBW Replay Player — generating `.zip` replays

The in-repo Flask `/replay/` UI reads `logs/game_log.jsonl` (JSON) — **not** the format the C# **AWBW Replay Player** expects. For the desktop viewer, export a site-compatible `.zip` from the Python engine.

#### Zip layout (what the C# parser consumes)

| Zip entry | Contents |
|-----------|----------|
| `<game_id>` | Gzipped PHP-serialized turn snapshots: `O:8:"awbwGame":{...}` one line per player-turn start |
| `a<game_id>` | Gzipped `p:` per-action stream (Move / Build / Fire / End JSON in PHP envelopes) — **ReplayVersion 2**, per-move animation |

Without the `a<game_id>` entry, the viewer still loads but units **teleport** at turn boundaries (snapshot-only / ReplayVersion 1).

**Implementation:**

- `tools/export_awbw_replay.py` — `write_awbw_replay()` builds the zip; serializes PHP snapshots; when `full_trace` is passed, delegates action stream to `tools/export_awbw_replay_actions.py`
- `tools/export_awbw_replay_actions.py` — replays `GameState.full_trace` deterministically, emits `p:` envelopes (End, Build, Move, Fire, Capt, …)
- `tools/export_awbw_replay.py` — `write_awbw_replay_from_trace()` rebuilds a zip from a saved `.trace.json` without re-running search

P0/P1 are forced to **Orange Star (red)** / **Blue Moon (blue)** in exports regardless of map faction layout so the viewer colors match training seats.

#### Primary path: one RHEA eval game → zip (+ optional viewer launch)

`scripts/eval_rhea_one_game.py` plays one full RHEA game with the same machinery as training, then calls `write_awbw_replay(..., full_trace=env.state.full_trace)`.

```powershell
cd D:\awbw
python scripts/eval_rhea_one_game.py `
  --checkpoint D:/awbw/checkpoints/value_rhea_latest.pt `
  --map-id 171596 `
  --co-p0 14,8,28,7 `
  --co-p1 14,8,28,7 `
  --max-days 30 `
  --device cuda `
  --rhea-autotune `
  --buy-mode exhaustive `
  --opening-book-path D:/awbw/data/designed_desires_opening_book.jsonl `
  --opening-book-prob 1.0 `
  --no-open-viewer `
  --output-dir replays
```

Output: `replays/<game_id>.zip`. Omit `--no-open-viewer` to auto-launch the C# player (see below).

Pass the same RHEA/fitness flags as production (`--phi-capture-phase-weighting`, tactical beam sizes, competency bonuses, etc.) when comparing apples-to-apples with the learner.

#### Other export entry points

| Script | When to use |
|--------|-------------|
| `rl/ai_vs_ai.py` | PPO checkpoint self-play → zip (calls `write_awbw_replay` at game end) |
| `scripts/rhea_config_symmetric_eval.py` | Config A vs B seat-balanced eval; optional per-game zip |
| `tools/export_awbw_replay_actions.py` CLI | Regenerate `a<game_id>` or full zip from `replays/<id>.trace.json` |
| `tools/export_one_replay.py` / `tools/export_awbw_replay.py` CLI | Ad-hoc export from trace JSON or snapshot list |

#### Opening the zip in the C# viewer

1. Set `AWBW_REPLAY_PLAYER_EXE` to the built `AWBW Replay Player.exe`, **or** build under `third_party/AWBW-Replay-Player/AWBWApp.Desktop/bin/Release/...` (see `rl/paths.py::resolve_awbw_replay_player_exe`).
2. Double-click the zip, or let `eval_rhea_one_game.py` launch it (`--open-viewer`, default on).
3. Manual: `AWBW Replay Player.exe D:\awbw\replays\<game_id>.zip`

**Not the same as:** `logs/game_log.jsonl` + Flask `/replay/` (training debugger). **Not vendored:** the C# sources live upstream ([DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player)); this repo only emits compatible zips.

**Sources:** `tools/export_awbw_replay.py`, `tools/export_awbw_replay_actions.py`, `scripts/eval_rhea_one_game.py`, `.cursor/skills/awbw-replay-system/SKILL.md`

### RHEA competency bar

Target: **above cartridge versus-mode AI**, approaching **fast-play amateur human** (ranked standard, no fog).

| Failure mode | Machinery |
|--------------|-----------|
| Capture under-saturation | Φ κ progress chips without completion bonus |
| Unit blunders | No G-trade term in fitness |
| Nonsense buys (missiles on air-less maps) | `_bias_genome_toward_expensive_builds` in `rl/rhea.py` |
| Corner parking / turtling | `PHI_CONTROL_*` default off |

**Eval scripts:** `scripts/rhea_config_symmetric_eval.py`, `scripts/rhea_head_to_head.py`, `scripts/eval_rhea_one_game.py`

**Source:** `.cursor/skills/awbw-rhea-competency-bar/SKILL.md`

---

## 12. Play UI & behaviour cloning

```powershell
python -m server.app    # → http://localhost:5000/play/
```

Human is **player 0** (red); bot is **player 1** (blue). Policy always learns P0 in training.

**BC pipeline:**

1. Human steps → `data/human_demos.jsonl` (auto-append from `/play/`)
2. `scripts/train_bc.py --demos data/human_demos.jsonl`
3. Fresh PPO run: `train.py --bc-init checkpoints/bc/bc_warmstart_*.zip`

**Offline ingest:** `scripts/replay_to_human_demos.py` from trace JSON or oracle zip.

**MOVE rows:** Flat `action_idx` does not encode destination — BC skips MOVE by default.

**Source:** `docs/play_ui.md`, `docs/player_seats.md`

---

## 13. Logs & monitoring

### General (all training modes)

| File | Written by | Contents |
|------|------------|----------|
| `logs/game_log.jsonl` | `AWBWEnv._log_finished_game` | **Per-episode** summaries — PPO **and** RHEA (`finalize_rhea_episode`). Winner, matchup, diagnostics, `log_schema_version`. |
| `logs/games_log.jsonl` | RHEA learner main loop only | Process events: `heartbeat`, `queue_timeout`, `game_done`, `actor_dead`, … — **not** full episode schema |
| `logs/slow_games.jsonl` | `AWBWEnv` on abnormal episodes | `episode_wall_s ≥ threshold` and/or invalid actions |
| `logs/nn_train.jsonl` | PPO `MaskablePPO` learner | Loss / KL / explained_variance per update — **not** RHEA |
| `logs/fleet_orchestrator.jsonl` | Orchestrator | Per-tick decisions (eval, promote, restart, MCTS) |
| `logs/fps_diag.jsonl` | PPO diagnostics | env_collect_s, ppo_update_s, worker p99 |
| `logs/start_solo_training.log` | Solo bootstrap | Probe/propose failures, child exit |
| `fleet/<id>/status.json` | Trainer heartbeat | `{role, machine_id, task, last_poll, current_target, ...}` |
| `logs/desync_register.jsonl` | `tools/desync_audit.py` | Engine vs oracle replay defects |

**Do not confuse:** `game_log.jsonl` (finished games) vs `games_log.jsonl` (RHEA learner heartbeats).

### workhorse1 RHEA v9+book (current production run)

Full path table, event glossary, and “not this run” list: [§11 workhorse1 logs](#workhorse1-v9book--logs--artifacts-current-run).

**TensorBoard:** not used for RHEA value training on this run. PPO TensorBoard runs live under `logs/MaskablePPO_*` on **pc-b**, not workhorse1.

---

## 14. Replay surfaces

Two replay surfaces — **do not conflate**:

| Surface | Format | Use |
|---------|--------|-----|
| In-repo Flask `/replay/` | `logs/game_log.jsonl` JSON | Training/debug UI — **not** desync QA |
| C# AWBW Replay Player zip | PHP `awbwGame` + `p:` stream | Site ground truth; desync oracle target |

Engine invariants, export pipeline, and C# zip generation: [§11 RHEA replay export](#c-awbw-replay-player--generating-zip-replays), `.cursor/skills/awbw-replay-system/SKILL.md`.

**Engine QA / desync:** [§5](#5-engine-qa--desync-oracle).

---

## 15. Strategic phases (MASTERPLAN summary)

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation PPO validation, curriculum ladder | Active on pc-b |
| 2 | MCTS integration | Prototype + fleet MCTS gate wired |
| 3 | Hierarchical RL | Future |
| 10 | Multi-PC fleet (pool, promote, reload, orchestrator) | Partially implemented |
| 10g | Curriculum advisor from competence metrics | In flight / not Tier 1 solo |

Full detail: `MASTERPLAN.md`

---

## 16. Known limitations & open work

| Item | Status |
|------|--------|
| Durable (state, action) game corpus | **Open** — only summaries + value transitions today |
| `--shared-training` async weight sync | **Deferred** |
| Cross-host PPO zip fan-out over SSH | **Designed**, not fleet-default |
| Samba for bulk checkpoint zips | **Discouraged** — use SSH/rsync |
| Tier 1 solo: auto-respawn on crash | **No** — manual restart |
| RHEA `value_rhea_latest.pt` strength ≠ deployable PPO bot | Search+value only |
| PPO candidate-4096 head vs AWBWNet factored 35k head vs flat BC/book/corpus indices | **Deferred by operator decision (2026-07-01)** — align after RHEA line proven; see [§3 desync note](#3-neural-network-contract) |
| `fleet/pc-b/status.json` may show `"role": "main"` | Metadata quirk; convention is `AWBW_MACHINE_ROLE=auxiliary` on aux |

---

## 17. Quick-reference commands

```powershell
# --- Git ship ---
python -m pytest -q --tb=line
git add ... ; git commit -m "..." ; git push
ssh sshuser@192.168.0.160 "D: && cd D:\awbw && git pull"

# --- PPO solo (pc-b) ---
python scripts/start_solo_training.py --machine-id pc-b --auto-apply

# --- Promotion eval ---
python scripts/symmetric_checkpoint_eval.py `
  --candidate Z:\checkpoints\pool\pc-b\latest.zip `
  --baseline Z:\checkpoints\latest.zip `
  --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 `
  --games-first-seat 4 --games-second-seat 3 `
  --max-env-steps 0 --max-days 150

# --- RHEA remote actor ---
python -m scripts.rhea_remote_actor --shared-root Z:\ --machine-id gpu-box1 --n-envs 8

# --- Play UI ---
python -m server.app

# --- BC ---
python scripts/train_bc.py --demos data/human_demos.jsonl --save checkpoints/bc/bc_warmstart_play.zip
python train.py --bc-init checkpoints/bc/bc_warmstart_play.zip

# --- Checkpoint zip sync (SSH, never latest.zip) ---
python tools/fleet_scp_sync.py --direction pull   # example; see script for args

# --- workhorse1 RHEA learner status ---
python scripts/monitor_workhorse1_rhea.py

# --- Desync QA smoke ---
python tools/desync_audit.py --games-id 272176
python tools/run_desync_cluster.py --pr-smoke
python tools/replay_state_diff.py --games-id 1623065

# --- Legality probe ---
python tools/engine_awbw_legality_probe.py random --map-id 123858 --turns 60 --seed 0

# --- RHEA one-game replay zip (C# viewer) ---
python scripts/eval_rhea_one_game.py `
  --checkpoint D:/awbw/checkpoints/value_rhea_latest.pt `
  --map-id 171596 --co-p0 14,8,28,7 --co-p1 14,8,28,7 `
  --max-days 30 --device cuda --rhea-autotune --buy-mode exhaustive `
  --opening-book-path D:/awbw/data/designed_desires_opening_book.jsonl `
  --no-open-viewer --output-dir replays
```

---

## 18. Source index

| Topic | Primary doc |
|-------|-------------|
| **This file** | Fleet + AWBW + NN + encoding + QA hub |
| Desync oracle (full) | `docs/desync_audit.md`, `.cursor/skills/desync-triage-viewer/SKILL.md` |
| Engine ⊂ AWBW parity | `docs/engine_awbw_parity.md`, `tools/engine_awbw_legality_probe.py` |
| Desync bug backlog | `docs/desync_bug_tracker.md` (regen via `run_desync_cluster.py`) |
| AWBW & engine | `.cursor/skills/awbw-engine/SKILL.md`, `docs/player_seats.md` |
| Encoder / observation | `rl/encoder.py`, `docs/hp_belief.md` |
| Policy network | `rl/network.py`, `docs/restart_arch/MASTER_SPEC.md` |
| RHEA value net | `rl/value_net.py` |
| Action space layout | `docs/restart_arch/action_space_inventory.md` |
| NN compute budget | `docs/restart_arch/compute_budget.md` |
| Fleet hosts & Samba | `.cursor/skills/awbw-auxiliary-main-machines/SKILL.md` |
| Ship workflow | `.cursor/skills/awbw-regression-then-ship-main/SKILL.md` |
| Pool vs shared latest | `.cursor/skills/awbw-pool-latest-vs-shared-latest/SKILL.md` |
| North star & corpus | `.cursor/skills/awbw-superhuman-ppo-north-star/SKILL.md` |
| RHEA quality bar | `.cursor/skills/awbw-rhea-competency-bar/SKILL.md` |
| Engine & replay | `.cursor/skills/awbw-engine/SKILL.md`, `awbw-replay-system/SKILL.md` |
| Play UI & BC | `docs/play_ui.md` |
| Solo PPO bootstrap | `docs/SOLO_TRAINING.md` |
| RHEA multi-machine | `docs/multi_machine_rhea_training.md` |
| workhorse1 prod launcher | `fleet/workhorse1/run_learner_v9_book.cmd` |
| workhorse1 log snapshot | `scripts/monitor_workhorse1_rhea.py` |
| C# replay zip export | `tools/export_awbw_replay.py`, `scripts/eval_rhea_one_game.py` |
| Weight sync design | `docs/multi_machine_weight_sync_design.md` |
| RHEA hyperparams | `docs/rhea_value_hparams.md` |
| Player seats | `docs/player_seats.md` |
| Desync audit | `docs/desync_audit.md` |
| Fleet path logic (code) | `rl/fleet_env.py` |
| README fleet summary | `README.md` |

---

*This file is the consolidated operator spec. Update it when fleet topology, transport policy, primary training paths, or tensor contracts change.*
