# Residual tower — compute budget (measurement + sizing)

**Scope:** STD ranked play, full observation (no fog). **Non-goals:** no mixed-precision / `torch.compile` claims unless enabled in-tree; trunk depth/width are **fixed** by the shipped restart bundle unless this doc and `rl/network.py` are revised together.

**Purpose:** Give the lead a sourced throughput baseline, parameter/FLOP scaling, a crude FPS projection, and sizing context anchored to the **shipped** restart network (`rl/network.py`, `rl/encoder.py`). Design narrative and migration notes: [superhuman restart architecture bundle](../../.cursor/plans/superhuman_restart_architecture_bundle.plan.md).

---

## Shipped restart contract (locked in code)

The following matches **`AWBWNet`** and shared trunk pieces in **`rl/network.py`** (constants from **`rl/encoder.py`** where noted):

- **`N_SPATIAL_CHANNELS = 77`**, **`N_SCALARS = 16`**.
- **Residual trunk:** **`TRUNK_CHANNELS = 128`**, **10×** **`_ResBlock128`** (stride-1 128→128, 3×3, BN + ReLU), after stem **`Conv2d(77 → 128, 3×3, padding=1)`** + ReLU — i.e. **10×128** trunk width/depth.
- **Scalar fusion:** **`scalar_to_plane`**: **`Linear(N_SCALARS, SCALAR_PLANES)`** with **`SCALAR_PLANES = 16`**; planes are **broadcast** over the full **30×30** grid and concatenated → **`FUSED_CHANNELS = TRUNK_CHANNELS + SCALAR_PLANES = 144`**.
- **Spatial path:** **Full 30×30** activations end-to-end on the trunk; **no** **`AdaptiveAvgPool2d((8,8))`** on the CNN trunk.
- **Policy:** Factored **1×1 conv** heads on **`xf`** — **`conv_select`**, **`conv_move`**, **`conv_attack`**, **`conv_repair`**, **`conv_build`** (27 build channels) — plus **`linear_scalar_policy`**: **`Linear(FUSED_CHANNELS, 16)`** for scalar logits (powers, capture/wait/load/join/dive-hide, unload slots). **No** dense **`Linear(hidden, 35_000)`**.
- **`ACTION_SPACE_SIZE = 35_000`**; flat logits are **assembled** in **`forward`** from the factored maps and scalar vector.
- **Value:** **`adaptive_avg_pool2d(xf, (1,1))`** → **144-d** **`g`** → **`value_head`**: **`Linear(144, hidden_size)`** → ReLU → **`Linear(hidden_size, 1)`** (default **`hidden_size = 256`**).
- **`AWBWFeaturesExtractor`** (same file): **same** stem, **10×** **`_ResBlock128`**, scalar broadcast fusion, then **`adaptive_avg_pool2d(..., (1,1))`** and a small **two-layer `fc`** to **`features_dim`** (default **256**) for Stable-Baselines3 **`features_dim`**.

---

## 1. Current measured throughput

| Machine | n_envs | env_steps_per_s_collect | env_steps_per_s_total | Source | Date |
|--------:|-------:|------------------------:|----------------------:|--------|------|
| pc-b (i7-13700F, RTX 4070) | 4 | ~720 | ~165 | `MASTERPLAN.md` §12 — cites real `train.py` validation at n_envs=4 | doc dated 2026-04-22 |
| pc-b | 6 | 716.9–1014.4 (iter 2–8 steady band) | 165.4–177.5 (after iter 2) | `docs/perf_audit/phase6b_iter5_cliff_findings.md` — `logs/fps_diag.jsonl` rows in repro | 2026-04-23 |

**Notes**

- §12 states the **binding gap** is ~**77% non-engine** (PPO update, IPC, lockstep wait), not raw engine speed; `env_steps_per_s_collect` stays far above `env_steps_per_s_total`.
- `train.py` **defaults** (this repo): `--n-envs 6`, `--n-steps 512`, `--batch-size 256`. `rl.self_play.SelfPlayTrainer` matches **n_steps=512**, **batch_size=256**, **n_epochs=10**. Many fleet notes still cite **n_envs=4** as a stable long-run setting on pc-b (see perf doc recommendation).
- The rows above are **end-to-end training** throughput; after the restart bundle, the learner forward path is the **shipped** **`AWBWNet`** (full-grid trunk + factored heads). Re-bench if comparing against pre-restart numbers.
- **`checkpoints_fps_validate_n4/` / `checkpoints_fps_validate_n6/`:** In this workspace the directories are **empty or absent** — no `metadata.json` or FPS logs were available to read.
- **Terminals:** A scan of `D:\Users\phili\.cursor\projects\c-Users-phili-AWBW\terminals\*.txt` showed no recent `env_steps_per_s_*` lines (e.g. active `ai_vs_ai` run only).

**If no on-machine number is trusted:** Run a clean bench **before** committing to tower size:

```text
python tools/bench_train_throughput.py --budget-seconds 300 --n-envs 4 --n-steps 512 --batch-size 256 --config-name tower_sizing_baseline
```

(`train.py` has **no** `--eval-fps` flag; throughput is taken from `logs/fps_diag.jsonl` via this harness or any normal training run.)

---

## 2. Current network parameter count

Constants: **`N_SPATIAL_CHANNELS = 77`**, **`N_SCALARS = 16`**, **`TRUNK_CHANNELS = 128`**, **`SCALAR_PLANES = 16`**, **`FUSED_CHANNELS = 144`**, **`ACTION_SPACE_SIZE = 35_000`** (`rl/encoder.py`, `rl/network.py`).

### Stem

- `Conv2d(77 → 128, 3×3, padding=1)` **with bias:**  
  weights `77 × 128 × 3 × 3 = 88 704`, bias `128` → **88 832**

### Residual blocks (`_ResBlock128`, conv layers **bias=False**; each `BatchNorm2d` has `2 × 128` trainable params)

Per block: two `Conv2d(128,128,3×3)` → `2 × (128×128×9) = 294 912`; BN → `512` → **295 424** × **10** = **2 954 240**

### Scalar broadcast

- `Linear(16, 16)` → `16×16 + 16 = **272**`

**CNN + scalar subtotal (stem + 10 blocks + `scalar_to_plane`):** **3 035 296**

### Factored policy heads (1×1 convs on 144 ch, plus scalar logits from pooled 144-d)

| Head | Shape | Params (weights + bias) |
|------|--------|------------------------:|
| `conv_select` | 144→1 | 145 |
| `conv_move` | 144→1 | 145 |
| `conv_attack` | 144→1 | 145 |
| `conv_repair` | 144→1 | 145 |
| `conv_build` | 144→27 | 3 915 |
| `linear_scalar_policy` | 144→16 | 2 320 |
| **Policy heads subtotal** | | **6 815** |

### Value head

- `Linear(144, 256)` + `Linear(256, 1)` → **37 377**

### Totals (`AWBWNet`, default `hidden_size=256`)

| Part | Params |
|------|-------:|
| **Trunk** (stem + 10× `_ResBlock128` + `scalar_to_plane`) | **3 035 296** |
| **Policy heads** (five 1×1 convs + `linear_scalar_policy`) | **6 815** |
| **Value head** | **37 377** |
| **Grand total** | **3 079 488** |

*Cross-check:* `python -c "from rl.network import AWBWNet; m=AWBWNet(); print(sum(p.numel() for p in m.parameters()))"` → **3 079 488** (workspace run).

Dense **`Linear(256, 35 000)`** (~9.0M params) is **gone**; the **~3.0M** trunk dominates trainable mass. **`AWBWFeaturesExtractor`** adds its own **`fc(144→256→256)`** when used under SB3 (**102 912** params for those two layers; **~3.14M** total extractor including shared trunk) — count separately if sizing an SB3 stack, not part of **`AWBWNet`** totals above.

---

## 3. Projected param count / FLOP scaling (10×128 locked)

**Locked unless revised:** **`D = 10`**, **`W = 128`** (`TRUNK_CHANNELS`), **70** input planes, **full 30×30** trunk, **144** fused channels, **factored** policy heads — as in **Shipped restart contract** above and `rl/network.py`. Do not treat §3 as a menu of alternate shipped depths; any **`D×W`** change implies a **code** change and must update this doc.

**Forward 3×3 conv MACs (batch 1, spatial 30×30, back-of-envelope):**

- **Stem:** `30×30 × 3×3 × 77 × 128` = **79 833 600**
- **Each** `_ResBlock128` **:** `2 × (30×30 × 3×3 × 128²)` = **265 420 800**; ×**10** = **2 654 208 000**
- **Shipped trunk 3×3 total:** **2 734 041 600** MACs (1×1 heads and value MLP are small vs this.)

**Ratio `Y` vs pre-restart narrow trunk:** The old **63→64→64→128→128** stack (3 residual blocks + **8×8** pool + wide `fc`) had a documented **~563.6M** MAC baseline in the historical appendix; **`Y ≈ 2.73e9 / 5.64e8 ≈ 4.84`** — same order as the prior **10×128** row in pre-restart planning tables, but the **parameter** story is different (**~3.08M** total **`AWBWNet`** vs **~11.8M** with dense policy).

For **hypothetical** future **`D'×W'`** exploration only (not shipped): reuse the usual AlphaZero-style per-block **`18·W² + 2·W`** (bias-off 3×3 pair; BN trainable params extra) and stem **`Conv2d(77 → W, 3×3)`** → **`77×W×9 + W`**; **omit** any **`AdaptiveAvgPool2d((8,8))`** flatten **`64·W`** fusion — that path is **pre-restart** (see appendix).

---

## 4. FPS projection model

**Given §12:** Faster engine mostly **does not** raise `env_steps_per_s_total`; extra **forward/backward** work in the learner **does**.

Let **`Y`** = ratio of **trunk 3×3 MACs** (candidate / **shipped** §3 baseline **2 726 784 000**) — proxy for per-forward conv cost; full PPO also scales with **`n_epochs=10`**, minibatch passes, and backward (~2–3× forward FLOPs), but **`Y` is a single knob** for comparison. For **pre-restart vs shipped**, use **`Y ≈ 4.8`** (§3).

**Naive lower bound (if NN were the whole wall clock):**

`env_steps_per_s_total,new ≈ env_steps_per_s_total,old / Y`

Example: `Y ≈ 4.83` → `165 / 4.83 ≈ 34` steps/s.

**Slightly less pessimistic (toy):** Suppose engine + non-NN overhead is **`~23%`** of wall (from §12’s “even 10× engine…” framing) and **all** of the remaining **`~77%`** scaled with NN cost **`Y`**. Then:

`T_new / T_old ≈ 0.23 + 0.77·Y` → `fps_new ≈ fps_old / (0.23 + 0.77·Y)`

For **`Y = 4.83`**, `0.23 + 0.77×4.83 ≈ 3.95` → **`165 / 3.95 ≈ 42`** env-steps/s.

**Bracket to use for planning:** For **`Y` near 4.8** (shipped vs old narrow trunk), expect **`env_steps_per_s_total` in the ~40–80 band** on the same box **if** the pc-b ~165–175 row remains representative — **measure**.

**Forwards per wall-clock second (orientation only):** Each env step does **≥1** policy/value forward in rollout; each PPO epoch does **multiple** forwards (and backwards) over **`n_steps × n_envs`** buffer with **`batch_size=256`**, **`n_epochs=10`**. The tower raises **both** rollout inference and update cost; **`Y` is not** a perfect multiplier on total FPS because backward/optimizer/IPC do not scale identically.

---

## 5. Recommendation

**Shipped:** **`D = 10`**, **`W = 128`** — matches the locked contract above and `rl/network.py`.

**Why (design intent, now reflected in code)**

- **~4.8×** trunk 3×3 MACs vs the **pre-restart** narrow CNN (§3–4) — meaningful capacity without the **~7.7×** regime of **16×128** in historical sensitivity tables.
- **Parameter mass** is **~3.1M** for **`AWBWNet`**; wall-clock remains the main risk, not raw weight count.
- **Width 192** at depth ≥10 was the **unattractive** region in pre-restart FLOP tables (**`Y ≥ 10.8`**); not pursued in the shipped bundle.

**Phase 1 Full horizon (§3 / §7: 5–20M steps)**

Single trainer, take **`fps ≈ 165`** as optimistic and **`fps ≈ 42–50`** as a **stress** bracket for **shipped vs pre-restart** from §4:

| Fleet | Steps | fps (optimistic) | Calendar (24/7) | fps (stress) | Calendar (24/7) |
|-------|-------:|-----------------:|----------------:|-------------:|----------------:|
| 1 PC | 5 M | 165 | ~8.4 h | 45 | ~30.9 h |
| 1 PC | 20 M | 165 | ~33.7 h | 45 | ~5.6 days |
| 2 PCs (§10 — independent PPO, shared weights) | 20 M | 2×165 | ~16.9 h | 2×45 | ~2.8 days |

**Two machines** roughly **halve calendar time** for the same global step target **if** both run at full duty cycle; they **do not** remove the need to keep **per-machine** `env_steps_per_s_total` acceptable.

---

## 6. Operational asks before implementation

1. **Per-machine FPS baseline** — same repo, same curriculum slice as production, pinned CLI. Example:

   `python tools/bench_train_throughput.py --budget-seconds 600 --n-envs 4 --n-steps 512 --batch-size 256 --config-name <machine_id>_pre_tower`

   Repeat with the fleet’s preferred **`n_envs`** (4 vs 6) and record **`diag/env_steps_per_s_total`** median from `logs/fps_diag.jsonl`.

2. **Optional micro-benchmark** — isolated `torch` forward (+ backward) on **`AWBWNet`** with batch **64** and observation shapes from `rl/encoder.py`, to separate **GEMM/conv** from **SubprocVecEnv** overhead.

3. **VRAM check** — after tower lands, one short `train.py` smoke with planned **`n_envs`**, **`n_steps`**, **`batch_size`**; confirm no OOM on the **4070-class** cards called out in perf docs.

4. **Log linkage** — stamp **`AWBW_MACHINE_ID`**, git SHA, and full `train.py` argv in bench JSON / run notes so **§1** can be filled from artifacts, not prose.

---

## 7. Risks / unknowns

- **FPS primary data** here is **MASTERPLAN §12** (n_envs=4) plus **one** detailed **n_envs=6** repro log — **no** fresh `fps_diag` from this workspace and **no** files under `checkpoints_fps_validate_*`.
- **GPU:** `train.py` defaults `--device auto` (CUDA when available). **`train.py` docstring** states **opponent inference stays on CPU** regardless — main learner uses **GPU** when `auto` resolves to CUDA.
- **Projection uncertainty:** §4 brackets are **order-of-magnitude**; real `env_steps_per_s_total` depends on **lockstep stragglers**, **RAM pressure**, and **PPO** constants (`n_epochs=10` fixed in `rl/self_play.py`).
- **FLOP proxy:** MAC count ignores **1×1** head convs (small vs trunk), **value MLP**, **optimizer**, and non-conv ops.
- **STD / no fog only** — no POMDP or belief-stack sizing.

---

## Appendix A. Pre-restart tower exploration (historical)

*The following assumed **`N_SPATIAL_CHANNELS = 63`**, **`AdaptiveAvgPool2d((8,8))`** on the trunk, **`Linear(256, 35_000)`** policy, and a **`fc`** on **`64·W + 17`** flattened features. It is **not** the shipped restart architecture; retained only for comparing old sensitivity tables to §3–4.*

**Modeling assumptions (sensitivity table only — pre-restart)**

- **Stem:** `Conv2d(63 → W, 3×3, padding=1, bias)` → `63×W×9 + W = 568W` params.
- **Tower:** `D` stride-1 blocks, fixed width `W`; per block (AlphaZero-style back-of-envelope per user brief): **`18·W² + 2·W`** (two `3×3` convs, bias off; BN/GN rounded as **`2·W`** in the brief — full PyTorch `BatchNorm2d` would be **`4·W`** per block).
- **Head of CNN path:** `AdaptiveAvgPool2d((8,8))` → flatten **`64·W`**; **`fc`**: `Linear(64W+17, 256)`, `Linear(256, 256)`.
- **Old policy:** **`Linear(256, 35_000)`** → **8 995 000** params.
- **Value head:** **`Linear(256, 1)`** → **257** params.

**Old trunk FLOPs/forward (rough):** Sum of `3×3` conv MACs at `30×30` (batch 1); stem `900×9×63×W`, each block two layers `2×900×9×W²`. **Baseline** pre-restart trunk (63→64→64→128→128): **563 558 400** MACs.

| Tower (D × W) | Trunk params | Trunk 3×3 MACs (× baseline) | New spatial-head params (`46W+318`) | Total params (trunk + head + value) | Mem @ B=64 (est.) |
|---------------|-------------:|----------------------------:|------------------------------------:|--------------------------------------:|-------------------:|
| 6 × 96 | 2 694 272 | 1.68× | 4 734 | **2 699 263** | ~253 MiB |
| 8 × 128 | 4 601 600 | 3.88× | 6 206 | **4 608 063** | ~562 MiB |
| 10 × 128 | 5 191 936 | 4.83× | 6 206 | **5 198 399** | ~703 MiB |
| 12 × 128 | 5 782 272 | 5.77× | 6 206 | **5 788 735** | ~844 MiB |
| 16 × 128 | 6 962 944 | 7.65× | 6 206 | **6 969 407** | ~1.13 GiB |
| 10 × 192 | 9 964 544 | 10.77× | 9 150 | **9 973 951** | ~1.05 GiB |
| 16 × 192 | 13 948 160 | 17.13× | 9 150 | **13 957 567** | ~1.69 GiB |

**Hand-check (8 × 128)** — trunk only (pre-restart):

- Blocks: `8 × (18×128² + 2×128) = 8 × 295 168 = 2 361 344`
- Stem: `568 × 128 = 72 704`
- fc: `(64×128+17)×256 + 256 + 65 792 = 2 101 760 + 65 792 = 2 167 552`
- Sum: `2 361 344 + 72 704 + 2 167 552 = 4 601 600` ✓

---

*Document: measurement + sizing for the restart bundle; architecture is defined in `rl/network.py` / `rl/encoder.py`. Bundle plan: [.cursor/plans/superhuman_restart_architecture_bundle.plan.md](.cursor/plans/superhuman_restart_architecture_bundle.plan.md).*
