# Multi-Machine Weight & Strength Sync — Design

## 1. Context

- **Target:** a four-machine fleet. **Today** only **pc-b** (Windows, operator desk) is active for training. **Main** (`192.168.0.160`, `D:/awbw`) is offline for upgrades; a Windows aux at **`192.168.0.122`** uses the same repo locally plus **`Z:\`** when the Samba share to Main is mapped; a fourth host is TBD.
- **Constraint:** Samba is a poor transport for multi-hundred-MB checkpoint zips (latency, share contention, partial-write visibility). **Passwordless SSH from the operator’s PC to every fleet host** is assumed for control and for bulk moves.
- **Constraint:** **`scripts/fleet_orchestrator.py` runs only on the operator’s PC.** Each `train.py` runs locally on its machine with its own rollouts; nothing implements distributed PPO gradients across hosts.
- **Operator goal:** **silent sync** — no manual copying; weaker machines receive stronger weights automatically, without trashing their current rollout or corrupting checkpoints.

---

## 2. Status quo recap (one paragraph each, with file pointers)

### Phase 10a — local-disk + async publish

**Intent:** `train.py` can write checkpoints to a fast local mirror first; `rl/checkpoint_publisher.py` copies to the shared root in a background thread so Samba does not block the training hot path. Publish order is documented there: enqueue `checkpoint_*.zip` before `latest.zip` so pool globbers see a consistent ordering. **Single-machine / single-writer assumption in practice;** no cross-host weight fan-out. *(Plan: `.cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md` — fleet-local-publish todo; code: `rl/checkpoint_publisher.py`.)*

### Phase 10b — curated pool prune

**Intent:** Replace naive FIFO cap with a **union** of three keeper sets: **K** newest zips by mtime, **M** top stems by **verdict win rate** (from `fleet/<machine_id>/eval/*.json`), and **D** diversity buckets by training-step or mtime deciles; young files are protected by `min_age_minutes`. Scoring for the “top by win rate” slice is **`winrate = candidate_wins / (candidate_wins + baseline_wins)`** derived from symmetric-eval JSON (see `verdict_summary_from_symmetric_json` in `rl/fleet_env.py`). **Runs per directory** (shared root + each `checkpoints/pool/<id>/`); **does not move weights between machines.** *(Code: `prune_checkpoint_zip_curated` in `rl/fleet_env.py`; orchestrator: `FleetOrchestrator.curate_pools` in `scripts/fleet_orchestrator.py`.)*

### Phase 10c — in-process opponent refresh

**Intent:** Between rollouts, envs can refresh the frozen opponent pool so new `checkpoint_*.zip` exports appear without restarting `train.py`. **Improves opponent diversity on the machine that sees the files;** it does not fetch remote zips. *(Plan todo `fleet-opp-refresh`; orchestrator does not implement this — `train.py` / `rl/self_play.py`.)*

### Phase 10d — hot weight reload

**Intent:** At rollout boundaries, the trainer may read `fleet/<machine_id>/reload_request.json`, load **`target_zip`** with `MaskablePPO.set_parameters(...)`, and ack by renaming the request file. **This is the correct hook for “inject stronger weights without restart,”** provided `target_zip` points at a **fully written, machine-readable** path. *(Code: `_maybe_apply_reload_request` in `rl/self_play.py`.)*

### Phase 10e — orchestrator (single-machine pool today)

**Intent:** A file-system-driven tick (default **dry-run**) curates pools, runs **symmetric checkpoint eval** (`scripts/symmetric_checkpoint_eval.py`) of each pool’s `latest.zip` vs **shared** `checkpoints/latest.zip`, optionally **promotes** a winner into shared `latest.zip` (with `latest.zip.publishing` + `os.replace`), and issues **reload requests** for chronic laggards. Decisions append to `logs/fleet_orchestrator.jsonl`. **Promotion rule:** for each machine’s fresh verdict, promote if `winrate > 0.5 + reload_margin` (default `reload_margin=0.25` ⇒ threshold **0.75**) and `games_decided >= 2 * games_first_seat` (default first-seat games **4** ⇒ need **≥ 8** decided games). **Laggard reload:** if `winrate <= reload_margin` (default **0.25**) for `reload_consecutive` cycles (default **2**), write `reload_request.json` pointing at shared `latest.zip`. Heartbeats: `fleet/<id>/status.json` via `write_status_json` (`rl/fleet_env.py`), read by orchestrator. **There is no curriculum advisor wired in this script yet** (Phase 10g remains separate / in flight). **No code path publishes checkpoint blobs to peer hosts over SSH.** *(Code: `scripts/fleet_orchestrator.py`.)*

### Phase 10f — per-machine arg auto-tuning

**Intent:** `tools/probe_machine_caps.py` writes `fleet/<id>/probe.json` (CPU, RAM, GPU, disk write probe to checkpoint root). `tools/propose_train_args.py` derives `--n-envs`, `--n-steps`, `--batch-size` (with **pc-b** hard-capped at **4** envs for known stability). The orchestrator can **surface** proposals; **`--auto-apply`** optionally restarts training from `train_launch_cmd.json` / `train.pid`. **Tuning is per host;** it does not synchronize weights. *(Code: `tools/probe_machine_caps.py`, `tools/propose_train_args.py`, restart logic in `scripts/fleet_orchestrator.py`.)*

### Phase 10g — curriculum advisor (Composer K, in flight)

**Intent:** Competence-gated bootstrap and decay of knobs (greedy mix, capture gates, narrow tier), driven by rolling game-log metrics and persisted machine state — **policy / curriculum coordination**, not checkpoint transport. Expect orchestrator to **read** advisor state later; **not** the carrier for zip binaries.

**Explicit fact:** **Every item above is single-machine (or shared-filesystem-local) today. None of them push weights to peers over SSH or HTTP.** Cross-host “strength” is only meaningful once eval artifacts + checkpoint paths exist on a layout every machine agrees on (`MASTERPLAN.md` §10; fleet skills under `.cursor/skills/`).

---

## 3. Strength signal — “who is the strongest?”

| Candidate signal | What it actually measures | Pros | Cons |
|------------------|---------------------------|------|------|
| **Pool curator internal ranking** | **Not a global Elo.** Per-pool pruning ranks checkpoint stems by **symmetric-eval win rate** (`candidate_wins / (candidate_wins + baseline_wins)`), tie-broken by stem name, when eval JSON exists (`_verdict_winrate_by_stem` + sort in `prune_checkpoint_zip_curated`, `rl/fleet_env.py`). | Cheap; already grounded in play | **Within one pool vs that pool’s recorded baselines**, not a fleet-wide ordinal |
| **`scripts/symmetric_checkpoint_eval.py`** | Head-to-head **on a fixed map / tier / CO pairing**, with symmetric seat counts — same machinery the orchestrator already shells out to | **Ground truth** for “A beats B” under the chosen contract | High variance; costs wall time; must freeze snapshots (skill `awbw-pool-latest-vs-shared-latest`) |
| **Heartbeat / `status.json`** | **Operational liveness + optional `current_target`** (`write_status_json`, `rl/fleet_env.py`) | Tells you who is training and against what path | **Not** a strength metric unless you add a scored field and define its meaning |
| **Cross-machine head-to-head** | Each machine contributes a **candidate** `pool/<id>/latest.zip`; run symmetric (or BoN) evals between candidates | **Only** signal that respects per-machine exploration | Most expensive; needs scheduling |

**Recommendation (primary):** Use **fresh symmetric-eval summaries against a single fleet baseline checkpoint** as the **authoritative strength signal** — the same family of numbers already used for **promote** and **laggard** in `scripts/fleet_orchestrator.py` (`verdict_summary_from_symmetric_json` / `winrate` / `games_decided`). Concretely: **designate one baseline zip** (by default shared `checkpoints/latest.zip` once Main line is trusted, or a pinned `promoted/baseline.zip` if you need immutability). For each machine `id`, either consume the **newest** `fleet/<id>/eval/*.json` from the last orchestrator eval tick or run eval on demand. **Leader** = machine with highest `winrate` subject to **minimum `games_decided`** (reuse promote’s `2 * games_first_seat` floor). **Ties:** break by **higher `games_decided`**, then **newer verdict timestamp**, then **lexicographic `machine_id`** (deterministic fleet-wide).

**Poll interval:** **One eval pass per orchestrator tick** (plan default **30 minutes** via `--tick-minutes`) is enough; **do not** eval more frequently than new checkpoints could plausibly matter unless a machine signals “I just published” (future optimization).

**Secondary / audit:** Occasionally run **round-robin** symmetric pairs between **top-two** machines when the scalar vs baseline ties — expensive, not on the hot path.

---

## 4. Sync mechanism — comparison table

| Option | How it works | Bandwidth | Latency | Disruption to receiver | SSH/Samba dep | Recommended? |
|--------|----------------|-----------|---------|------------------------|---------------|--------------|
| **A. Push/pull rsync over SSH (machine-to-machine)** | Orchestrator (or a receiver-side script) triggers **`rsync` over SSH**: leader host holds the golden `latest.zip`; each laggard pulls into a **staging name** on **local disk** or **machine-private subtree**, then **`os.replace`**. Optional **`--bwlimit`**, **`--partial-dir`**. | **Low** (delta compression) | **One network hop** per pair | **Low** if staging + atomic rename + reload only at rollout boundary | **SSH only** for blobs | **Yes — primary steady-state** |
| **B. Hub via operator PC (rsync/scp twice)** | pc-b runs **`rsync` leader → local temp → laggard** (or `scp` equivalent) | **~2×** bytes across LAN vs A | **Two hops**; pc-b disk | **Low** (same staging discipline) | **SSH only**; fits **“pc-b keys to all”** day-one | **Yes — bootstrap / when leader lacks keys to laggards** |
| **C. Shared central blobs on Samba** | Everyone reads/writes `\\Main\awbw\checkpoints\...` | **High** (full copies + lock chatter) | **High** | **Medium–high** (partial reads, share stalls) | **Samba** | **No** |
| **D. Hybrid: tiny metadata on Samba, binaries via SSH** | Samba (or git) carries **manifest JSON** only: `{leader_id, sha256, schema_version, path_hint}`; orchestrator still moves bytes via SSH | **Low** for data | **One tick** + manifest read | **Low** | **SSH +** small Samba reads | **Optional adjunct**, not required if orchestrator already reaches all hosts |
| **E. HTTP file server per train.py** | Each host serves `/latest.zip`; laggards `curl` | **Low** | **Low** | **Low** if staged | **New** service, auth, firewall | **No** (needless operational surface area) |

**Winner:** **Option A** for **steady state** — **one hop**, **delta-friendly**, **no new daemons**, aligns with “SSH is already assumed.” **Option B** is the **explicit fallback** when only the operator PC can reach every host (no `leader → laggard` key mesh yet). **Reject C** for bulk sync; use Samba only for **git-synced repo**, small fleet JSON, logs — consistent with `.cursor/skills/awbw-auxiliary-main-machines/SKILL.md` and Phase 10a’s motivation. **Reject E** unless you later need WAN-scale transfer.

**One-paragraph justification:** You already treat checkpoint zips as **files with integrity and ordering constraints** (`checkpoint_publisher.py`, promote path). **Rsync over SSH** preserves those constraints with **partial transfer recovery**, **`--bwlimit`**, and a clear story for **staging outside** the path `train.py` reads until commit. Hub-and-spoke through pc-b costs **2×** bandwidth but **zero** changes to SSH trust topology and is the right **Phase B** bridge until Main and aux keys are normalized.

---

## 5. “Silent” criteria

1. **Reload boundary:** Apply injected weights only at **rollout boundaries** where Phase 10d already runs (`_maybe_apply_reload_request` after opponent refresh in `rl/self_play.py`). **Never** replace the zip **under** the path the publisher is currently writing.
2. **Staging path:** Land incoming files as e.g. `checkpoints/pool/<receiver_id>/incoming/leader_<leader_id>_<sha256_prefix>.zip.partial`, finish rsync, verify hash, **`os.replace`** into `.../incoming/leader_<...>.zip`. **Reload JSON** should point at this **final** path (or a symlink you control), **not** at a `.partial` file.
3. **Shared `latest.zip` caution:** Orchestrator’s reload template today targets **shared** `checkpoints/latest.zip` (`decide_reload` in `scripts/fleet_orchestrator.py`). On a **shared Samba root**, that path is **global** — racing writers from multiple machines can **poison** readers. **Fleet extension (opinionated):** prefer **`target_zip` per receiver** under that machine’s **`checkpoints/pool/<id>/...`** or local mirror, **or** a **single fleet promoted** file with **read-heavy / atomic replace** discipline and **no concurrent multi-writer** to the same object — document which invariant you enforce.
4. **Throttling / “no sync during receiver checkpoint”:** Before starting rsync, orchestrator reads **`fleet/<receiver_id>/status.json`**; if `task` indicates save/publish or a **“checkpoint epoch”** counter (future field), **defer**. Optionally require a **receiver-side `.sync_lock`** written by `train.py` around `_save_checkpoint_with_publish` (design-only here).
5. **Bandwidth cap:** Use **`rsync --bwlimit`** (or `scp -l`) defaulting conservative on 1 GbE shared with interactive use; **escalate** only when status heartbeats are green and no eval job runs.

---

## 6. Trust & integrity

| Topic | Today | Fleet requirement |
|-------|-------|-------------------|
| **Checksum sidecar** | Publisher path focuses on ordering; **no universal `sha256` sidecar** is enforced in `checkpoint_publisher.py` | **Mandatory:** `latest.zip.sha256` (or manifest JSON) **next to** every pushed zip; receiver **verifies before rename** |
| **Schema / version** | Eval JSON has `schema_version` in verdict summaries; checkpoint **encoder layout** must match across loaders (`rl/ckpt_compat.py` patterns) | **Mandatory:** embed **`log_schema_version` / encoder contract / git SHA** in a small JSON next to zip; orchestrator **refuses** mismatch |
| **Interrupted transfer** | Local publish uses replace; symmetric eval copies to `.tmp/eval_snap_*` first | **Never** load `*.partial`; only **`os.replace` into final name** after hash OK |
| **Promote path** | `_apply_promote` already uses `latest.zip.publishing` + `os.replace` (`scripts/fleet_orchestrator.py`) | **Reuse the same atomic pattern** for any global line |

---

## 7. Failure modes (table)

| Failure | Behavior |
|---------|----------|
| **SSH key revoked / host unreachable** | Orchestrator marks machine **skipped** for this tick; append **`noop` / alert** row to `fleet_orchestrator.jsonl`; **never** block other hosts |
| **Disk full on receiver** | `probe.json` already surfaces **free space**; orchestrator **skips** sync + emits **heartbeat_alert-style** detail; **no partial rename** |
| **Schema mismatch** | **Refuse** inject; surface **explicit** audit entry; optional **`reload_request` with reason** for operator visibility only |
| **Receiver `train.py` crashed** | No hot reload consumer; **pool files still update** for next process start; heartbeats go **stale** → existing stuck-worker path |
| **Operator PC offline** | Fleet **keeps training** on last weights; **no sync** until orchestrator returns — acceptable **eventual consistency** |
| **Network partition / split brain** | **Leader** must be **computed from last completed eval artifacts + deterministic tie-break**, not from live RPC; if a partition isolates the true leader, **each side may nominate a different leader** until eval JSON is visible again — mitigate by **writing leader choice to a small versioned `fleet/leader_manifest.json`** (on Samba or via **scp to all**) **only after** eval success |

---

## 8. Phased rollout

- **Phase A (now, single active host):** Lock the design; **dry-run** orchestrator paths; **fake** a second peer by staging a second `pool/` tree + `eval/` JSON on disk to validate **leader selection** and **manifest** parsing — **no new runtime deps.**
- **Phase B (Main online):** Restore **universal SSH** from pc-b to Main; validate **B (hub)** sync Main → pc-b → verify hash → **reload_request** on pc-b against a **staging** zip path; then **A** once `192.168.0.160` can **push** or aux can **pull** directly.
- **Phase C (122 + fourth machine, N general):** **Leader election** = **argmax winrate vs baseline** with tie-breaks in §3; **no distributed consensus protocol** needed. Optional **sticky leader** (hysteresis): require **Δ winrate > ε** or **two consecutive ticks** before flipping leader to avoid churn.
- **Phase D (steady state):** Fixed **tick** + **bwlimit** policy; **dashboard** = tail `fleet_orchestrator.jsonl` + per-machine `status.json` + **promotion / reload** counts; page if **sync skipped** N times.

---

## 9. Open questions for operator (numbered)

1. **SSH posture:** Are you willing to have the orchestrator invoke **`rsync`/`ssh` as your user** on pc-b (and optionally **non-interactive** on leader) — or do you require a dedicated **fleet-sync** Unix account with **forced command** restrictions?
2. **Disk budget:** Curator caps **per pool** are on the order of **K+M+D = 24** zips by default (`--keep-newest 8`, `--keep-top-winrate 12`, `--keep-diversity 4`). **Four machines × pool exports + shared root + promoted history** — what **GiB ceiling** per host and on Main is acceptable? Should **incoming/** be **one file** or a **ring**?
3. **Tie-breaking:** If two machines **tie** on win rate and games decided, is **lexicographic `machine_id`** acceptable, or do you want **prefer Main**, **prefer higher `num_timesteps`**, or **prefer newer checkpoint mtime**?
4. **Global vs per-machine `latest.zip`:** Should **reload** continue targeting **one shared** `latest.zip` (fleet-wide line) or **per-machine target paths** to kill Samba races?
5. **Leader baseline immutability:** Should symmetric eval always use **moving** shared `latest.zip` or a **pinned baseline** updated only on promote (reduces evaluation non-stationarity)?
6. **Windows rsync:** Will you install **cwRsync/WSL rsync** on every Windows aux, or standardize on **`scp` + ssh** only (simpler deps, worse delta)?

---

## 10. Recommendation summary

**TL;DR:** Treat **symmetric-eval win rate vs one fleet baseline** (already computed into `fleet/*/eval/*.json`) as **the strength oracle**. **Sync checkpoints with `rsync` over SSH** into **per-receiver staging paths**, verify **SHA-256**, then **atomic rename**; trigger **Phase 10d** with a **`reload_request.json` whose `target_zip` points at the final local path** — avoid streaming multi-GB objects through Samba for sync. Until **leader→laggard** SSH is wired, **relay through pc-b** (2× bandwidth, simplest trust model). **Samba** remains for **repo + small metadata**, not binary diffusion.

**Estimated implementation effort:** **~3–6 composer-days** for production-grade sync (manifest, staging, orchestrator hooks, failure telemetry), assuming SSH/rsync already works fleet-wide; **+2–3 days** if Windows packaging and **per-machine reload paths** require train/orchestrator contract changes.

**Tiering:** **Tier 2** (“operator walks away, fleet syncs silently”) = **A/B transport + hash + atomic rename + rollout-boundary reload + disk/SSH guards** on **2–N** machines. **Tier 3** (“full cluster polish”) = **hysteresis**, **round-robin audits**, **dashboard**, **automatic leader manifest broadcast**, and **bandwidth-adaptive** scheduling tied to eval queue depth.

---

*Checklist note for reviewers:* **Every Phase 10a–10e behavior cited here is single-machine / shared-tree-local today;** peer weight sync is **greenfield** on top of Phase 10d and the orchestrator shell.
