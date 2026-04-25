# Phase 6B — n_envs=6 iter-5 cliff investigation

## Setup

- Machine: pc-b (16P/24L i7-13700F, 32GB RAM, RTX 4070)
- Repro command:

  ```text
  python tools/_repro_iter5_cliff.py --n-envs 6 --max-iters 8 --out logs/repro_iter5_n6.json --timeout-s 1200
  ```

- Date: 2026-04-23 (wall-clock run on operator dev box)
- Run summary:
  - Rollouts observed in `fps_diag.jsonl`: **8 / 8** (no early death)
  - Train subprocess: stopped via repro harness `terminate()` after iteration 8 (expected); **`exit_code` in summary is `1`** because a terminated child rarely reports `0` — not a training crash
  - Wall time (repro harness): **~139 s** (~2.3 min) from start to summary write
  - Pre-flight: **no** heavy `python` processes were running (`Get-Process python*` empty)

## Iter-by-iter trajectory

Source: `logs/fps_diag.jsonl` (eight lines, this run only). TB names in code use `diag/worker_step_time_p99_*_across_envs`; JSONL uses `worker_step_time_p99_max_s` / `worker_step_time_p99_min_s`.

| iteration | env_collect_s | main_proc_rss_mb | main_proc_rss_delta_mb | sum_worker_rss_mb | system_ram_used_pct | worker_step_time_p99_max_s | worker_step_time_p99_min_s | env_steps_per_s_collect | env_steps_per_s_total |
| ----------: | -------------: | ----------------: | ---------------------: | ----------------: | ------------------: | -------------------------: | -------------------------: | ----------------------: | --------------------: |
| 1 | 2.96 | 2179.9 | 1102.1 | 2695.8 | 70.4 | 0.00831 | 0.000615 | 1037.1 | 0.0 |
| 2 | 3.03 | 2494.6 | 1416.8 | 2695.6 | 70.9 | 0.01484 | 0.000744 | 1014.4 | 177.5 |
| 3 | 3.37 | 2494.1 | 1416.3 | 2705.6 | 70.5 | 0.01816 | 0.01278 | 912.6 | 173.9 |
| 4 | 3.81 | 2494.7 | 1416.9 | 2711.8 | 70.8 | 0.02876 | 0.01599 | 807.2 | 169.8 |
| 5 | 3.39 | 2494.5 | 1416.7 | 2717.6 | 70.7 | 0.01850 | 0.01503 | 905.1 | 170.6 |
| 6 | 4.29 | 2494.5 | 1416.7 | 2724.2 | 71.6 | 0.01799 | 0.000614 | 716.9 | 165.4 |
| 7 | 3.56 | 2494.8 | 1417.0 | 2731.0 | 72.0 | 0.01800 | 0.000755 | 863.2 | 171.7 |
| 8 | 3.59 | 2494.6 | 1416.7 | 2736.3 | 71.6 | 0.01823 | 0.00394 | 855.6 | 172.2 |

**Iter 5 vs operator-reported cliff:** collect time **3.39 s** (not 100+ s). End-to-end `env_steps_per_s_total` stayed **~165–177** after iter 2.

**Parallel psutil thread** (`logs/repro_iter5_n6_psutil.jsonl`): sampled every 2 s. Reported `sum_python_rss_mb` often **~5.2–5.9 GB** while callback `sum_worker_rss_mb` stayed **~2.7–2.74 GB**. That gap is consistent with **different definitions** (repro sums all Python descendants recursively vs callback’s rate-limited child scan) and **sampling during PPO spikes** (main-process RSS in psutil swings ~1.9–3.2 GB between samples). Peak `system_ram_used_pct` in summary: **74.1%**. Final sample shows `0` RSS (process already gone after terminate).

## Hypothesis verdict

| # | Hypothesis | Verdict | Evidence |
|---|------------|---------|----------|
| 1 | Memory leak / paging — RSS and `system_ram_used_pct` climb into the cliff | **REFUTED** (this run) | After iter 2, `main_proc_rss_mb` and `main_proc_rss_delta_mb` **plateau** (~2495 MB / ~1417 MB). `sum_worker_rss_mb` drifts **+41 MB** over 8 iters only. `system_ram_used_pct` **~70–72%** with no runaway. No iter-5 discontinuity. |
| 2 | Straggler / lockstep — `worker_step_time_p99_max_s` spikes vs flat min | **REFUTED** (this run) | `worker_step_time_p99_max_s` remains **&lt; 0.03 s** every iteration. Iter 5 is **not** an outlier vs iters 3–4. Nothing matches a barrier wait on the order of **100 s**. |
| 3 | Main-process bloat — delta concentrated in parent | **INDETERMINATE** | **~1.42 GB** main delta is **large but stable** after iter 2; no monotonic growth across these 8 rollouts. Could still interact with **longer** runs or different hyperparameters, but it did not explain a cliff here. |
| 4 | System-wide pressure — RAM % up while per-proc RSS flat | **REFUTED** (this run) | `system_ram_used_pct` is **flat** in the callback; no pattern of “host eats RAM while our RSS looks fine” leading into iter 5. |

## Root cause

**Not isolated.** Under the Phase 6b reproducer recipe (cold random opponent, narrow curriculum, `n_steps=512`, `batch_size=256`, map 123858, T3, fresh temp checkpoint dir, **24 576 total timesteps**), the **iter-5 cliff did not reproduce**. Diagnostics show a **healthy** n_envs=6 short run (~**170** env-steps/s total steady state), not the operator’s **~100 s** collect spike and silent death.

Most plausible reconciliation:

1. **Environment / host state** — background RAM/CPU/I/O, browser, other Python jobs, or higher baseline RAM use (~76% cited in plan vs ~70–72% here) could push a marginal configuration into paging or OOM-like failure **without** a clean Event Log signature.
2. **Run length / schedule mismatch** — operator observations used **longer** `train.py` sessions; this harness stops after **eight** rollouts. A leak or episodic blow-up might need more wall time or a different termination condition.
3. **Stochastic game length** — iter 8 logged **first** completed episode (`ep_len_mean` ≈ 3908). A rare long synchronous episode could still act as a straggler in other seeds; **not seen** in this sample.
4. **Config drift** — any difference vs the operator’s full CLI (curriculum, tiers, map pool, `n_steps`, opponent, checkpoint resume) can change memory and step-time tails.

**Single-run gate:** Per mission rules, **no second run** was executed. Non-reproduction is itself a **finding**: the cliff is **not deterministic** under the reproducer’s pinned settings on this day.

## Recommended remediation (ranked by ROI)

1. **Reproduce under the exact failing CLI** — capture full `train.py` argv from a run that dies; diff against repro defaults (`tools/_repro_iter5_cliff.py`). Extend `--iters` / wall time only after a match.
2. **If cliff returns** — save `fps_diag.jsonl` + `*_psutil.jsonl` + Windows **Resource Monitor** / RAM map for the hour; correlate iter-5 row with **ep_len**, **sum_worker** resample freshness, and **system_ram_used_pct**.
3. **Operational risk reduction** — until the failure is deterministic, keep **n_envs ≤ 4** on pc-b for long campaigns (already the documented sweet spot); treat n_envs=6 as **experimental**.
4. **Instrumentation tweak (low cost)** — log **`ep_len_max` / `episodes_per_rollout`** into the same `fps_diag.jsonl` row (currently in TB only) so cliff runs tie latency spikes to **episode completions** without cross-file joins.
5. **Host discipline** — for the next repro, log **baseline** `Get-Process` top RAM/CPU offenders and **free physical memory** before launch to test hypothesis 4 under controlled conditions.

## Caveats / limits of this investigation

- **Single run**; cliff reported by operator **did not fire**.
- Repro **short** (~8 rollouts, **~139 s** wall); may miss slow leaks or rare episodic tails.
- **`exit_code: 1`** in `repro_iter5_n6.json` reflects **harness termination**, not an observed Python traceback or OOM.
- **psutil vs callback RSS** disagree in absolute GB; interpret trends **within** each series, not across them blindly.
- No engine or training code was changed; **no fix** shipped from this pass.

---

*“The first principle is that you must not fool yourself — and you are the easiest person to fool.”* — Richard Feynman, *Cargo Cult Science* (1974)  
*Feynman: Caltech physicist; the essay is about keeping honest contact with what the experiment actually showed.*
