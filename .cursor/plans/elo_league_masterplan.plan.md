---
name: Elo league MASTERPLAN
overview: >-
  When ready, paste §10.5 into MASTERPLAN and add light cross-refs in §2 / §7.
  League-style ratings (Elo or Glicko-2) as eval/promotion only — not a training
  change. No code in this workstream until explicitly scheduled.
todos:
  - id: paste-masterplan-105
    content: >-
      Insert [Draft §10.5] below into [MASTERPLAN.md](c:\Users\phili\AWBW\MASTERPLAN.md) after §10.4;
      add optional one-liner in §2 stack + optional §7 row; bump Last updated
    status: pending
  - id: read-through-105
    content: Proofread §10 flow (10.1→10.5→§11) and tone (deferred / not Phase 1 gate)
    status: pending
isProject: true
---

# Elo league → MASTERPLAN (documentation task)

**Status:** Plan only. **Do not** implement Elo scripts or change training until a separate task says so.

## Intent

Document **league-style ratings** (Elo or Glicko-2) as a **strategic backlog** for **evaluation and promotion**—to mitigate **cyclic non-transitivity** (A beats B, B beats C, C beats A) that pairwise or single-baseline methods under-weight. This is **not** a PPO or weight-sync change.

| Artifact | Role |
|----------|------|
| [scripts/symmetric_checkpoint_eval.py](c:\Users\phili\AWBW\scripts\symmetric_checkpoint_eval.py) | Seat-symmetric H2H |
| [scripts/promote.py](c:\Users\phili\AWBW\scripts\promote.py) / fleet verdicts | `best.zip` gate |
| [rl/fleet_env.py](c:\Users\phili\AWBW\rl\fleet_env.py) `prune_checkpoint_zip_curated` | Pool quality/diversity |
| **Gap** | No global rating table over checkpoint identities |

Belongs in the same **eval ladder** family as MASTERPLAN §10.3 (BoN / Bo11, `best.zip` thresholding).

## Where to insert

**Immediately after** `### 10.4 Phase 3 — research` and before the `---` that precedes `## 11`.

## Draft text for MASTERPLAN (copy verbatim, adjust date if needed)

```markdown
### 10.5 Elo / league eval (backlog)

**Problem:** Head-to-head promotion on a **fixed baseline** or a **single pairwise** result can mis-rank policies when the checkpoint pool is **non-transitive** (rock-paper-scissors). The same effect shows up in pure self-play exploit cycles (see §9.2). Symmetric two-seat eval and win-rate heuristics help but do not replace a **global** ordering over many policies.

**Definition (this repo):** A **league** is a maintained **rating per checkpoint identity** (e.g. zip stem or `shared_training` manifest id), updated from **decided games** (win/loss/draw) using protocols that are **seat-symmetric** where possible (first/second seat balance, as in [`scripts/symmetric_checkpoint_eval.py`](scripts/symmetric_checkpoint_eval.py)).

**In scope | out of scope**

| In scope | Out of scope |
|----------|---------------|
| Inform **who plays whom** for fleet / BoN-style series; supplement §10.3 `best.zip` and threshold stories | Replace MaskablePPO, replace async weight sync, or block Phase 1a–1b on a league table |
| Optional tie-in to §10.4 “Opponent sampling policy” (matchmaking from measured rating bands) | “League only” as the sole Phase 1 Full gate (Phase 1 Full remains §3) |

**Inputs and outputs:** **Inputs** — H2H results from eval scripts, `fleet/<machine_id>/eval/*.json` verdicts (§10), and/or `game_log` when opponent checkpoint is tagged; keep **map / CO / tier** (or `curriculum_tag`) consistent or use **separate sub-leagues** so ratings are comparable. **Outputs** — a **persisted** rating table (format TBD: JSON, SQLite, or other) and optional **uncertainty** (Glicko-2 RD or Elo with confidence / minimum play counts).

**Phasing:** Treat as **Phase 2 eval infrastructure** (after Phase 1 Full on the target distribution and alongside “production” MCTS eval maturity). Not required for basic two-PC sync (§10.1–10.2).

**Risks and hygiene:** Low **N** per pair → require minimum games, pooled opponents, or Bayesian / Glicko-style uncertainty. **Non-stationary** line: `latest.zip` keeps moving—decide whether frozen checkpoints are rated in a **retired** pool vs continuously updating. Map and opening variance: prefer stratified or fixed eval slices for automation.

```mermaid
flowchart LR
  Checkpoints[Checkpoints]
  MatchResults[MatchResults]
  RatingTable[RatingTable]
  PromoteOrSchedule[PromoteOrSchedule]
  Checkpoints --> MatchResults
  MatchResults --> RatingTable
  RatingTable --> PromoteOrSchedule
```

**Implementation note:** No committed script path until built; a future tool may live under `scripts/` or `analysis/`.
```

## Optional cross-refs when pasting (same edit session as §10.5)

- **§2** — one line under the dependency stack code block: optional **league-style ratings** after Phase 1 Full, eval-only.
- **§7** — optional table row: Elo/league eval | after Phase 1 Full + enough eval games | low risk as telemetry; higher if gating promotion automatically.

## Verification (after paste)

- Single canonical `10.5` / “Elo” subsection; §10 reads 10.1 → … → 10.5 → §11.

---

## Plan file

```text
c:\Users\phili\AWBW\.cursor\plans\elo_league_masterplan.plan.md
```
