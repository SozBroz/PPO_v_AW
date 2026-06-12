---
name: awbw-rhea-competency-bar
description: >-
  Defines the RHEA competency bar (above the cartridge Advance Wars versus-mode
  AI, approaching fast-play amateur human) for ranked standard AWBW (no fog),
  with the failure taxonomy (capture under-saturation, unit blunders, nonsense
  buys like missiles/cruisers on air-less maps, corner parking, turtling), the
  repo machinery map, and AWBW-wiki-grounded doctrine. Use when improving RHEA
  competency, tuning fitness/phi shaping/buy logic, judging whether bot play is
  "good enough", building behavior metrics or eval gauntlets, or referencing
  the cartridge AI baseline. Ground truth for game rules must come from AWBW
  wikis; cartridge-AI internals from Wars World News disassembly threads.
---

# RHEA competency bar — ranked standard

## The bar (set by the imperator, fixed)

Between **better than the cartridge Advance Wars versus-mode AI** and **as good as the imperator playing fast turns**. Scope: ranked standard only — no fog, no gimmicks, standard tier bans (T0 + Black Bomb banned). Not scientifically quantifiable, but operationalized by the behavior gates below.

A bot at the bar:

1. **Saturates the map capture-wise** — neutral bases first, efficient capture phase, income saturated by midgame.
2. **Does not blunder units** — every attack/exposure survives a G-trade estimate; no value donations.
3. **Builds sanely** — no missiles/cruisers on maps with no air threat; infantry from bases nearly every turn; counters opponent builds (tank → AA → B-copter triangle).
4. **Positions with purpose** — no corner parking, no idle units, no turtling around own bases while contestable properties and key central areas go uncontested.

## Failure taxonomy → machinery map

Surveyed 2026-06-10. The active improvement plan is
`.cursor/plans/rhea_competency_suite.plan.md`.

| Failure | Root cause in code | Fix direction |
|---|---|---|
| Captures started, not finished | Φ κ rewards *progress* chips, no completion bonus (`rl/env.py` ~L3451-3501, pathology note ~L3459) | Completion bonus in fitness; contextual capture mults |
| Unit blunders | No trade-estimation term; value net only 10-20% of fitness; `engine/threat.py` planes unused in scoring | G-trade fitness gate using existing threat planes |
| Missile/cruiser spam | `_bias_genome_toward_expensive_builds` (88% flip to max-cost, `rl/rhea.py` ~L3027) + engine-legality-only buys | Map-context buy priors (enemy air? airports? ports?) |
| Corner parking / turtling | `PHI_CONTROL_*` (denial/center/forward) fully implemented but **default off** (`PHI_CONTROL_DELTA=0`) | Enable + tune; add idle-unit penalty |
| Not contesting | Capture-greedy teacher + phase weighting exist only in training env, never in RHEA search fitness | Bring contest signals into search-visible horizon |
| Can't tell if fixes help | No behavior-level metrics anywhere | `tools/competency_audit.py` (plan WS0) |

## RHEA machinery quick map

| Component | Path | Notes |
|---|---|---|
| Planner, genomes, autotune budget | `rl/rhea.py` | Two-phase (move RHEA → buy RHEA) default; pop 64 / gen 10 defaults; capture/attack seed bias 85% |
| Fitness | `rl/rhea_fitness.py` | `0.9×phi_delta + 0.1×value_advantage` + illegal/build penalties |
| Φ shaping (α army, β props, κ capture chips, control terms) | `rl/env.py` ~L3451-3794 | Control terms off by default |
| Tactical beam (finish_capture/killshot/strike buckets) | `rl/tactical_beam.py` | Default on in training; Cython |
| Buy search | `rl/buy_exhaustive.py` | `score_buy_candidate`; skip/hoard/bank shaping |
| Threat/influence planes | `engine/threat.py` | 6 planes, cached; feeds encoder ch 63-68 + budget complexity only |
| Autotune configs | `configs/rhea_autotune_*.json` | 18 files; prod family `rhea_autotune_prod_contact*` |
| Eval: config A vs B | `scripts/rhea_config_symmetric_eval.py` | Seat-balanced, capture/army metrics |
| Eval: ckpt vs ckpt | `scripts/rhea_head_to_head.py` | |
| Eval: one game + replay zip | `scripts/eval_rhea_one_game.py` | Qualitative review path |
| Win-rate from logs | `tools/analyze_hist_winrate.py` | `games_log.jsonl` |

## Doctrine (what "competent standard play" means)

Compressed from AWBW wiki — full extracts with sources in [reference.md](reference.md):

- **Build from every base every turn** (infantry default). Idle factories are the cardinal sin. Airports/ports need not fire every turn.
- **Efficient capture phase**: neutral bases first; early infantry chain multiple properties; balance backline vs frontline capture timing.
- **First vehicle is usually a tank** on small/mid maps; recon/APC openings are situational outside fog.
- **Spend money; counter new units immediately**: enemy copter → AA; enemy tank → tank/artillery. Saving up is a late-game move.
- **Meta units (standard)**: Infantry, Tank, B-Copter, Anti-Air core; Artillery and Md. Tank support; naval rarely; transports only for unreachable properties.
- **Combat = G-value trades**: free attacks > attacks from better terrain > attacks with protected attacker > attacks with backup/reinforcements. Cover everything, including infantry.
- **Game stages**: capture phase → midgame (tank/AA/copter triangle, formations, COP timing) → lategame (high-tech breakers, remote props, pipe seams).

## Cartridge AI baseline (the floor)

The versus-mode bot we must strictly beat is a **1-turn-deep, phase-ordered, reactive script** — decoded rules in [reference.md](reference.md). Key facts: per-turn phase list (captures → indirect fire → direct attacks → … → builds); attack iff estimated G-damage dealt ≥ G-taken (infantry valued ×2; skip if ≤10% dealt or ≥70% counter unless 4× favorable); reactive build table; no long-term planning; ignores terrain advantage often; obsesses over infantry. Its strengths (never idles factories, never suicides into bad trades, always captures with free infantry) are exactly the fundamentals RHEA currently fumbles — which is why a scripted reimplementation (`rl/scripted_baseline.py`, plan WS1) doubles as the promotion-gate opponent.

## Rules for agents working this campaign

1. **Game-rule ground truth = AWBW wikis only** (awbw.fandom.com, awbw.amarriner.com chart pages). Cartridge-AI internals may cite Wars World News forum disassembly. Do not trust generic Advance Wars fan content for AWBW-specific rules.
2. **Measure before tuning.** No fitness/Φ knob changes without the behavior audit (plan WS0) demonstrating the targeted failure and the post-change improvement.
3. **Fixes must be search-visible**: RHEA sees one turn + value net. Reward terms realized many turns out belong to the value net, not fitness shaping.
4. **Every new shaping term** ships with a sign/scale unit test and a gauntlet ablation. κ's progress-without-completion pathology is the standing cautionary tale.
5. **Keep it out of the PPO path** unless explicitly promoted — this is bootstrap work under `.cursor/skills/awbw-superhuman-ppo-north-star/SKILL.md`.
6. **Update this skill** with measured results and config verdicts when workstreams land; prune superseded claims.
