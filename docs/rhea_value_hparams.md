# RHEA Value-Learning Hyperparameters

The RHEA fork is not PPO. PPO hyperparameters such as `n_steps`, PPO
`batch_size`, clipping, entropy coefficient, and action logprob storage do not
drive the RHEA machine.

The RHEA machine uses:

- turn-level RHEA search budget
- turn-level replay buffer
- value batch size
- value learning rate
- turn-level gamma
- reward/value fitness schedule
- target network update cadence
- encoder freezing schedule

Machine A can keep running PPO. Machine B should use this value-guided RHEA loop.

## Initial search budget

Start conservative:

```yaml
rhea_population: 16
rhea_generations: 3
rhea_elite: 4
rhea_mutation_rate: 0.20
rhea_top_k_per_state: 16
rhea_max_actions_per_turn: 96
```

After legality and stability are proven:

```yaml
rhea_population: 32
rhea_generations: 5
rhea_elite: 4
rhea_mutation_rate: 0.20
rhea_top_k_per_state: 24
rhea_max_actions_per_turn: 128
```

Do not start huge. First prove that evolution improves over the initial
population. Log:

- `initial_best_score`
- `final_best_score`
- `evolved_gain`
- `illegal_gene_count`
- `actions_executed`
- `ms_per_turn`

If `evolved_gain` is usually near zero, evolution is not doing useful work yet.

## Reward/value fitness schedule

Early:

```yaml
reward_weight: 0.90
value_weight: 0.10
```

This means:

```python
fitness = 0.90 * shaped_turn_reward + 0.10 * value_head(afterstate)
```

Only increase value weight after ablations prove that value improves play:

```yaml
reward_weight: 0.80
value_weight: 0.20
```

Later:

```yaml
reward_weight: 0.70
value_weight: 0.30
```

Required ablations:

- RHEA reward-only
- RHEA value-only
- RHEA 90/10
- RHEA 80/20
- RHEA 70/30

Expected early winner: reward-only or 90/10.

## Value learner

Initial config:

```yaml
value_lr: 1.0e-4
value_batch_size: 128
replay_buffer_size: 50_000
min_replay_before_train: 1_000
updates_per_real_turn: 1
gamma_turn: 0.99
gradient_clip_norm: 1.0
weight_decay: 0.0
target_network: true
target_update_interval: 1000
target_clip: 5.0
```

If GPU utilization is low and replay is large:

```yaml
value_batch_size: 256
updates_per_real_turn: 2
```

If value loss oscillates or predictions explode:

```yaml
value_lr: 5.0e-5
value_batch_size: 128
gradient_clip_norm: 0.5
```

If learning is stable but slow:

```yaml
value_lr: 2.0e-4
value_batch_size: 256
updates_per_real_turn: 2
```

Do not start with the PPO LR unless only the final value head is unfrozen.

## Turn-level gamma

RHEA transitions are full acting-player turns, not single unit actions.

Recommended first value:

```yaml
gamma_turn: 0.99
```

Reasonable range:

```yaml
gamma_turn: 0.97 to 0.995
```

Use lower values if TD targets are unstable. Use higher values once terminal
outcomes and longer-horizon consequences are more reliable.

## TD target

Start with TD(0):

```python
target = turn_reward + gamma_turn * V_target(after_turn) * (1 - done)
loss = mse(V_online(before_turn), target)
```

Use a target network. Either hard update every 1000 learner updates or soft update
with `target_tau=0.005` later.

Later improvements:

- 3-5 turn n-step returns
- terminal win/loss mixing
- prioritized replay

## Encoder freezing schedule

Do not immediately fine-tune the whole PPO donor trunk on noisy early RHEA TD
targets.

Stage 1:

```yaml
freeze_encoder: true
unfreeze_last_resblocks: 0
value_lr: 1.0e-4
min_turn_transitions: 5_000
```

Stage 2:

```yaml
freeze_encoder: true
unfreeze_last_resblocks: 2
value_lr: 5.0e-5
min_turn_transitions: 20_000
```

Stage 3:

```yaml
freeze_encoder: false
value_lr: 3.0e-5 to 1.0e-4
min_turn_transitions: 50_000
```

If value drift makes play worse, go back one stage.

## RAM policy

The RHEA value learner should not store or train on:

- candidate features
- candidate masks
- action logprobs
- PPO advantages
- PPO rollout tensors

Store only turn-level spatial/scalar observations and scalar transition metadata.

The first value-only network should keep the PPO trunk and remove the actor side.
Do not shrink the trunk until the RHEA system works. Later, distill into a
smaller student such as 64 trunk channels / 6 residual blocks / 128 hidden size.