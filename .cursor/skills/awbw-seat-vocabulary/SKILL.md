---
name: awbw-seat-vocabulary
description: >-
  Translates informal “player 1 / player 2” or P1/P2 (first vs second human) into
  this repo’s engine seats P0/P1 (indices 0/1), red/blue, and log keys like co_p0/co_p1.
  Use when the user discusses seats, turn order, who is stronger, training vs opponent,
  or uses P1/P2 without saying “engine”; when reading prose that says “player 1” (ambiguous);
  or when AWBW red/blue vs engine index could be confused. Ask the user to disambiguate
  when P1 could mean either informal first human or engine seat 1 (second mover).
---

# AWBW seat vocabulary (informal P1/P2 → engine P0/P1)

The codebase uses **0-based engine seats** everywhere (`active_player`, `Unit.player`, `winner`, `funds[0]`/`[1]`). Names like **`co_p0` / `co_p1`** follow **P0 = seat 0**, **P1 = seat 1** — not “human player #1 / #2” in the colloquial sense.

Full human-readable spec: `docs/player_seats.md`. Training wiring (learner vs opponent): `rl/env.py`.

## 1. Canonical mapping (source of truth)

| Concept | Engine | Typical UI / training |
|--------|--------|-------------------------|
| First mover on symmetric starts | `active_player` **0**, `Unit.player` **0**; logs `p0_co`, `winner == 0` for that seat | Red in `/play/`; **MaskablePPO learner** steps seat 0 |
| Second mover | **1** | Blue; checkpoint opponent or heuristic in `AWBWEnv` |

## 2. Imperator informal notation → engine (default)

When the user says **“P1 / P2”** or **“player 1 / player 2”** without the word **engine**, assume:

| User says | Engine seat | Repo / log names |
|-----------|-------------|------------------|
| **Your P1** (first human / first side in your head) | **0** | Comments **P0**, keys **`p0_*`** (e.g. `p0_co`) |
| **Your P2** (second) | **1** | Comments **P1**, keys **`p1_*`** (e.g. `p1_co`) |

**Footgun:** In this repo, prose **“player 1”** often means **engine index 1** (second seat / blue), which is **not** casual “player one goes first.” Example: calendar `turn` advances when **seat 1** ends their turn — see `test_trace_182065_seam_validation.py` docstring.

## 3. When to AskQuestion (do not guess)

Stop and ask a short clarifying question (or multiple-choice) if:

- The user writes **“P1”** alone and it could mean **(A)** their informal first human **or** **(B)** engine seat 1 / blue / second mover (legacy “P1 seat” in code comments).
- They mix **“P1”** with **`--co-p1`**, **`p1_co`**, or **“Phase 1”** — those strings are **repo/roadmap** terms, not automatically “informal P1.”
- They ask **who is stronger** or compare models without tying to **color, seat index, or `winner` 0/1**.

## 4. Out-of-scope (do not remap)

- **MASTERPLAN “Phase 0 / Phase 1”** is program roadmap vocabulary, unrelated to player seats.
- **AWBW site / zip** `players[]` order and PHP IDs are mapped by oracle/export code; do not assume they equal engine `0`/`1` without pipeline context.

## 5. Agent workflow

1. **Restate** in chat: “You mean engine seat **0** (first mover / red) vs **1** (second / blue)?” when translating informal language.
2. **In code and logs**, keep repo identifiers (`co_p0`, `active_player`, etc.); optionally add a parenthetical: `(your P1 = seat 0)`.
3. **New user-facing copy:** prefer **red / blue** or **first seat / second seat** over bare **“P1”** if two readings exist.

## Non-goals

This skill does **not** rename identifiers, JSON catalogs, or `game_log` schema. It only aligns **language** between the user and the existing codebase.
