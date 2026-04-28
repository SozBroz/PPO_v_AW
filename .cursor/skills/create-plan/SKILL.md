---
name: create-plan
description: >-
  Authoring and evolving Cursor plans: CreatePlan tool, .plan.md files, plan
  mode, todos, risks, absolute-path close. Use whenever the user references a
  plan, creates a new plan, edits or updates an existing plan, iterates or
  refines plan sections, critiques or extends a plan, merges feedback into a
  plan, discusses plan todos or plan file paths, mentions .plan.md or
  .cursor/plans, or asks for planning before implementation — including
  MASTERPLAN or strategy docs when the ask is structured as a plan deliverable.
  Do not use as the primary driver when the user is only giving a bare execute
  order with no plan-structure change (then implement). Read this skill before
  writing or rewriting plan content.
---

# Creating plans (Cursor)

Plans are the contract before execution. This skill standardizes how to write them and how to **close** them so the imperator always knows where the artifact lives.

## When to use (read this skill)

Treat the following as **mandatory** pre-read before you draft or change plan text:

- **Reference:** user points at a plan file, past plan, “the plan says…”, todo list in a plan, or pastes a plan path.
- **Create:** new feature / refactor strategy; user asks for a plan first; you use **CreatePlan**.
- **Edit / iterate:** user says update, revise, expand, merge, critique, ruthlessly review, add a section, change todos, or “reflect this in the plan”.
- **Plan mode:** any iteration that updates the plan document instead of executing.
- **Exclude:** user only says execute / implement / ship the approved plan with **no** change to plan structure — implement code; optional one-line plan path reminder per “Mandatory close” if helpful.

## Tooling

- Prefer **`CreatePlan`** so the plan is stored as a durable `.plan.md` file and the user can confirm in UI.
- If you **edit an existing plan** by hand, keep the same file path — do not scatter duplicates.

## Content standards

1. **Title** — First line of plan body: level-1 heading with a clear name.
2. **Overview** — One or two sentences: goal, scope, non-goals if needed.
3. **Todos** — Frontmatter `todos` array: short `id`, actionable `content`, `status: pending` until executed. Merge duplicates; avoid vague todos.
4. **Repository truth** — Cite real paths as **markdown links with full absolute paths** on Windows, e.g. `[train.py](D:\AWBW\train.py)`. Prefer “where to find X” tables for large efforts.
5. **Critical section** — Explicit risks, wrong assumptions, and “what not to do” beats polite vagueness.
6. **Diagrams** — Mermaid only when it clarifies; follow plan-tool rules (no spaces in node IDs, quote edge labels with special chars).
7. **Proportionality** — Small change = short plan. Do not paste the whole codebase.

## Mandatory close (user-facing)

**Every time you present a plan to the user** (chat reply after `CreatePlan` or after saving/editing the plan file), the **last substantive line** of your message must be the **absolute path** to that plan file on disk, on its own line, plain text (no markdown link required), for example:

```text
D:\Users\phili\.cursor\plans\human_vs_bot_ui_and_learning_1b156a53.plan.md
```

Rules:

- Use the **actual** path returned by the tool or shown in the workspace / `~/.cursor/plans/` listing — do not shorten, do not use `~`, do not make up a name.
- If the plan was created under the **project** tree (e.g. `D:\AWBW\.cursor\plans\...`), use that absolute path instead.
- Optional but good: add the same line as a final footer **inside** the plan markdown under a `---` and heading `## Plan file` so the document is self-locating when copied.

## After confirmation

When the user **approves** the plan, switch to execution mode and work todos in order unless they specify otherwise. Update todo `status` in the plan frontmatter as work completes (if editing that file is part of the workflow).

## Workspace rule (always-on trigger)

Cursor matches skills partly via `description`; for **every chat** that might only *reference* a plan without opening `*.plan.md`, add a project rule so the agent always sees the hook.

Create file **`D:\AWBW\.cursor\rules\create-plan.mdc`** (same folder as `persona.mdc`). If the environment blocks creating `.mdc` files, paste the template from the repo or switch to Agent mode and add it.

Template (full file body — copy verbatim):

````text
---
description: >-
  When the conversation is about creating, editing, referencing, or iterating
  a Cursor plan (.plan.md, CreatePlan, plan mode refinements, plan todos) and
  not solely a bare execute order, apply the create-plan skill.
alwaysApply: true
---

# Plan authoring (trigger)

If the user’s message involves **any** of: creating a plan; updating, editing, or merging into a plan; referencing a plan file or its contents; iterating plan-mode deliverables; critiquing or extending written plan sections; or discussing `.plan.md` / `.cursor/plans` artifacts — **read and follow**:

`.cursor/skills/create-plan/SKILL.md` (project root: D:\AWBW)

**When to skip this as primary work:** the user is **only** ordering execution of an already-fixed plan (e.g. “go implement the plan”) with **no** request to change plan text — then implement; the skill’s “mandatory close” (absolute plan path) still applies when you surface a plan to the user.

When **editing files** matching `*.plan.md`, follow this skill’s conventions.
````

Until that file exists, rely on the **skill `description`** above for discovery.

## Related

- Cursor skill authoring: [create-skill](D:\Users\phili\.cursor\skills-cursor\create-skill\SKILL.md) (personal; do not write new skills into `skills-cursor`).
- This repo’s engine/replay context: [awbw-engine](D:\AWBW\.cursor\skills\awbw-engine\SKILL.md), [awbw-replay-system](D:\AWBW\.cursor\skills\awbw-replay-system\SKILL.md).
