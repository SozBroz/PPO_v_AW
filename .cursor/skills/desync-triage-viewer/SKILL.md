---
name: desync-triage-viewer
description: >-
  Triages desync_register.jsonl one defect at a time: pick the next row, report
  games_id/class/locator/action counts, and must start the local C# AWBW Replay
  Player with the zip plus --goto-* flags (never stop at copy-paste-only
  commands). Take extreme ownership of the affected zip: clear or document every
  other desync in that same replay, not only the row that triggered triage — see
  "Replay ownership". Each defect must close with replay delete (if scuffed),
  oracle fix, or engine fix — see "Mandatory closure". Use when validating engine
  vs site replays, desync_audit output, envelope indices, or "next bug" / "one
  replay at a time" triage. C# viewer location and lookup order: section 4a. Do not
  edit this skill after partial fixes; update it only when a replay is fully
  debugged (see "Skill updates").
---

# Desync triage + Replay Player deep link

Single-threaded workflow: **one replay at a time** — advance a saved cursor so the commander always knows where we are in the queue. Within that replay, take **full ownership** of every defect the zip surfaces (**Replay ownership**), not a single envelope and done.

## Ground rules

| Source of truth | Role |
|-----------------|------|
| `tools/desync_audit.py` → `logs/desync_register.jsonl` | Ordered defect list (per audit run); schema in `docs/desync_audit.md` |
| `third_party/AWBW-Replay-Player` (Desktop build; **exe paths — section 4a**) | Human comparator; must open the **same** `replays/amarriner_gl/{games_id}.zip` bytes as the audit |
| `tools/oracle_zip_replay.py` / `parse_p_envelopes_from_zip` | Envelope = one `p:` line = `(awbw_player_id, day, [action_json,…])` in file order |

Do **not** use the Flask `/replay` UI for AWBW zip parity (see `docs/desync_audit.md`).

## Replay ownership (the whole zip, not one row)

When triage points at a **`games_id`** / `replays/amarriner_gl/{games_id}.zip`, you
**own that entire replay**, not only the register row’s first failure.

**Do**

- After addressing the triaged defect (or reclassifying it), drive **`desync_audit
  --games-id {games_id}`** and/or **`replay_oracle_zip`** on that zip until the run
  **completes** (`ok`) **or** you have a **written list of every remaining blocker**
  in that same file (class, message, envelope/action depth). Treat “fixed the row
  that brought us here” as insufficient if the next audit would immediately surface
  a different `oracle_gap` / `engine_bug` later in the same stream.
- If fixing one issue unmasks another in that zip, **keep working that `games_id`**
  in the same campaign: oracle, engine, pool/seating, or delete-scuffed-artifact per
  **Mandatory closure** until the zip is honest or every residual failure is
  explicitly documented (for a follow-on owner).
- Prefer **batching investigation** on one `games_id` (multiple envelopes, multiple
  messages) over bouncing to unrelated zips before the current replay is exhausted.

**Relationship to “one defect at a time”**

- The **queue / cursor** (§2) may still advance **one register row** per user-facing
  handoff for bookkeeping — but **engineering work** on that `games_id` is not
  “done” until the zip passes the agreed oracle path **or** all failures in that
  replay are accounted for. Do not close Mandatory closure for that replay while
  a known second failure on the same zip remains unrun, unfixed, and undocumented.

## Mandatory closure (no open-ended triage)

For **every** defect taken from the register (each `games_id` / row the workflow hands off), triage is **not complete** until **exactly one** of the following is true — document which in chat or `docs/desync_audit.md` before advancing the cursor. Apply **Replay ownership** first: closure must reflect the **whole zip** for that `games_id`, not a single envelope, unless every other failure in that replay is already listed and deferred with cause.

| Outcome | When |
|---------|------|
| **1. Replay removed** | The zip or trace is **scuffed**: corrupt bytes, wrong game, **ReplayVersion 1** snapshot-only with no actionable `p:` stream, mirror garbage, or any case where keeping the file would keep re-burning the same false signal. **Delete** `replays/amarriner_gl/{games_id}.zip` (and any sidecar trace) after recording why; re-fetch only if the mission still needs a good copy. |
| **2. Oracle change** | The site/engine contract is understood and **`tools/oracle_zip_replay.py`** (or related replay tooling) is updated so **this defect class no longer surfaces** for honest zips (replay passes the oracle path, or the row is reclassified to a different actionable bucket). |
| **3. Engine change** | The failure is a real rules/state bug; **`engine/`** (or tightly coupled loader) is fixed so behavior matches AWBW / the viewer for that case. |

**Not allowed:** closing a handoff with only “hypothesis” or “needs more investigation” with none of the above. If the row is a **false positive** (register noise, catalog gap), either fix the **audit/oracle** so it stops appearing, or **delete** the bad artifact that caused it — still pick column **1**, **2**, or **3** and write it down.

## Skill updates (when to edit this file)

**Do not** revise `desync-triage-viewer/SKILL.md` on every intermediate debug step
(hypothesis, first patch, new `oracle_gap` after clearing `Load`, etc.).

**Do** edit this skill only when a **single replay** is **completely** debugged
for triage purposes, for example:

- `desync_audit` (or the agreed oracle path) runs that `games_id` **to completion**
  with no remaining failure for that row’s mission, **or**
- the defect is conclusively classified and the workflow/register conventions need
  a lasting doc change others will reuse.

Put transient findings in chat, `docs/desync_audit.md`, code comments, or the
register — not here — until closure.

## 1. Build or refresh the register

From repo root (paths default per `desync_audit.py` / `rl/paths.py`):

```powershell
python tools/desync_audit.py --register logs/desync_register.jsonl
```

Filter triage candidates with `--games-id` / `--max-games` when scoping.

## 2. Queue + cursor (one defect at a time)

Maintain **project-local** state (gitignored is fine) so sessions resume:

**Recommended path:** `logs/desync_triage_state.json` (alongside other `logs/` artifacts).

Minimal schema:

```json
{
  "source_register": "logs/desync_register.jsonl",
  "filter_class": ["engine_bug", "oracle_gap", "loader_error", "state_mismatch_investigate"],
  "skip_status": ["ok", "replay_aborted"],
  "sort": "games_id_asc",
  "items": [
    {
      "games_id": 1605367,
      "class": "oracle_gap",
      "zip_path": "replays/amarriner_gl/1605367.zip",
      "approx_day": 6,
      "approx_envelope_index": 10,
      "approx_action_kind": "Load",
      "actions_applied": 47,
      "envelopes_applied": 11,
      "message": "…"
    }
  ],
  "next_index": 0
}
```

**Rules for the agent**

1. **Build `items` once** from the register: read JSONL lines, drop rows whose `class` is not in `filter_class` (and optionally drop `status: ok` / aborted-only rows per mission).
2. **Sort** deterministically (`games_id` ascending, then `approx_envelope_index`, then `class`) so order never drifts between machines.
3. **Each user-facing handoff:** read `items[next_index]`, present **exactly one** item, then set `next_index += 1` after the user acknowledges (or when moving on). Never batch multiple defects in one message unless the user overrides.
4. If the user fixes oracle/engine and **re-runs** the audit, **rebuild `items`** from the new register and reset or merge `next_index` deliberately (document the choice in chat).
5. Before bumping `next_index` for that item, satisfy **Mandatory closure** (replay delete vs oracle vs engine) and record which column applied — and satisfy **Replay ownership** for that `games_id` (full zip clean or all remaining defects in that zip documented).

## 3. What to report for “where it broke”

From the chosen register row, always surface:

| Field | Meaning |
|-------|---------|
| `games_id`, `class`, `message`, `zip_path` | Identity + taxonomy |
| `approx_day`, `approx_envelope_index`, `approx_action_kind` | Locator from the failing oracle action (see `desync_audit._ReplayProgress`) |
| **`actions_applied`** | **Global oracle depth:** count of AWBW **JSON action objects** successfully applied **before** the one that threw. The failing action is **# (`actions_applied` + 1)** in the whole-replay oracle stream (1-based for humans). |
| `envelopes_applied` / `envelopes_total` | **Envelope progress:** half-turn `p:` lines fully completed before the run stopped, vs total envelopes in the zip. |

### “How many actions in the desync” (required wording)

Every handoff must state **both**:

1. **`actions_applied`** — e.g. “**80** oracle actions had already been applied when it blew.”
2. **Position inside the failing half-turn (optional but encouraged)** — use
   `parse_p_envelopes_from_zip` on the row’s `zip_path`, index
   `approx_envelope_index`, and report: *“In that `p:` line there are **N**
   actions; the failure is the **(k+1)**th (0-based index **k**), kind
   **`approx_action_kind`**.”*  
   Example pattern: same-turn **Load** then later **Unload** in one envelope
   often means transport/APC sequencing; if the register says `Load` and
   `oracle_gap`, the engine never reached **Unload** — the mapper choked on
   **Load** first.

Do **not** conflate `actions_applied` with “actions inside this envelope only”;
the register field is **cumulative** across all prior envelopes in file order.

Optional extra (helps the viewer): resolve **AWBW player id** for envelope
`approx_envelope_index` with `parse_p_envelopes_from_zip` — add
`diverge_awbw_player_id` to `AuditRow` in `tools/desync_audit.py` if you want it
persisted.

## 4. Open the replay on the right day / player (mandatory agent action)

### 4a. Where the improved C# Replay Player is (lookup order — do not skip)

Triage uses the **desktop AWBW Replay Player** with our **local improvements**
(`--goto-day`, `--goto-envelope`, `--goto-player`, etc.). That is the same C#
solution as upstream [DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player),
typically maintained as an **AWBW fork/clone with additions** — not the Flask
`/replay` UI. Sources and `Program.cs` live under:

`third_party/AWBW-Replay-Player/` (optional clone; often **gitignored** when
present — see repo `README.md` “Optional local clone”).

**Resolved executable path — try in this order until `Test-Path` succeeds:**

1. **`AWBW_REPLAY_PLAYER_EXE`** (environment variable): full path to
   `AWBW Replay Player.exe` when the commander keeps the build outside
   `third_party/` or on another drive.
2. **Release (default):**  
   `<repo>\third_party\AWBW-Replay-Player\AWBWApp.Desktop\bin\Release\net6.0\AWBW Replay Player.exe`
3. **Debug:**  
   `<repo>\third_party\AWBW-Replay-Player\AWBWApp.Desktop\bin\Debug\net6.0\AWBW Replay Player.exe`
4. **Newer TFM:** under `bin\Release\` or `bin\Debug\`, pick the subdirectory
   (e.g. `net8.0`) that contains `AWBW Replay Player.exe` if `net6.0` is absent.

If **no exe** after the above: the clone or build is missing — instruct to
`git clone` the fork into `third_party/AWBW-Replay-Player`, then from that tree
`dotnet build AWBWApp.Desktop/AWBWApp.Desktop.csproj -c Release` and re-check
path (2). **Do not** report “viewer not found” as a dead end without listing
which of (1)–(4) were tested.

**Do not** end triage with only a shell command for the user to copy. In a normal
desktop session the agent **must** start the viewer process itself (e.g.
Windows: `Start-Process` with absolute paths to the resolved exe and the
`zip_path`, plus the goto flags below).

Only skip launching if the user explicitly opts out or the environment cannot
spawn GUI processes (state so in chat — and still paste the **exact** resolved
exe path you would have used).

| Flag | Meaning |
|------|---------|
| `--goto-turn=N` | 0-based half-turn index (highest precedence) |
| `--goto-envelope=N` | Same index when one `p:` envelope ↔ one `TurnData` row |
| `--goto-day=D` | First half-turn with that AWBW **day** |
| `--goto-player=ID` | With `--goto-day`, prefer `TurnData.ActivePlayerID == ID`; else first half-turn on that day |

**Launch shape (PowerShell):** pass each argument separately so paths with
spaces resolve: `-FilePath <exe> -ArgumentList '<abs.zip>','--goto-day=6','--goto-player=3712502'`.

Implementation: `AWBWAppGame` goto statics + `ResolveAndConsumeInitialGotoTurn`;
`Program.cs` argv parsing; `ReplayController.LoadReplay` applies the turn after
load. Hints clear after one load.

**Within-turn action offset** is not automated — after the window opens, tell
the user to use **Next action** until `approx_action_kind` matches if needed.

### Manual fallback

If the agent truly cannot spawn the process, say why **and** echo the resolved
exe path from section 4a (or state that (1)–(4) all failed `Test-Path`), then
give the same absolute-path launch line as a last resort.

## 5. `desync_audit` enhancement backlog (optional)

Persist on the register row at failure:

- `diverge_awbw_player_id` (from current envelope tuple before the failing action)
- `diverge_action_index_in_envelope` (0-based within that half-turn’s action list)

That removes guesswork for the C# `InitialGotoActionOffset` step.

## 6. Related repo paths

| Artifact | Path |
|----------|------|
| Audit tool | `tools/desync_audit.py` |
| Register schema / triage doc | `docs/desync_audit.md` |
| Oracle + envelopes | `tools/oracle_zip_replay.py` |
| PHP snapshot diff | `tools/replay_state_diff.py` |
| Replay skill (zip/export) | `.cursor/skills/awbw-replay-system/SKILL.md` |
| Desktop entry + zip argv | `third_party/AWBW-Replay-Player/AWBWApp.Desktop/Program.cs` |
| **Built viewer exe** | `third_party/AWBW-Replay-Player/AWBWApp.Desktop/bin/Release/net6.0/AWBW Replay Player.exe` (or `Debug/…`, or override `AWBW_REPLAY_PLAYER_EXE`) |
| Post-load navigation | `third_party/AWBW-Replay-Player/AWBWApp.Game/Game/Logic/ReplayController.cs` (`GoToTurn`) |
| Turn metadata | `third_party/AWBW-Replay-Player/AWBWApp.Game/API/Replay/ReplayData.cs` (`TurnData.Day`, `ActivePlayerID`) |
| Subtype clustering (incl. `oracle_fire`) | `tools/cluster_desync_register.py` → `logs/desync_clusters.json` or `--markdown docs/desync_bug_tracker.md` |

## 7. Subtype focus: `oracle_fire` (`oracle_gap`)

Register rows only store `class` + `message`; the **`oracle_fire`** bucket is the
`desync_subtype()` rule in `tools/cluster_desync_register.py`: `oracle_gap` whose
message is recognized as **Fire-shaped** before other buckets. If the message
contains **`active_player`**, that rule classifies **`oracle_turn_active_player`**
instead — so “pure” `oracle_fire` in `desync_clusters.json` usually excludes
seat-skew strings (triage those under turn/envelope alignment first).

### Typical `message` prefixes (what to verify in the C# viewer)

| Shape | Meaning |
|-------|---------|
| `Fire (no path): no attacker …` | `Move.paths` empty; attacker tile + `units_id` from `Fire.combatInfoVision.global.combatInfo.attacker`. Oracle finishes stale ACTION, may use `copValues.attacker.playerId` / `units_players_id`, then `_resolve_fire_or_seam_attacker`. |
| `Fire (no path): cannot advance to acting player …` | Attacker resolved but `END_TURN` could not reach that seat — overlaps **envelope vs half-turn** issues; confirm `p:` owner vs engine `active_player` at failure depth. |
| `Fire for engine P… but active_player=…` | **With** `Move.paths`: mover’s `units_players_id` maps to engine seat `eng`, but engine is still on the other seat (often same root cause as `oracle_turn_active_player`). |
| `Fire: no attacker for engine P… at path … / global …` | Path branch: walk geometry + AWBW id hints failed to place the firing unit on the board the oracle expects. |
| `… [oracle_fire: strike_possible_in_engine=… triage=…]` | **Suffix on no-attacker failures** (`_oracle_fire_no_attacker_message_suffix`): `=1` / `resolver_gap_or_anchor` → some unit can still `get_attack_targets` that tile (narrow resolver fix); `=0` / `drift_range_los_or_unmapped_co` → no legal strike in current engine snapshot (viewer + engine parity). Full table: `docs/desync_audit.md` § `oracle_fire`. |

### Code map (where to patch)

- **`tools/oracle_zip_replay.py`** — `apply_oracle_action_json` → `if kind == "Fire":`  
  Two branches: **no path** (`not paths`) vs **path** (`_apply_move_paths_then_terminator` / direct attack resolution). Helpers: `_resolve_fire_or_seam_attacker`, `_guess_unmoved_mover_from_site_unit_name`, dense path / anchor bridge scans on the path branch.

### Queueing Fire-only defects

1. Regenerate clusters:  
   `python tools/cluster_desync_register.py --register logs/desync_register.jsonl --json logs/desync_clusters.json`
2. Read the **`oracle_fire`** array from the JSON (sorted `games_id`), **or**
3. Filter JSONL in-process: keep lines where `class == "oracle_gap"` and
   `(message.startswith("Fire") or "Fire " in message[:50])` and **`active_player` not in message** if you want to avoid the turn-skew slice.

In the desktop viewer, land on `approx_day` / `approx_envelope_index`, then step
actions until **`Fire`** matches; compare nested **`Move`** (paths + `unit.global`)
and **`Fire.combatInfoVision`** defender/attacker tiles to the engine state at
`actions_applied`.

## 8. Agent response template (one handoff)

Use a fixed shape so every defect looks the same:

1. **Queue:** `next_index / len(items)` + short note if register was refreshed.
2. **Replay:** `games_id`, `matchup`, `tier`, `map_id`, absolute `zip_path`.
3. **Defect:** `class`, one-line `message`, `status`.
4. **Locator:** `approx_day`, `approx_envelope_index`, `approx_action_kind`, AWBW player id (if resolved).
5. **Depth:** `actions_applied` (global oracle actions before throw = failing action #`actions_applied+1`); `envelopes_applied / envelopes_total`; plus **in-envelope index / N** when computed (section 3).
6. **Viewer:** state that the process was **started** (and how), or why it could
   not be; optional short note for in-viewer clicks if action-level jump is not
   automated.

Stop after **one** item unless the user asks for the next immediately.
