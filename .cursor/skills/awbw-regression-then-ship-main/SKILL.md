---
name: awbw-regression-then-ship-main
description: >-
  After code changes, runs CI-equivalent pytest as full regression; on success,
  commits, pushes from Auxiliary, then SSHs to Main and git pull in D:/awbw. Use
  when the user wants to ship validated work, sync both machines, post-regression
  commit, push and pull on main, or "green then push" after engine/test edits.
  Pair with awbw-auxiliary-main-machines for host roles. Does not replace
  optional heavy gates (e.g. full desync corpus) unless the user names them.
---

# Validated code → commit → push → pull on Main

## When to apply

After **substantive code or test changes** in this repo, when the user (or the task) treats work as **ready to land** and expects **Main** to match **Auxiliary** via git — not before regression is green.

**Do not** commit or push on failing regression. **Do not** force-push or rewrite published history unless the user explicitly orders it.

## Full regression (default)

**CI-equivalent** — matches [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```powershell
cd D:/AWBW   # or the active clone; use the workspace root
python -m pytest -q --tb=line
```

- Exit code **0** = regression gate **passed**.
- This is the repo’s **routine full unit regression**. It is **not** a full `desync_audit` / replay-zip sweep (see repo docs: those need local zips and are not default CI on every change).

If the user names a **stricter** gate (e.g. a specific `tools/` script, oracle pass), run **that** in addition to or instead of the default; state which gate passed in the commit context.

## After regression passes (Auxiliary)

1. **`git status`** — confirm there is something to commit; if the user only wanted validation, stop after pytest unless they asked to ship.
2. **Stage and commit** — message should state what changed and that pytest (or named gate) passed.
3. **`git push`** — push the current branch to `origin` (or the remote/branch the user uses; follow their setup).

## Sync Main (D:/awbw)

**Main** is `sshuser@192.168.0.160`, repo on disk **`D:/awbw`**, per `awbw-auxiliary-main-machines`. After a successful **push** from Auxiliary:

```powershell
ssh sshuser@192.168.0.160 "cd /d D:\awbw && git pull"
```

- Adjust if Main uses a different remote default (e.g. `git pull origin main`). Prefer **fast-forward**; if pull reports conflicts, **stop** and report — do not resolve on Main blindly without user input.
- If the remote shell is not `cmd`, use an equivalent: e.g. PowerShell:  
  `ssh sshuser@192.168.0.160 "powershell -NoProfile -Command \"Set-Location D:/awbw; git pull\""`

Confirm success (short `git log -1` on Main via SSH) when the user cares about proof.

## Order of operations (checklist)

1. Run **full regression** (pytest as above, or user-specified gate).
2. If **fail** → fix or report; **no** commit/push/pull.
3. If **pass** → commit (if there are changes) → push → **SSH** → `git pull` in `D:/awbw`.

## Related

- `awbw-auxiliary-main-machines` — Auxiliary vs Main, `Z:/`, `192.168.0.160`.
