---
name: awbw-auxiliary-main-machines
description: >-
  AWBW fleet: **Main** — sshuser@192.168.0.160, D:/awbw. **Aux** machines are
  flat (no tiers). The same Main **awbw** tree is **Samba-mounted** on each
  aux: **Windows** → **`Z:`** (`.122` is now Windows + `C:\\Users\\sshuser\\AWBW`);
  **Linux** non-`.122` → **`/mnt/awbw`**. Set `AWBW_SHARED_ROOT` to
  match. Local dev clone is often C:/Users/phili/AWBW. Use for fleet, Main, Z:,
  /mnt/awbw, 192.168.0.160, 192.168.0.122, D:/awbw, pool IDs, or multi-host work.
---

# Fleet: Main + auxiliary machines (flat aux model)

## Roles

| Role | What it is |
|------|------------|
| **Main** | One canonical host: `sshuser@192.168.0.160` — Windows. Repo on disk: **`D:/awbw`**. Use **SSH** for shell, git, and anything that must run *on* Main. |
| **Auxiliary** | **Any number** of machines — **no tiers** among them. Each may have a local git clone, pool ID, and SSH identity. The **current session** is often a **Windows** aux with this repo at **`C:/Users/phili/AWBW`**. **`192.168.0.122`** is a **Windows** aux (`C:\Users\sshuser\AWBW` + **`Z:\`**). **Other Linux** auxes use **`/mnt/awbw`**, not a drive letter. |

## Main’s `awbw` on aux: one Samba tree, two mount styles

**Main** publishes **`D:/awbw`** over **Samba (SMB)**. On each auxiliary, **mount that share** so training and pool layout see the **same** `checkpoints/`, `data/`, etc. as Main — it is not a second independent copy to invent.

| Platform | Mount point | `AWBW_SHARED_ROOT` |
|----------|-------------|--------------------|
| **Windows** aux | **`Z:\`** (drive mapped to the share) | Default in `rl/fleet_env.py` is `Z:\` if unset. |
| **Linux** aux (this fleet) | **`/mnt/awbw`** | Set **`/mnt/awbw`** (or export `AWBW_SHARED_ROOT=/mnt/awbw`). **No `Z:`** on these hosts. |

- **Fleet I/O** under the shared tree — use the mount for **that** OS: `Z:\...` on Windows, **`/mnt/awbw/...`** on Linux.
- **Run something on Main** — **`ssh sshuser@192.168.0.160`** and use `D:/awbw` in that shell.

## SSH quick reference

- **Main**: `sshuser@192.168.0.160` — passwordless when keys are configured.
- **Other aux** (example): `sshuser@192.168.0.122` (Windows, repo + venv: **`C:\Users\sshuser\AWBW`**, shared root **`Z:\`** when mapped). **Name the host** when several apply.

When the user says **"both machines"**, they often mean **this dev box and Main**. **`.122` (Windows)** — **SSH** + **`C:\Users\sshuser\...`**; map **`Z:\`** in an **interactive** session (SMB from the OpenSSH service session often returns **1312**). Legacy Linux 122 was **`/mnt/awbw`**; that path no longer applies to this host.

**Main** stays **`192.168.0.160`** unless they name another main.

## Windows aux (`sshuser@192.168.0.122` — reinstalled; was Linux 18.04 + `/mnt/awbw`)

- **Repo + venv on disk** (not only SMB): **`C:\Users\sshuser\AWBW`**, venv **`.venv`**, base Python **3.12** embeddable at **`C:\Users\sshuser\py312`** (full MSI failed unattended; `virtualenv` created `.venv`). **`requirements.txt`** was copied in when GitHub `main` zip lacked it.
- **MSVC++ runtime**: machine had **no** `vcruntime140_1` / `msvcp140` in `System32`. Until an admin runs **`vc_redist.x64.exe`**, a **sidecar copy of those four DLLs** may live next to `torch\lib\*.dll` (workaround; proper fix is a **system** install of [VC++ 2015–2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)).
- **Z:** — map `\\192.168.0.160\AWBW` → **`Z:\`** in **Console or RDP** (see **`net use` error 1312** from SSH). Then set fleet env: see **`%USERPROFILE%\windows_aux_train_env.cmd`**, or set **`AWBW_SHARED_ROOT=Z:\`**, **`AWBW_MACHINE_ROLE=auxiliary`**, **`AWBW_MACHINE_ID=...`**.
- **Smoke**: `C:\Users\sshuser\AWBW\.venv\Scripts\python.exe C:\Users\sshuser\AWBW\train.py --watch-only --device cpu --n-envs 1` (sync repo if you need parity with this workspace / Main).
- **Copy this workspace** to 122: `git` not installed; use **`git` + clone**, pull from `Z:\` after mapping, or `scp`/`rsync` from a dev box.

## Agent behavior

1. **Edits** default to **this workspace** `C:/Users/phili/AWBW` unless the task targets Main, a named aux, read-only **shared** tree, or SSH.
2. For Main’s awbw **via the network share**: on **this Windows session** use **`Z:/`**. On **Linux** fleet hosts, use **`/mnt/awbw/`** and set / assume **`AWBW_SHARED_ROOT=/mnt/awbw`** for training and pool paths.
3. **Do not** assume every aux uses `Z:`; **map OS → mount** before building paths.
4. **No hierarchy** among aux machines; **disambiguate host** when needed.

## Related

- `awbw-regression-then-ship-main` — pytest green → commit, push, `git pull` on Main `D:/awbw`.
- `awbw-pool-latest-vs-shared-latest` — shared `checkpoints/...` vs pool `latest` (same path logic under the chosen `AWBW_SHARED_ROOT`).
- `docs/play_ui.md` — `AWBW_MACHINE_ID`, `AWBW_SHARED_ROOT`, pool behavior.
- `rl/fleet_env.py` — `load_shared_root_for_role`, shared-root validation on auxiliary.
