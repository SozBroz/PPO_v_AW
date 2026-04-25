"""One-off: cifs-utils + mount //192.168.0.160/AWBW -> /mnt/awbw. Env: KERAS_SUDO_PASSWORD. Optional: CIFS_USER, CIFS_PASS (Main)."""
import base64
import os
import shlex
import sys

import paramiko

HOST = "192.168.0.122"
SHARE = "//192.168.0.160/AWBW"
MOUNT = "/mnt/awbw"
K_SUDO = os.environ.get("KERAS_SUDO_PASSWORD", "")
CIFS_USER = os.environ.get("CIFS_USER", "")
CIFS_PASS = os.environ.get("CIFS_PASS", "")


def _ensure_fstab(
    ssh,
    pw: str,
    *,
    guest: bool,
    out_log: list,
) -> None:
    line = (
        "//192.168.0.160/AWBW /mnt/awbw cifs guest,uid=1000,gid=1000,iocharset=utf8,"
        "file_mode=0775,dir_mode=0775,vers=3.0,_netdev 0 0"
        if guest
        else "//192.168.0.160/AWBW /mnt/awbw cifs credentials=/root/.creds-awbw-smb,uid=1000,"
        "gid=1000,iocharset=utf8,file_mode=0775,dir_mode=0775,vers=3.0,_netdev 0 0"
    )
    code, o = run_sudo(
        ssh,
        pw,
        f"grep -qF '//192.168.0.160/AWBW' /etc/fstab || echo {shlex.quote(line)} >> /etc/fstab",
    )
    out_log.append(f"--- fstab\n{o}")
    if code != 0:
        out_log.append(f"fstab append exit {code}")


def run_sudo(ssh, pw: str, cmd: str) -> tuple[int, str]:
    """Run a command as root: echo pw | sudo -S ..."""
    s = ssh.get_transport().open_session()
    s.get_pty()
    inner = f"set -e; {cmd}"
    full = f"echo {shlex.quote(pw)} | sudo -S -p '' bash -c {shlex.quote(inner)}"
    s.exec_command(full)
    out, err = s.makefile("r", 4096), s.makefile_stderr("r", 4096)
    b_out, b_err = b"", b""
    import time

    while not s.exit_status_ready():
        b_out += out.read(65536) or b""
        b_err += err.read(65536) or b""
        time.sleep(0.02)
    b_out += out.read() or b""
    b_err += err.read() or b""
    st = s.recv_exit_status() if s.exit_status_ready() else -1
    s.close()
    return st, b_out.decode("utf-8", "replace") + b_err.decode("utf-8", "replace")


def main() -> int:
    if not K_SUDO:
        print("Set KERAS_SUDO_PASSWORD for keras sudo on the Linux host.", file=sys.stderr)
        return 1
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="keras", look_for_keys=True, allow_agent=True, timeout=45)

    out_log = []
    for step in [
        "apt-get update -qq",
        "apt-get install -y cifs-utils",
        f"umount {MOUNT} 2>/dev/null || true",
        f"mkdir -p {MOUNT} && chmod 755 {MOUNT}",
    ]:
        code, out = run_sudo(c, K_SUDO, step)
        out_log.append(f"--- {step}\n{out}")
        if code != 0 and "umount" not in step and "2>/dev/null" not in step:
            c.close()
            print("\n".join(out_log), file=sys.stderr)
            print("FAILED", code, file=sys.stderr)
            return code

    base_opts = "uid=1000,gid=1000,iocharset=utf8,file_mode=0775,dir_mode=0775,vers=3.0"
    # 1) guest
    code, o = run_sudo(
        c,
        K_SUDO,
        f"mount -t cifs {shlex.quote(SHARE)} {MOUNT} -o guest,{base_opts} 2>&1",
    )
    out_log.append(f"--- guest mount\n{o}")
    if code == 0:
        code2, check = run_sudo(
            c,
            K_SUDO,
            f"df -T {MOUNT} 2>&1; ls {MOUNT} 2>&1 | head -5",
        )
        out_log.append(f"--- verify\n{check}")
        _ensure_fstab(c, K_SUDO, cred_path=None, guest=True, out_log=out_log)
        c.close()
        print("\n".join(out_log))
        print("Mount OK (guest).", file=sys.stderr)
        return 0

    if not CIFS_USER or not CIFS_PASS:
        c.close()
        print("\n".join(out_log), file=sys.stderr)
        print(
            "Guest failed. Re-run with CIFS_USER and CIFS_PASS for Main (same as Windows use for \\\\192.168.0.160\\AWBW).",
            file=sys.stderr,
        )
        return 1

    cred_path = "/root/.creds-awbw-smb"
    cred_content = f"username={CIFS_USER}\npassword={CIFS_PASS}\n"
    b64 = base64.b64encode(cred_content.encode("utf-8")).decode("ascii")
    code, o = run_sudo(
        c,
        K_SUDO,
        f"echo {shlex.quote(b64)} | base64 -d > {cred_path} && chmod 600 {cred_path} && "
        f"mount -t cifs {shlex.quote(SHARE)} {MOUNT} -o credentials={cred_path},{base_opts} 2>&1",
    )
    out_log.append(f"--- user mount\n{o}")
    code2, check = run_sudo(
        c, K_SUDO, f"df -T {MOUNT} 2>&1; ls {MOUNT} 2>&1 | head -5"
    )
    out_log.append(f"--- verify\n{check}")
    if code != 0:
        c.close()
        print("\n".join(out_log), file=sys.stderr)
        return code
    _ensure_fstab(c, K_SUDO, guest=False, out_log=out_log)
    c.close()
    print("\n".join(out_log))
    print("Mount OK (credentials).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
