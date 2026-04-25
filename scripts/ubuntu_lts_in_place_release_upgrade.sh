#!/usr/bin/env bash
# In-place LTS release upgrade (e.g. 18.04 -> 20.04 -> 22.04, one hop per run).
# Run *on the Ubuntu host* as root:   sudo bash scripts/ubuntu_lts_in_place_release_upgrade.sh
# Over SSH, use a stable session (screen/tmux) — upgrades can take 30–90+ minutes and can drop the connection.
# If the script exits 10, reboot, then re-run the script until you reach the target LTS; repeat until `do-release-upgrade` says no new release.
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none
export NEEDRESTART_MODE=a 2>/dev/null || true

echo "[0] Source: /etc/os-release"
grep -E '^(NAME|VERSION|VERSION_CODENAME)=' /etc/os-release || true

echo "[1] apt update + full-upgrade (required before do-release-upgrade)"
apt-get -y update
# Hold third-party conffile noise; allow upgrade to proceed
apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" full-upgrade
apt-get -y autoremove

if [ -f /var/run/reboot-required ]; then
  echo
  echo "[REBOOT REQUIRED] Install /var/run/reboot-required* before release upgrade. Example:"
  echo "    sudo reboot"
  echo "Then SSH back in and re-run: sudo bash $0"
  exit 10
fi

echo
echo "[2] Starting noninteractive release upgrader (next LTS). This can take a long time."
# third-party repos (Chrome, etc.) are common on dev machines — do not always comment them out
if ! do-release-upgrade -f DistUpgradeViewNonInteractive -m server --allow-third-party; then
  echo "[warn] do-release-upgrade exited non-zero. Check /var/log/dist-upgrade/ and $HOME/*upgrade*" >&2
  exit 1
fi

echo
echo "[3] If the upgrader finished, reboot when prompted or if /var/run/reboot-required exists."
if [ -f /var/run/reboot-required ]; then
  echo "    sudo reboot"
fi
echo "After boot, re-run this script to step to the *next* LTS until you reach 22.04/24.04 (or your target)."
echo "Re-create Python venvs (uv) after several hops — glibc and paths change."
exit 0
