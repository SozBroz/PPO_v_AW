"""Phase 6 diff aggregator — replaces lane B's broken inline script."""
import re
from collections import Counter

TAGS = [
    ("vs_pre_purge", "logs/phase6_diff_vs_pre_purge.log"),
    ("vs_post_purge", "logs/phase6_diff_vs_post_purge.log"),
    ("vs_post_phase5", "logs/phase6_diff_vs_post_phase5.log"),
]


def main() -> None:
    for tag, path in TAGS:
        text = open(path, encoding="utf-8", errors="replace").read()
        regs = len(re.findall(r"^  REGRESSION", text, re.M))
        fixed = len(re.findall(r"^  FIXED", text, re.M))
        drift = len(re.findall(r"^  CLASS_DRIFT|^  class_drift", text, re.M))
        attack_inv = len(re.findall(r"_apply_attack: target", text))
        print(f"{tag}: regressions={regs} fixed={fixed} class_drift={drift} attack_inv_msgs={attack_inv}")

    print()
    text = open("logs/phase6_diff_vs_post_phase5.log", encoding="utf-8", errors="replace").read()

    print("=== FIXED (post_phase5 -> post_phase6) — top 10 ===")
    fixed_msgs: Counter[str] = Counter()
    for m in re.finditer(r"^  FIXED gid=(\d+) -> (\w+)[^\n]*\n      msg: ([^\n]+)", text, re.M):
        msg = m.group(3)[:80]
        print(f"  {m.group(1)} -> {m.group(2)}: {msg}")
        fixed_msgs[msg] += 1

    print()
    print("=== REGRESSION (post_phase5 -> post_phase6) — top 10 ===")
    reg_msgs: Counter[str] = Counter()
    for m in re.finditer(r"^  REGRESSION gid=(\d+) -> (\w+)[^\n]*\n      msg: ([^\n]+)", text, re.M):
        msg = m.group(3)[:80]
        print(f"  {m.group(1)} -> {m.group(2)}: {msg}")
        reg_msgs[msg] += 1

    print()
    print("=== CLASS_DRIFT (post_phase5 -> post_phase6) — top 10 ===")
    drift_msgs: Counter[str] = Counter()
    for m in re.finditer(r"^  CLASS_DRIFT gid=(\d+) (\w+) -> (\w+)[^\n]*\n      msg: ([^\n]+)", text, re.M):
        msg = m.group(4)[:80]
        print(f"  {m.group(1)} {m.group(2)}->{m.group(3)}: {msg}")
        drift_msgs[msg] += 1

    print()
    print("=== Aggregated message shapes ===")
    print("FIXED shapes:")
    for msg, n in fixed_msgs.most_common(10):
        print(f"  {n:4d}  {msg}")
    print("REGRESSION shapes:")
    for msg, n in reg_msgs.most_common(10):
        print(f"  {n:4d}  {msg}")
    print("CLASS_DRIFT shapes:")
    for msg, n in drift_msgs.most_common(10):
        print(f"  {n:4d}  {msg}")


if __name__ == "__main__":
    main()
