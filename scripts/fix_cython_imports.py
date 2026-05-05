"""Add Cython auto-recompile to scripts that use rl.* imports.

Properly inserts the Cython check function and its call before the first
`import numpy` line (before any rl.* imports).
"""

import sys
from pathlib import Path

CYTHON_BLOCK = '''
# ---------------------------------------------------------------------------
# Cython auto-recompile: if any .pyx is newer than its compiled .pyd/.so,
# rebuild the Cython extensions before importing rl.* modules.
# ---------------------------------------------------------------------------
def _maybe_recompile_cython() -> None:
    """Rebuild Cython extensions if any .pyx source is newer than the binary."""
    import subprocess
    from pathlib import Path as _Path
    import sys as _sys

    project_root = _Path(__file__).resolve().parents[1]
    setup_script = project_root / "setup_cython.py"
    if not setup_script.exists():
        return

    pyx_dirs = [project_root / "rl", project_root / "engine"]
    pyx_files = []
    for d in pyx_dirs:
        if d.exists():
            pyx_files.extend(d.glob("*.pyx"))

    if not pyx_files:
        return

    if _sys.platform.startswith("win"):
        ext_suffix = ".pyd"
    else:
        ext_suffix = ".so"

    needs_rebuild = False
    for pyx in pyx_files:
        compiled = pyx.with_suffix(ext_suffix)
        if not compiled.exists():
            needs_rebuild = True
            break
        if pyx.stat().st_mtime > compiled.stat().st_mtime:
            needs_rebuild = True
            break

    if needs_rebuild:
        print("Cython sources changed; rebuilding extensions...", flush=True)
        try:
            result = subprocess.run(
                [_sys.executable, str(setup_script), "build_ext", "--inplace"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(
                    f"Cython rebuild failed (rc={result.returncode}):\\n"
                    f"{result.stdout}\\n{result.stderr}",
                    file=_sys.stderr,
                    flush=True,
                )
            else:
                print("Cython rebuild complete.", flush=True)
        except Exception as exc:
            print(f"Cython rebuild error: {exc}", file=_sys.stderr, flush=True)


# Run the check before importing rl.* (which may import the .pyd files)
_maybe_recompile_cython()

'''


def add_cython_recompile(script_path: Path) -> None:
    """Add Cython recompile block before 'import numpy' line."""
    content = script_path.read_text(encoding="utf-8")
    
    # Find position of first "import numpy" that's at module level
    # (not indented, and before any rl imports)
    lines = content.split('\n')
    
    # Find the line with "import numpy" (the first one, which is before rl imports)
    target_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "import numpy as np" or stripped == "import numpy":
            # Check that this is before any "from rl." imports
            before_this = '\n'.join(lines[:i])
            if 'from rl.' not in before_this and 'import rl.' not in before_this:
                target_idx = i
                break
    
    if target_idx == -1:
        print(f"  ERROR: Could not find 'import numpy' in {script_path.name}")
        return
    
    # Insert the Cython block before this line
    new_lines = lines[:target_idx] + CYTHON_BLOCK.strip().split('\n') + [''] + lines[target_idx:]
    
    script_path.write_text('\n'.join(new_lines), encoding="utf-8")
    print(f"  Added Cython recompile to {script_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_cython_imports.py <script1> [script2] ...")
        sys.exit(1)
    
    for script_str in sys.argv[1:]:
        script = Path(script_str)
        if not script.exists():
            print(f"  ERROR: {script} not found")
            continue
        add_cython_recompile(script)
