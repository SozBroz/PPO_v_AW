"""Add Cython auto-recompile logic to a Python script."""

import sys
from pathlib import Path

def add_cython_recompile(script_path: str) -> None:
    """Add _maybe_recompile_cython() and call it before rl.* imports."""
    p = Path(script_path)
    content = p.read_text(encoding="utf-8")

    # The function definition to add (after imports, before rl imports)
    cython_func = '''
# ---------------------------------------------------------------------------
# Cython auto-recompile: if any .pyx is newer than its compiled .pyd/.so,
# rebuild the Cython extensions before importing rl.* modules.
# ---------------------------------------------------------------------------
def _maybe_recompile_cython() -> None:
    """Rebuild Cython extensions if any .pyx source is newer than the binary.

    Checks for .pyd (Windows) or .so (Linux) files that correspond to
    .pyx sources under the project.  Runs ``python setup_cython.py build_ext --inplace``
    only when a rebuild is needed, then re-imports the updated modules.
    """
    import subprocess
    from pathlib import Path as _Path
    import sys as _sys

    project_root = _Path(__file__).resolve().parents[1]
    setup_script = project_root / "setup_cython.py"
    if not setup_script.exists():
        return

    # Collect .pyx files
    pyx_dirs = [project_root / "rl", project_root / "engine"]
    pyx_files = []
    for d in pyx_dirs:
        if d.exists():
            pyx_files.extend(d.glob("*.pyx"))

    if not pyx_files:
        return

    # Determine compiled extension suffix
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

    # Find where to insert: after "from typing import Any" and before "import numpy"
    # We need to add the function and its call
    lines = content.split('\\n')
    
    # Find the line index of "import numpy as np" (the first one, before rl imports)
    numpy_idx = -1
    for i, line in enumerate(lines):
        if 'import numpy as np' in line and 'rl.encoder' not in '\\n'.join(lines[:i]):
            numpy_idx = i
            break
    
    if numpy_idx == -1:
        print(f"Error: could not find 'import numpy as np' in {script_path}")
        sys.exit(1)

    # Insert the cython function before the numpy import
    new_lines = lines[:numpy_idx] + cython_func.strip().split('\\n') + [''] + lines[numpy_idx:]
    
    p.write_text('\\n'.join(new_lines), encoding="utf-8")
    print(f"Added Cython recompile logic to {script_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_cython_recompile.py <script1> [script2] ...")
        sys.exit(1)
    for script in sys.argv[1:]:
        add_cython_recompile(script)
