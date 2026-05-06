"""
Build script for Cython extensions used in AWBW RL training.

Usage:
    python setup_cython.py build_ext --inplace

On Windows, if .pyd files are locked (e.g. by running Python processes),
the --inplace copy will fail. This script works around that by building
to the build/ directory and manually copying files with lock handling.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import shutil

# Compiler optimization: maximum speed for release builds
if sys.platform == "win32":
    extra_compile_args = ["/O2"]  # MSVC: full optimization
else:
    extra_compile_args = ["-O3"]  # GCC/Clang: aggressive optimization

extensions = []
all_modules = [
    # Core engine modules
    ("engine._action_cython", "engine/_action_cython.pyx"),
    ("engine._occupancy_cython", "engine/_occupancy_cython.pyx"),
    ("engine._search_clone_cython", "engine/_search_clone_cython.pyx"),
    # RL modules
    ("rl._candidate_actions_cython", "rl/_candidate_actions_cython.pyx"),
    ("rl._tactical_beam_cython", "rl/_tactical_beam_cython.pyx"),
    ("rl._rhea_cython", "rl/_rhea_cython.pyx"),
    ("rl._rhea_fitness_cython", "rl/_rhea_fitness_cython.pyx"),
    ("rl._encoder_cython", "rl/_encoder_cython.pyx"),
]

for mod_name, pyx_path in all_modules:
    if os.path.exists(pyx_path):
        extensions.append(
            Extension(
                mod_name,
                [pyx_path],
                include_dirs=[np.get_include()],
                extra_compile_args=extra_compile_args,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
            )
        )
    else:
        print("Warning: {} not found, skipping...".format(pyx_path))


def get_compiler_directives():
    return {
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
        "nonecheck": False,
        "overflowcheck": False,
        "language_level": 3,
    }


# Custom inplace copy that handles locked .pyd files on Windows
def robust_inplace_copy(build_dir, src_dir):
    """Copy .pyd files from build dir to source dir, handling locked files on Windows."""
    copied = []
    for root, dirs, files in os.walk(build_dir):
        for fname in files:
            if fname.endswith('.pyd'):
                src = os.path.join(root, fname)
                # Compute destination relative to src_dir
                rel_path = os.path.relpath(root, build_dir)
                dst = os.path.join(src_dir, rel_path, fname)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    # Try normal copy first
                    shutil.copy2(src, dst)
                except (PermissionError, OSError):
                    # On Windows, rename the locked file, then copy
                    if os.path.exists(dst):
                        locked_name = dst + ".locked"
                        if os.path.exists(locked_name):
                            os.remove(locked_name)
                        os.rename(dst, locked_name)
                    shutil.copy2(src, dst)
                    print("  Copied {} (had to rename locked file)".format(fname))
                copied.append(dst)
    return copied


if __name__ == "__main__":
    inplace = "--inplace" in sys.argv
    if inplace:
        sys.argv.remove("--inplace")

    setup(
        name="awbw-cython",
        ext_modules=cythonize(extensions, compiler_directives=get_compiler_directives()),
    )

    if inplace:
        # Manually copy .pyd files to source directories
        src_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(src_dir, "build")
        plat_dir = None
        # Find the platform-specific build dir
        if os.path.exists(build_dir):
            for d in os.listdir(build_dir):
                if d.startswith("lib."):
                    plat_dir = os.path.join(build_dir, d)
                    break
        if plat_dir:
            print("\nCopying .pyd files to source directories...")
            copied = robust_inplace_copy(plat_dir, src_dir)
            for f in copied:
                print("  {}".format(f))
            print("Done.")
        else:
            print("\nNo build output found to copy.")

