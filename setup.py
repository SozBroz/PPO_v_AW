from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Cython extensions with numpy headers
extensions = cythonize([
    Extension("engine._action_cython", ["engine/_action_cython.pyx"], include_dirs=[np.get_include()]),
    Extension("engine._occupancy_cython", ["engine/_occupancy_cython.pyx"], include_dirs=[np.get_include()]),
    Extension("rl._encoder_cython", ["rl/_encoder_cython.pyx"], include_dirs=[np.get_include()])
], compiler_directives={
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
})

setup(
    name="awbw_engine",
    ext_modules=extensions,
    zip_safe=False,
)