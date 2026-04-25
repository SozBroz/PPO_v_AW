"""
Setup script for compiling the Cython encoder module.
"""

import os
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
encoder_extension = Extension(
    name='rl._encoder_cython',
    sources=['_encoder_cython.pyx'],
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args=['/O2', '/openmp'] if os.name == 'nt' else ['-O3', '-fopenmp'],
    extra_link_args=['/openmp'] if os.name == 'nt' else ['-fopenmp']
)

# Setup configuration
setup(
    name='awbw_encoder_cython',
    ext_modules=cythonize(
        [encoder_extension],
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)