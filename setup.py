#!/usr/bin/env python3

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys

with open("README.md", "r") as f:
    long_description = f.read()

if 'linux' in sys.platform or 'darwin' in sys.platform:
    # gcc and clang (linux and macOS)
    compile_flags = ["-std=c++17", "-O3", "-march=native", "-Wno-#warnings"]
else:
    # msvc
    compile_flags = ["/std:c++17", "/Ox", "/arch:AVX2"]

ext_modules = [
    Extension("pyTTP.localsearch", ["pyTTP/localsearch.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=compile_flags),
    Extension("pyTTP.neighborhoods", ["pyTTP/neighborhoods.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=compile_flags)
]

# Use cythonize on the extension object.
setup(ext_modules=cythonize(ext_modules, language_level="3"),
      packages=["pyTTP"],
      name="pyTTP",
      version='0.0.1',
      author="Jonathan Helgert",
      url="https://github.com/jhelgert/pyTTP",
      description="A simple package to solve the Traveling Tournament Problem",
      long_description=long_description,
      python_requires='>=3.6',
      classifiers=[
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: MIT License",
    "Topic:: Scientific/Engineering:: Mathematics"],
    setup_requires=["numpy", "cython", "mip"],
    install_requires=["numpy", "cython", "mip"],
    zip_safe=False)
