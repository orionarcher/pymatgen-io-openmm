# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from setuptools import setup, find_namespace_packages

import os

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SETUP_PTH, "README.rst")) as f:
    desc = f.read()


setup(
    name="pymatgen-io-openmm",
    packages=find_namespace_packages(include=["pymatgen.io.*"]),
    version="0.0.1",
    install_requires=["pymatgen>=2022.3.22"],
    extras_require={},
    package_data={},
    author="orion cohen",
    author_email="orion@lbl.gov",
    maintainer="orion cohen",
    url="https://github.com/orioncohen/pymatgen-io-openmm",
    license="BSD",
    description="A set of tools for setting up OpenMM simulations of battery systems.",
    long_description=desc,
    keywords=["pymatgen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
