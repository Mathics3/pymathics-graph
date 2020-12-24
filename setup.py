#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import platform
import os
import os.path as osp
from setuptools import setup, find_namespace_packages

# Ensure user has the correct Python version
if sys.version_info < (3, 6):
    print("Mathics support Python 3.6 and above; you have %d.%d" % sys.version_info[:2])
    sys.exit(-1)

def get_srcdir():
    filename = osp.normcase(osp.dirname(osp.abspath(__file__)))
    return osp.realpath(filename)


def read(*rnames):
    return open(osp.join(get_srcdir(), *rnames)).read()


# stores __version__ in the current namespace
exec(compile(open("pymathics/graph/version.py").read(), "version.py", "exec"))
long_description = read("README.rst") + "\n"

is_PyPy = platform.python_implementation() == "PyPy"

setup(
    name="pymathics-graph",
    version=__version__,
    packages=find_namespace_packages(include=["pymathics.*"]),
    install_requires=["Mathics3>=1.1.1", "networkx", "pydot", "matplotlib"],
    # don't pack Mathics in egg because of media files, etc.
    zip_safe=False,
    maintainer="Mathics Group",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    # metadata for upload to PyPI
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Interpreters",
    ],
    # TODO: could also include long_description, download_url,
)
