#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists, dirname, realpath
from setuptools import setup, find_packages
import sys

author = u"bmlab developers"
# authors in alphabetical order
authors = [
    "Matthias Bär",
    "Paul Müller",
    "Raimund Schlüßler",
    "Timon Beck",
]
description = 'Library for Brillouin microscopy data analysis'
name = 'bmlab'
year = "2022"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version
except BaseException:
    version = "unknown"


setup(
    name=name,
    author=author,
    author_email='dev@craban.de',
    url='https://github.com/BrillouinMicroscopy/bmlab',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="GPL v3",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["h5py>=2.10.0",
                      "matplotlib",
                      "numpy>=1.17.0",
                      "packaging>=20.8",
                      "pillow",
                      "pytest",
                      "pytest_mock",
                      "scikit-image>=0.19.0",
                      "scipy",
                      "shapely>=1.8.2"
                      ],
    # not to be confused with definitions in pyproject.toml [build-system]
    python_requires=">=3.7",
    keywords=["Brillouin microscopy"],
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Visualization',
                 'Intended Audience :: Science/Research',
                 ],
    platforms=['ALL'],
)
