#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skeleton-methods",
    version="0.0.4",
    author="Zeyu Wei, Yikun Zhang",
    author_email="zwei5@uw.edu, yikunzhang@foxmail.com",
    description="Skeleton-Based Methods for Clustering and Regression on Underlying Manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyk8/skeleton-methods",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["skeleton_methods"]),
    install_requires=["numpy", "scipy", "scikit-learn", "igraph"],
    python_requires=">=3.6",
)
