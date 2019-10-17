#!/usr/bin/env python
# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import glob
import os

from setuptools import find_packages
from setuptools import setup


requirements = [
    "torchvision",
    "ninja",
    "yacs",
    "cython",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "scikit-image"
]

setup(
    name="charnet",
    version="0.0.1",
    author="Malong Technologies",
    url="https://github.com/MalongTech/research-charnet",
    description="Convolutional Character Networks",
    packages=find_packages(),
    include_package_data=True,
)
