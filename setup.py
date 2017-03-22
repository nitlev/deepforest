#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import re
from os.path import dirname
from os.path import join

from setuptools import setup, find_packages


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ).read()


setup(
    name='deepforest',
    url="https://github.com/nitlev/deepforest",
    author='Veltin DUPONT',
    author_email='veltind@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data scientists',
        'Topic :: Deep Learning :: Deep Forests',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python 2.7 :: Python 3.3+',
    ],
    keywords='random forest :: deep learning',
    version='0.1.1',
    license='MIT',
    description='This is a simple implementation of the deep forest method',
    long_description=re.compile('^.. start-badges.*^.. end-badges',
                                re.M | re.S).sub('', read('README.md')),
    packages=["deepforest"],
    package_dir={"deepforest": "deepforest"},
    include_package_data=True,
    zip_safe=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
