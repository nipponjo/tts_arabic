#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py
from pathlib import Path

from setuptools import setup, find_packages


NAME = 'tts_arabic'
DESCRIPTION = 'Arabic TTS models'
URL = 'https://github.com/nipponjo/tts_arabic'
EMAIL = 'nipponjo.git@gmail.com'
AUTHOR = 'nipponjo'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.0.1'

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,

    packages=find_packages(),
    install_requires=[
        'numpy',
        "onnxruntime-gpu; sys_platform != 'darwin'",  # for Windows, Linux
        "onnxruntime; sys_platform == 'darwin'",  # for Mac
        'gdown>=5.1.0'
    ],
    include_package_data=True,
    data_files=[
        ('license', ['tts_arabic/ThirdPartyLicenses',])
    ],


    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Natural Language :: Arabic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Environment :: GPU :: NVIDIA CUDA',
    ])