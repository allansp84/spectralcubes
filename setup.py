# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(
    name='antispoofing.spectralcubes',
    version=open('version.txt').read().rstrip(),
    url='',
    license='New BSD Licence',
    author='Pinto et al.',
    author_email='allansp84@gmail.com',
    description='Face Spoofing Detection Through Visual Codebooks of Spectral Temporal Cubes',
    long_description=open('README.md').read(),

    packages=find_packages(exclude=['tests']),

    setup_requires=['numpy==1.8.1'],
    build_requires=['numpy==1.8.1'],

    install_requires=open('requirements.txt').read().splitlines()[::-1],

    entry_points={
        'console_scripts': [
            'spectralcubesantispoofing.py = antispoofing.spectralcubes.scripts.spectralcubesantispoofing:main',
        ],
    },
)
