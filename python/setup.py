import os
import codecs
from shutil import copy2

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


# Not the best solution, but I guess to keep things in one place for R and
# python this is sufficient for now.
if not os.path.exists(os.path.join(here, 'README.md')):
    copy2(
        os.path.join(here, '../README.md'),
        os.path.join(here, 'README.md'),
    )

if not os.path.exists(os.path.join(here, 'hamstrpy/hamstr.stan')):
    copy2(
        os.path.join(here, '../inst/stan/hamstr.stan'),
        os.path.join(here, 'hamstrpy'),
    )

with open('README.md') as f:
    long_description = f.read()

setup(
    name='hamstrpy',
    version=get_version('hamstrpy/__init__.py'),
    description=('Hierarchical age-depth modeling using stan'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='A. Dolman',
    author_email='andrew.dolman@awi.de',
    packages=find_packages(),
    install_requires=[
        'statsmodels',
        'cmdstanpy',
        'numpy',
        'scipy',
        'matplotlib',
        'appdirs',
        'arviz',
    ],
    extras_require={
        'tests': [
            'pyreadr',
            'rpy2',
        ],
    },
    package_data={
        'hamstrpy': ['hamstr.stan'],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ],
)
