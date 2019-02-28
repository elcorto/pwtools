# publish on pypi
# ---------------
#   $ python3 setup.py sdist
#   $ twine upload dist/<this-package>-x.y.z.tar.gz

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as fd:
    long_description = fd.read()

# Instead of fighting with numpy.distutils (see
# https://stackoverflow.com/a/41896134), we use our Makefile which we know
# works well (OpenMP etc, see "make help") and copy the *.so files using the
# package_data trick below. Makefile will copy the *.so files to pwtools/, so
# we just need to tell setuptools that these are "data files" that we wish to
# copy when installing, along with all *.py files.
from subprocess import run
run("cd src; make", shell=True, check=True)

setup(
    name='pwtools',
    version='0.9.0',
    description='pre- and postprocessing of atomic calculations',
    long_description=long_description,
    url='https://github.com/elcorto/pwtools',
    author='Steve Schmerler',
    author_email='git@elcorto.com',
    license='BSD 3-Clause',
    keywords='ase scipy atoms simulation database postprocessing',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3',
    package_data={'pwtools': ['*.so']},
)
