"""
Notes on building extensions
----------------------------

Since we build extensions using f2py, we need numpy during the build phase.

We use

    setup(setup_requires=['numpy'],...)

which, if no numpy install is found, downloads smth like

    .eggs/numpy-1.17.2-py3.7-linux-x86_64.egg/

and used its f2py to compile extensions.

We build the extensions using src/Makefile (see doc/source/written/install.rst)
instead of setuptools.extension.Extension or
numpy.distutils.extension.Extension, so we sneak a call to "make" into the
setup.py build_py phase using build_extensions() and make_cmd_class() below.

``pip install`` seems to trigger ``setup.py build_py`` and ``setup.py
install``, so adding build_extensions() to build_py is sufficient. The dev
install ``pip install -e`` only triggers ``setup.py develop`` as it seems, so
we need to add that as well.
"""

import os
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst")) as fd:
    long_description = fd.read()


def build_extensions():
    """
    Instead of fighting with numpy.distutils (see
    https://stackoverflow.com/a/41896134), we use our Makefile which we know
    works well (OpenMP etc, see "make help") and copy the *.so files using the
    package_data trick below. Makefile will copy the *.so files to pwtools/, so
    we just need to tell setuptools that these are "data files" that we wish to
    copy when installing, along with all *.py files.
    """
    subprocess.run(
        r"cd src/_ext_src; make clean; make ${PWTOOLS_EXT_MAKE_TARGET:-}",
        shell=True,
        check=True,
    )


def make_cmd_class(base):
    """
    Call build_extensions() in "python setup.py <base>" prior to anything else.
    base = build_py, install, develop, ..., i.e. a setup.py command. Use in
    setup(cmdclass={'<base>': make_cmd_class(<base>)}, ...).

    Parameters
    ----------
    base : setuptools.command.<base>.<base> instance

    Notes
    -----
    https://stackoverflow.com/a/36902139
    """

    class CmdClass(base):
        def run(self):
            build_extensions()
            super().run()

    return CmdClass


setup(
    name="pwtools",
    version="1.2.3",
    description="pre- and postprocessing of atomistic calculations",
    long_description=long_description,
    url="https://github.com/elcorto/pwtools",
    author="Steve Schmerler",
    author_email="git@elcorto.com",
    license="BSD 3-Clause",
    keywords="ase scipy atoms simulation database postprocessing qha",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude="pwtools/src"),
    install_requires=open("requirements.txt").read().splitlines(),
    setup_requires=["numpy"],
    python_requires=">=3",
    package_data={"pwtools": ["*.so"]},
    cmdclass={
        "build_py": make_cmd_class(build_py),
        "develop": make_cmd_class(develop),
    },
)
