[![image](https://zenodo.org/badge/51149109.svg)](https://zenodo.org/badge/latestdoi/51149109)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/elcorto/pwtools/ci.yml?label=tests)

# About

`pwtools` is a Python package for pre- and postprocessing of atomistic
calculations, mostly targeted to [Quantum
Espresso](http://www.quantum-espresso.org), [CPMD](http://www.cpmd.org),
[CP2K](http://cp2k.org) and [LAMMPS](http://lammps.org). It is almost,
but not quite, entirely unlike [ASE](https://wiki.fysik.dtu.dk/ase),
with some tools extending
[numpy](http://www.numpy.org)/[scipy](http://www.scipy.org). It has a
set of powerful parsers and data types for storing calculation data. See
the [feature
overview](http://elcorto.github.io/pwtools/written/features.html) for
more.

The [dcd
code](https://github.com/elcorto/pwtools/blob/master/src/pwtools/dcd.py)
is now part of [ASE](https://wiki.fysik.dtu.dk/ase)'s [dcd reader for
CP2K files](https://gitlab.com/ase/ase/blob/master/ase/io/cp2k.py).
[Thanks](https://gitlab.com/ase/ase/merge_requests/1109)!

# Documentation

Have a look at [the docs](http://elcorto.github.io/pwtools). Quick start
instructions can be found in [the
tutorial](http://elcorto.github.io/pwtools/written/tutorial.html). Many
examples, besides the ones in the doc strings are in [the
tests](https://github.com/elcorto/pwtools/tree/master/test).

# Install

See the [install
docs](http://elcorto.github.io/pwtools/written/install.html).
