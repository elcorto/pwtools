Installation
============

To build the extension modules and install all Python dependencies via pip in
one go, use::

    $ pip3 install .

or the setuptools "development install" (no copy of files)::

    $ pip3 install -e .

If all dependencies are installed, e.g. by a package manager such as ``apt``,
then use a virtual environment that uses the system site-packages (here we use
virtualenvwrapper_)::

    $ mkvirtualenv --system-site-packages -p /usr/bin/python3 pwtools
    (pwtools) $ pip3 install -e .

Alternatively, you may also simply build the extensions and set ``PYTHONPATH``::

    $ cd src && make && cd ..
    $ [ -n "$PYTHONPATH" ] && pp=$(pwd):$PYTHONPATH || pp=$(pwd)
    $ export PYTHONPATH=$pp

Dependencies
------------
See ``requirements.txt`` for packages installable by pip. On a Debian-ish system,
you may install::

    # apt
    python3-numpy
    python3-scipy
    python3-nose
    python3-dev     # for compiling extensions
    python3-h5py
    python3-matplotlib
    python3-ase
    python3-numpydoc
    python3-sphinx
    gfortran        # or ifort, see src/Makefile
    liblapack-dev

    # pip
    PyCifRW
    pyspglib  # renamed to spglib after v1.8.x, not supported yet

You can use ``test/check_dependencies.py`` to find out what your system has
installed.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

* VMD_ (``examples/rpdf/``, :func:`~pwtools.crys.call_vmd_measure_gofr`,
  :func:`~pwtools.visualize.view_vmd_axsf`,
  :func:`~pwtools.visualize.view_vmd_xyz`), must register before download

* The ``fourier.x`` tool from the CPMD_ contrib sources (for
  ``examples/``). Need to register before download.

* eos (for :mod:`~pwtools.eos`): The tool "eos" from the Elk_ code must
  be on your path. Note that the executable is assumed to be named ``eos.x``
  instead of the default name "eos". See :class:`~pwtools.eos.ElkEOSFit` for
  usage. Can be installed directly from Elk or pwextern-free_.

The pwextern-free_ package contains add-on tools which we don't want / can ship
directly with pwtools, such as eos.

.. note:: pwextern-free also contains very old versions of PyCifRW and
   pyspglib, don't use those, use pip versions! Also, don't use the
   install.sh script provided there. If needed, only compile eos.x and place it
   somewhere in PATH.

.. note:: :class:`~pwtools.eos.ElkEOSFit` is deprecated now, and so you don't
   really need the Elk code's eos tool anymore. It was used for generating
   refernce EOS fit data once, which is stored in test/files/, but there is no
   use of eos.x in pwtools anymore.

Running tests
-------------

See test/README. Actually, all of these are good examples, too!

Python versions
---------------

Only Python3, tested: Python 3.6

The package was developed mostly with Python 2.5..2.7 and ported using 2to3.

Compiling Fortran extensions and OpenMP notes
---------------------------------------------

Use ``src/Makefile``:

.. code-block:: shell

    $ make help
    make gfortran            # gfortran, default
    make gfortran-omp        # gfortran + OpenMP
    make gfortran-mkl        # gfortran, Intel MKL lapack, set MKL_LIB
    make ifort               # ifort
    make ifort-omp           # ifort + OpenMP
    make ifort-mkl           # ifort, Intel MKL lapack, set MKL_LIB

Generates ``*.so`` and ``*.pyf`` (f2py interface) files.

You need:

* numpy for f2py
* a Fortran compiler
* Python headers (Debian/Ubuntu: python-dev)
* Lapack (Debian: liblapack3)

The module is compiled with f2py (currently part of numpy, tested with numpy
1.1.0 .. 1.13.x).

Compiler / f2py
^^^^^^^^^^^^^^^
Instead of letting numpy.distutils pick a compiler + special flags, which is
not trivial and therefore almost never works, it is much easier to simply
define the compiler to use + architecture-specific flags. See F90 and ARCH in
the Makefile.

Also, numpy.distutils has default -03 for fcompiler. ``--f90flags="-02"`` does NOT
override this. We get ``-O3 -O2`` and a compiler warning. We have to use f2py's
``--opt=`` flag.

On some systems (Debian), you may have:

.. code-block:: shell

  /usr/bin/f2py -> f2py2.6
  /usr/bin/f2py2.5
  /usr/bin/f2py2.6

and such. But usually ``F2PY=f2py`` is fine.

OpenMP
^^^^^^
We managed to speed up the calculations by sprinkling some OpenMP
pragmas in ``*.f90``. Use ``make ifort-omp`` or ``make gfortran-omp`` in that
case.

If all went well, ``_flib.so`` should be linked to ``libgomp`` (or ``libiomp``
for ``ifort``). Check with:

.. code-block:: shell

    $ ldd _flib.so

Setting the number of threads:

.. code-block:: shell

    $ export OMP_NUM_THREADS=2
    $ python -c "import numpy as np; from pwtools.pydos import fvacf; \
                 fvacf(np.random.rand(5000,1000,3))"

If this env var is NOT set, then OpenMP uses all available cores (e.g. 4 on a
quad-core box).

IMPORTANT:
    Note that we may have found a f2py bug (see test/test_f2py_flib_openmp.py)
    re. OMP_NUM_THREADS. We have a workaround for that in pydos.fvacf().

There is also an optional arg 'nthreads' to _flib.vacf(). If this is
supplied, then it will override OMP_NUM_THREADS. Currently, this is the
safest way to set the number of threads.

Tests
^^^^^
When developing OpenMP code, you may find that code doesn't produce correct
results, even if it runs, if OpenMP is used incorrectly :) The test script
``test/runtests.sh`` calls `make gfortran-omp`, so if code is broken by OpenMP,
all test using the Fortran extensions might fail. To run tests with other
builds, use one of

.. code-block:: shell

    $ make gfortran
    $ make ifort
    $ make ifort-omp

and

.. code-block:: shell

    $ cd test
    $ ./runtests.sh --nobuild


.. include:: refs.rst
