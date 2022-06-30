Installation
============

Quick start
-----------

Debian and derivatives: Fortran compiler, Python headers, lapack

.. code-block:: sh

    $ sudo apt install python3-dev gfortran liblapack-dev

Then


.. code-block:: sh

    # <<probably make a venv>>
    $ pip install git+https://github.com/elcorto/pwtools

or

.. code-block:: sh

    $ git clone https://github.com/elcorto/pwtools
    $ cd pwtools
    $ pip install [-e] .


Detailed instructions
---------------------

To build the extension modules and install all Python dependencies via
``pip``

.. code-block:: sh

    $ git clone https://github.com/elcorto/pwtools
    $ pip install .

or the setuptools "development install" (no copy of files)

.. code-block:: sh

    $ pip install -e .

In both cases we build the extension modules via

.. code-block:: sh

    $ cd src && make clean && make $PWTOOLS_EXT_MAKE_TARGET && cd ..

in the background, where ``$PWTOOLS_EXT_MAKE_TARGET`` is optional (more details
in the :ref:`extensions` section).

If all dependencies are installed, e.g. by a package manager such as ``apt``,
then recent ``pip`` versions should pick those up. Alternatively force ``pip``
to install only the package without installing dependencies from pypi

.. code-block:: sh

    $ pip install --no-deps .

When using a virtual environment, you can map in the system site-packages (here
we use virtualenvwrapper_)

.. code-block:: sh

    $ mkvirtualenv --system-site-packages -p /usr/bin/python3 pwtools
    (pwtools) $ pip install .

Alternatively, you may also simply build the extensions and set ``PYTHONPATH``

.. code-block:: sh

    $ cd src && make clean && make && cd ..
    $ [ -n "$PYTHONPATH" ] && pp=$(pwd):$PYTHONPATH || pp=$(pwd)
    $ export PYTHONPATH=$pp


Dependencies
------------

Python dependencies: see ``requirements*.txt``. You can use
``test/check_dependencies.py`` to find out what your system has installed

.. code-block:: sh

    $ ./check_dependencies.py
    requirements_test.txt
      pytest               ... ok (import)
      pytest-xdist         ... ok (pip list)
      pytest-timeout       ... ok (pip list)
    requirements.txt
      PyCifRW              ... ok (pip list)
      h5py                 ... ok (import)
      matplotlib           ... ok (import)
      numpy                ... ok (import)
      scipy                ... ok (import)
      spglib               ... ok (import)
    requirements_optional.txt
      ase                  ... ok (import)
      sklearn              ... ok (import)
    requirements_doc.txt
      sphinx               ... ok (import)
    optional executables:
      eos.x                ... NOT FOUND

Optional packages (``ase``, ``sklearn``) are only used in corner cases and are
therefore not hard dependencies. Importing modules that use them (e.g. ``crys``
might use ``ase``) won't fail at import time, but later at runtime when e.g.
``crys.Structure.get_ase_atoms()`` is called. What packages are optional might
change depending on usage.

You also need the following to compile extensions. On a Debian-ish system

.. code-block:: sh

    # apt
    python3-dev     # for compiling extensions
    gfortran        # or ifort, see src/Makefile
    liblapack-dev

Also note that you can get may Python packages via your system's package
manager

.. code-block:: sh

    python3-ase
    python3-h5py
    python3-matplotlib
    python3-numpy
    python3-pytest
    python3-pytest-timeout
    python3-pytest-xdist
    python3-scipy
    python3-sklearn
    python3-sphinx

But usually not ``PyCifRW`` and ``spglib``.


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
  usage.

.. note:: :class:`~pwtools.eos.ElkEOSFit` is deprecated, you don't
   really need the Elk code's eos tool anymore. It was used to generate
   reference EOS fit data for unit tests, which is stored in test/files/, but
   there is no use of eos.x in pwtools anymore.

Running tests
-------------

See test/README. Actually, all of these are good examples, too!

Python versions
---------------

Only Python3 is supported, tested: Python 3.6-3.10

The package was developed mostly with Python 2.5-2.7 and ported using 2to3 +
manual changes. Therefore, you might find some crufty Python 2 style code
fragments in lesser used parts of the code base.


.. _extensions:

Fortran extensions and OpenMP notes
-----------------------------------

Use ``src/Makefile``:

.. code-block:: sh

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
* Lapack (Debian: liblapack-dev)

The module is compiled with f2py (currently part of numpy, tested with numpy
1.1.0 .. 1.21.x).

The default ``make`` target is "gfortran" which tries to build a serial version
using system BLAS and LAPACK (e.g. from ``liblapack-dev``). If you want another
target (e.g. ``ifort-mkl``), then

   .. code-block:: sh

    $ cd src
    $ make clean
    $ make ifort-mkl

or when using ``pip`` (or anything calling ``setup.py``) set
``$PWTOOLS_EXT_MAKE_TARGET``.

.. code-block:: sh

    $ PWTOOLS_EXT_MAKE_TARGET=ifort-mkl pip install ...

This will use the Intel ``ifort`` compiler instead fo the default ``gfortran`` and
link against the MKL.

In the MKL case, the Makefile uses the env var ``$MKL_LIB`` which sould point
to the location where things like ``libmkl_core.so`` live. You may need to set
this. On a HPC cluster, that could look like this.

.. code-block:: sh

    # module only sets MKL_ROOT=/path/to/intel/20.4/mkl
    $ module load intel/20.4
    $ MKL_LIB=$MKL_ROOT/lib/intel64 PWTOOLS_EXT_MAKE_TARGET=ifort-mkl pip install ...

See ``src/Makefile`` for more details.

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

.. code-block:: sh

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

.. code-block:: sh

    $ ldd _flib.so

Setting the number of threads:

.. code-block:: sh

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

.. code-block:: sh

    $ make gfortran
    $ make ifort
    $ make ifort-omp

and

.. code-block:: sh

    $ cd test
    $ ./runtests.sh --nobuild


.. include:: refs.rst
