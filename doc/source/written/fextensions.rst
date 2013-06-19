.. _fextensions:

Compiling Fortran extensions and OpenMP notes
=============================================

Use the ``Makefile``::

    $ make help
    make gfortran            # gfortran, default
    make gfortran-omp        # gfortran + OpenMP
    make ifort               # ifort
    make ifort-omp           # ifort + OpenMP

Generates ``*.so`` and ``*.pyf`` (f2py interface) files.

You need:

* numpy
* a Fortran compiler
* Python headers (for Linux: usually a package python-dev or python-devel,
  see the package manager of your distro)

The module is compiled with f2py (currently part of numpy, tested with numpy
1.1.0 .. 1.6.x). 

Compiler / f2py
---------------
Instead of letting numpy.distutils pick a compiler + special flags, which is
not trivial and therefore almost never works, it is much easier to simply
define the compiler to use + architecture-specific flags. See F90 and ARCH in
the Makefile.

Also, numpy.distutils has default -03 for fcompiler. ``--f90flags="-02"`` does NOT
override this. We get ``-O3 -O2`` and a compiler warning. We have to use f2py's
``--opt=`` flag.

On some systems (Debian), you may have::

  /usr/bin/f2py -> f2py2.6
  /usr/bin/f2py2.5
  /usr/bin/f2py2.6

and such. But usually ``F2PY=f2py`` is fine.

OpenMP 
------
We managed to speed up the calculations by sprinkling some OpenMP
pragmas in ``*.f90``. This works pretty good. If you wanna try, use 
``make ifort-omp`` or ``make gfortran-omp``.

If all went well, _flib.so should be linked to libgomp (or libiomp for ifort).
Check with::
	
	$ ldd _flib.so

Setting the number of threads::  
	
	$ export OMP_NUM_THREADS=2
	$ python -c "import numpy as np; from pwtools.pydos import fvacf; \
	             fvacf(np.random.rand(1000,3,5000))"

If this env var is NOT set, then OpenMP uses all available cores (e.g. 4 on a
quad-core box).

IMPORTANT: 
	Note that we may have found a f2py bug (see test/test_f2py_flib_openmp.py)
	re. OMP_NUM_THREADS. We have a workaround for that in pydos.fvacf().

There is also an optional arg 'nthreads' to _flib.vacf(). If this is
supplied, then it will override OMP_NUM_THREADS. Currently, this is the
safest way to set the number of threads.

Tests
-----
When developing OpenMP code, you may find that code doesn't produce correct
results, even if it runs, if OpenMP is used incorrectly :) The test script
test/runtests.sh just calls `make`, which builds the default target, i.e.
*non*-OpenMP code. Therefore, tests will pass b/c the code is correct in serial
mode. So, for development, it is better use ``make ifort-omp`` + ``runtests.sh
--nobuild`` in separate steps to make sure the right version of the extension
is being built.
