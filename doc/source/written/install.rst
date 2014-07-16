Installation
============

There is no installation script (i.e. setup.py). Just copy the whole package
somewhere and run ``make`` to compile extensions::

    $ tar -xzf pwtools-x.y.z.tgz
    $ mv pwtools-x.y.z $HOME/python/pwtools
    $ cd $HOME/python/pwtools
    $ make
    # ~/.bashrc or ~/.profile
    export PATH=$HOME/python/pwtools/bin:$PATH
    if [ -n "$PYTHONPATH" ]; then 
        export PYTHONPATH=$HOME/python:$PYTHONPATH
    else
        export PYTHONPATH=$HOME/python
    fi

Make sure that ``PYTHONPATH`` doesn't start with ``:`` and that you don't have
``::`` in it. Otherwise, other modules may try to import ``pwtools/io.py``
instead of the std lib's ``io``. For example `Mercurial
<http://mercurial.selenic.com>`_ imports ``tempfile``, which imports
``io``.


Add-on packages / other required tools
--------------------------------------

On Debian:: 

    apt-get install python-numpy python-scipy python-nose python-dev python-h5py gfortran \
                    python-matplotlib liblapack-dev

Must have:    

* numpy
* scipy
* nose (for running tests in ``test/``)
* python headers (development files for compiling Fortran extension)  
* Fortran compiler (e.g. gfortran will do fine)
* Blas and Lapack (for ``flib.f90``)
* Unix tools: grep, sed, awk, tail, wc (for :mod:`~pwtools.parse`); gzip/gunzip (for
  ``test/``)

Almost must have:
  
* PyCifRW [pycifrw_orig]_: For Cif files. 
  With Python 2.6, you may get a DeprecationWarning regarding the sets module
  when the CifFile module from the pycifrw package is imported. There is a
  patched version from [pwextern-free]_, which deals with that.
  Note that recent versions of PyCifRW may already include that fix, so first
  try that. Hint: google for "pycifrw" :)
* eos (for :mod:`~pwtools.eos`): The tool "eos" from the Elk code [elk]_ must
  be on your path. Note that the executable is assumed to be named "eos.x"
  instead of the default name "eos". See :class:`pwtools.eos.ElkEOSFit` for
  usage. Can be installed directly from Elk or also [pwextern-free]_.
* h5py: only for some functions in :mod:`~pwtools.io` currently
* pyspglib [spglib]_: used in :mod:`~pwtools.symmetry`

Effects of missig dependencies:

* PyCifRW: You will get import warnings, some parsing
  classes and IO functions will not work (Cif files) and some tests will fail.
  If you don't need that functionality, uncomment the warnings and import
  statements at the top of :mod:`~pwtools.parse` and :mod:`~pwtools.io`.
* eos: :mod:`pwtools.eos.ElkEOSFit` and related tests won't work.
* spglib: some symmetry finding functions in :mod:`~pwtools.symmetry` won't
  work

Suggested:

* matplotlib (``examples/``)
* VMD [vmd]_ (``examples/rpdf/``, :func:`~pwtools.crys.call_vmd_measure_gofr`,
  :func:`~pwtools.visualize.view_vmd_axsf`,
  :func:`~pwtools.visualize.view_vmd_xyz`), must register before download

Optional:

* The ``fourier.x`` tool from the CPMD [cpmd]_ contrib sources (for
  ``examples/``). Need to register before download.

The "pwextern-free" package [pwextern-free]_ over at bitbucket.org contains
add-on tools which we don't want / can ship directly with pwtools.

Running tests
-------------

See tests/README. Actually, all of these are good examples, too!

Python versions
---------------

Developed mostly with Python 2.5..2.7. Should work with all versions from 2.4
on, but not yet 3.x. 

Compiling Fortran extensions and OpenMP notes
---------------------------------------------

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
* Python headers (Debian/Ubuntu: python-dev)
* Lapack (Debian: liblapack3)

The module is compiled with f2py (currently part of numpy, tested with numpy
1.1.0 .. 1.7.x). 

Compiler / f2py
^^^^^^^^^^^^^^^
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
^^^^^^
We managed to speed up the calculations by sprinkling some OpenMP
pragmas in ``*.f90``. This works pretty good. If you wanna try, use 
``make ifort-omp`` or ``make gfortran-omp``.

If all went well, _flib.so should be linked to libgomp (or libiomp for ifort).
Check with::
	
	$ ldd _flib.so

Setting the number of threads::  
	
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
builds, use one  of::
    
    make gfortran
    make ifort
    make ifort-omp

and::

    cd test 
    ./runtests.sh --nobuild 
