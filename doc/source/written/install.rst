Installation
============

There is no installation script (i.e. ``setup.py``). Just copy the whole
package somewhere and run ``make`` to compile extensions

.. code-block:: shell

    $ cd $HOME/python/
    $ git clone https://github.com/elcorto/pwtools.git
    $ cd pwtools
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

Debian:

.. code-block:: shell

    $ sudo apt-get install python-numpy python-scipy python-nose python-dev \
                           python-h5py gfortran python-matplotlib \
                           liblapack-dev
    $ sudo apt-get install pip
    $ pip install pycifrw 
    $ pip install pyspglib

Must have
~~~~~~~~~

* numpy_
* scipy_
* nose_ (for running tests in ``test/``)
* python headers (development files for compiling Fortran extension)  
* Fortran compiler (e.g. gfortran will do fine)
* Blas and Lapack (for ``flib.f90``)
* Unix tools: grep, sed, awk, tail, wc (for :mod:`~pwtools.parse`); gzip/gunzip (for
  ``test/``). If possible, install mawk, which is much faster than GNU awk. It
  will be used automatically if found on your system.

Almost must have
~~~~~~~~~~~~~~~~
  
* `PyCifRW <pycifrw_orig_>`_: For Cif files in  :class:`~pwtools.parse.CifFile`,
  :func:`~pwtools.io.read_cif` and :func:`~pwtools.io.write_cif`. You may get a
  DeprecationWarning regarding the ``sets`` module. There is a patched version
  from pwextern-free_, which deals with that. But check the `pip version 
  <https://pypi.python.org/pypi/PyCifRW>`_ first.
* `pyspglib <pyspglib_>`_: used in :mod:`~pwtools.symmetry`, also shipped with
  pwextern-free_. Again, check `pip <https://pypi.python.org/pypi/pyspglib>`_.
* h5py_: for some functions in :mod:`~pwtools.io` currently

Suggested
~~~~~~~~~

* matplotlib_ (``examples/``)
* VMD_ (``examples/rpdf/``, :func:`~pwtools.crys.call_vmd_measure_gofr`,
  :func:`~pwtools.visualize.view_vmd_axsf`,
  :func:`~pwtools.visualize.view_vmd_xyz`), must register before download
* ASE_: :mod:`~pwtools.calculators`

Optional
~~~~~~~~

* The ``fourier.x`` tool from the CPMD_ contrib sources (for
  ``examples/``). Need to register before download.
* eos (for :mod:`~pwtools.eos`): The tool "eos" from the Elk_ code must
  be on your path. Note that the executable is assumed to be named "eos.x"
  instead of the default name "eos". See :class:`pwtools.eos.ElkEOSFit` for
  usage. Can be installed directly from Elk or also pwextern-free_.

The pwextern-free_ package contains add-on tools which we don't want / can ship
directly with pwtools, such as eos, PyCifRW and pyspglib, together with an
install script.

.. note:: :class:`pwtools.eos.ElkEOSFit` is deprecated now, and so you don't
   really need the Elk code's eos tool anymore. The other two things which come
   with pwextern-free_ can be installed by::
   
    pip install pycifrw 
    pip install pyspglib

All imports of optional Python modules will silently fail such that the code
can be used anywhere without errors or annoying warnings. The code parts which
use the dependencies will then fail only if used. And of course the related
tests will fail. That is no problem if you don't need the corresponding
functionality.

You can use ``test/check_dependencies.py`` to find out what your system has
installed.

Running tests
-------------

See tests/README. Actually, all of these are good examples, too!

Python versions
---------------

Developed mostly with Python 2.5..2.7. Should work with all versions from 2.4
on, but not yet 3.x. 

Compiling Fortran extensions and OpenMP notes
---------------------------------------------

Use the ``Makefile``:

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

On some systems (Debian), you may have:

.. code-block:: shell

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
Check with:

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
builds, use one  of
    
.. code-block:: shell

    $ make gfortran
    $ make ifort
    $ make ifort-omp

and

.. code-block:: shell

    $ cd test 
    $ ./runtests.sh --nobuild 


.. include:: refs.rst
