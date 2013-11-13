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
    export PYTHONPATH=$PYTHONPATH:$HOME/python

See :ref:`fextensions` for more information on how to build the extension
modules.

Add-on packages / other required tools
--------------------------------------

On Debian:: 

    aptitude install python-numpy python-scipy python-nose python-dev python-h5py gfortran \
                     python-beautifulsoup python-matplotlib liblapack3 

Must have:    

* numpy
* scipy
* nose (for running tests in test/)
* python headers (development files for compiling Fortran extension)  
* Fortran compiler (e.g. gfortran will do fine)
* Blas and Lapack (for ``flib.f90``)
* Unix tools: grep, sed, awk, tail, wc (for :mod:`~pwtools.parse`); gzip/gunzip (for
  ``test/``)

Almost must have:
  
* BeautifulSoup [beautifulsoup]_: XML parser (for .cml files)
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
* h5py: only for some functions in :mod:`~pwtools.io.` currently
* pyspglib [spglib]_

Effects of missig dependencies:

* BeautifulSoup and PyCifRW: You will get import warnings, some parsing
  classes and IO functions will not work (Cif and CML parsing currently) and some
  tests will fail. If you don't need that functionality, uncomment the warnings
  and import statements at the top of :mod:`~pwtools.parse` and :mod:`~pwtools.io`.
* eos: :mod:`pwtools.eos.ElkEOSFit` and related tests won't work.
* spglib: some symmetry finding functions in :mod:`~pwtools.symmetry` won't
  work

Suggested:

* matplotlib (``examples/``)
* VMD [vmd]_ (``examples/rpdf/``, :func:`~pwtools.crys.call_vmd_measure_gofr`,
  :func:`~pwtools.visualize.view_vmd_axsf`,
  :func:`~pwtools.visualize.view_vmd_xyz`), must register before download

Optional:

* The "fourier.x" tool from the CPMD [cpmd]_ contrib sources (for
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


