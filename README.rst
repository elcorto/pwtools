About
-----
The pwtools package. Copyright (C) 2016 Steve Schmerler 

``pwtools`` is a Python package for pre- and postprocessing of atomic
calculations, mostly targeted to `Quantum Espresso`_, CPMD_, CP2K_ and
LAMMPS_. It is almost, but not quite, entirely unlike ASE_, with some tools
extending numpy_/scipy_. It has a set of powerful parsers and data types for
storing calculation data. See the `feature overview`_ for more.

Documentation
-------------
Have a look at `the docs`_ -- html version of the files in `doc/`_ along with
auto-generated `API documentation`_ (by using sphinx-autodoc_). Quick start
instructions can be found in `the tutorial`_. Many examples, besides the ones
in the doc strings are in `the tests`_.

Quick install
-------------
See the `install notes`_. Basically, the dependencies are::

	# apt
	python-numpy
	python-scipy
	python-nose
	python-dev
	python-h5py 
	python-matplotlib 
	python-ase
	python-numpydoc
	python-sphinx
	gfortran
	liblapack-dev

	# pip
	pycifrw
	pyspglib

.. FIXME as of v ersion 1.9.x, pyspglib is renamed to spglib, update dependency
   list once we tested this

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


.. ---------------------------------------------------------------------------
   link tagrgets, see also doc/source/written/refs.rst
   ---------------------------------------------------------------------------

.. _QE: http://www.quantum-espresso.org
.. _CPMD: http://www.cpmd.org
.. _CP2K: http://cp2k.org   
.. _LAMMPS: http://lammps.sandia.gov   
.. _ASE: https://wiki.fysik.dtu.dk/ase
.. _numpy: http://www.numpy.org
.. _scipy: http://www.scipy.org

.. _install: http://elcorto.github.io/pwtools/written/install.html
.. _tutorial: http://elcorto.github.io/pwtools/written/tutorial.html
.. _docs_html: http://elcorto.github.io/pwtools
.. _docs_files: https://github.com/elcorto/pwtools/tree/master/doc
.. _overview: http://elcorto.github.io/pwtools/written/features/overview.html#overview
.. _api: http://elcorto.github.io/pwtools/generated/api/index.html
.. _tests: https://github.com/elcorto/pwtools/tree/master/test   
.. _sphinx-autodoc: https://github.com/elcorto/sphinx-autodoc

.. Define derived link names here. Reason: We have nice and short labels which
   we may want to use multiple times. Since GitHub's rst renderer doesn't
   support the valid rst
   
       Have a look at `the website <foo_>`_
       
       .. _foo: http://www.foo.com
      
   we need to use either direct inline (which is impossible to read in the
   text-only version)
       
       Have a look at `the website <http://www.foo.com>`_
   
   or 
       
       Have a look at `the website`_
       
       .. _foo: http://www.foo.com
       .. _the website: foo_  

.. _the tutorial: tutorial_
.. _the tests: tests_
.. _Quantum Espresso: QE_   
.. _install notes: install_   
.. _feature overview: overview_
.. _the docs: docs_html_
.. _doc/: docs_files_
.. _API documentation: api_
