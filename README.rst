About
-----
The pwtools package. Copyright (C) 2015 Steve Schmerler 

pwtools is a Python package for pre- and postprocessing of atomic calculations,
mostly targeted to Quantum Espresso, CPMD, CP2K and LAMMPS. It is almost, but
not quite, entirely unlike ASE, with some tools extending numpy/scipy. It has a
set of powerful parsers and data types for storing calculation data.

Documentation
-------------
`Here <http://elcorto.bitbucket.org/pwtools>`_ you find the html version of the
files in ``doc/``. Quick start instructions can be found in the `tutorial
<http://elcorto.bitbucket.org/pwtools/written/tutorial.html>`_. Have a look at
the `API documentation
<http://elcorto.bitbucket.org/pwtools/generated/api/index.html>`_ for a
reference of all implemented functions and classes. This lists also all doc
strings. Many examples, besides the ones in the doc strings are in ``test/``.

Quick install
-------------
See ``doc/source/written/install.rst`` . For the impatient:

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

.. _API reference doc: http://elcorto.bitbucket.org/pwtools/generated/api/index.html
