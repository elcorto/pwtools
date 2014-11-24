About
-----
The pwtools package. Copyright (C) 2014 Steve Schmerler 

pwtools is a package for pre- and postprocessing of atomic calculations, mostly
targeted to Quantum Espresso, CPMD, CP2K and LAMMPS.

Documentation
-------------
The ``doc/`` directory contains most documentation, and a nice HTML version is
`here <http://elcorto.bitbucket.org/pwtools>`_. Quick start instructions can be
found in the `tutorial
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
