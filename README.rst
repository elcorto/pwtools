About
-----
The pwtools package. Copyright (C) 2008-2014 Steve Schmerler 

pwtools is a package for pre- and postprocessing of atomic calculations, mostly
targeted to Quantum Espresso, CPMD, CP2K and LAMMPS.

Documentation
-------------
`Have a look here <http://elcorto.bitbucket.org/pwtools>`_ for API
documentation, extended install notes (e.g. dependencies on other python
packages) and much more. This is built from the rst files in the ``doc/``
directory, which you may also read directly, of course.

By far the most documentation is contained in the source doc strings. Therefore
it is most helpful to look at the `API reference doc`_ and/or simply:

    Read the Source, Luke!

Many examples, besides the ones in The Source are in ``test/``.

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
