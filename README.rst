About
-----
.. image:: .em.png
   :scale: 40%
   :align: right

The pwtools package. Copyright (C) 2008-2013 Steve Schmerler 

pwtools is a package for pre- and postprocessing of atomic calculations, mostly
targeted to Quantum Espresso, CPMD and CP2K.

Documentation
-------------
See doc/ for doumentation and extended install notes (e.g. dependencies on
other python packages). This dir contains rst source files which can be used to
build nice sphinx html docu. See doc/README to find out how to build the html
docu by youself. A semi-recent version of that can be found at
http://elcorto.bitbucket.org/pwtools

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
    export PYTHONPATH=$PYTHONPATH:$HOME/python

.. _API reference doc: http://elcorto.bitbucket.org/pwtools/generated/api/index.html
