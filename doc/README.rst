This dir contains the documentation rst source files. To build the html doc,
do::

    make html
    firefox build/html/index.html

For this to work, you need Sphinx, the Python documention generation tool
(Debian: ``apt-get install python-sphinx``). If you don't have that, you can
simply read the ``source/written/*.rst`` files :)

By default (i.e. unpacked tarball or checked out source), this will build the
hand-written documentation (like the tutorial and install notes) located in
``source/written/``.

If you like to generate the optional API doc source ``*.rst`` files
automagically before ``make html``, you need ``sphinx-autodoc.py`` [1]_ which
is used in ``generate-apidoc.sh`` to generate ``source/generated/*``. You
also need numpydoc. Then::

    sudo apt-get install python-numpydoc
    ./generate-apidoc.sh
    make html
    firefox build/html/index.html

``generate-apidoc.sh`` will try to find ``sphinx-autodoc.py`` on PATH or clone
its git repo. Then, ``make html`` will run longer and produce many more
warnings :) You may also want to check sphinx-autodoc's README file for more
information.

.. [1] https://github.com/elcorto/sphinx-autodoc
