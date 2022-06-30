.. image:: https://zenodo.org/badge/51149109.svg
   :target: https://zenodo.org/badge/latestdoi/51149109

About
-----
``pwtools`` is a Python package for pre- and postprocessing of atomistic
calculations, mostly targeted to `Quantum Espresso`_, CPMD_, CP2K_ and
LAMMPS_. It is almost, but not quite, entirely unlike ASE_, with some tools
extending numpy_/scipy_. It has a set of powerful parsers and data types for
storing calculation data. See the `feature overview`_ for more.

The `dcd code <dcd_code_>`_ is now part of ASE_'s `dcd reader for CP2K
files <dcd_ase_code_>`_. `Thanks <dcd_ase_pr_>`_!


Documentation
-------------
Have a look at `the docs`_. Quick start instructions can be found in `the
tutorial`_. Many examples, besides the ones in the doc strings are in `the
tests`_.

Install
-------
See the `install docs`_.


.. ---------------------------------------------------------------------------
   link targets, see also doc/source/written/refs.rst
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
.. _features: http://elcorto.github.io/pwtools/written/features.html
.. _api: http://elcorto.github.io/pwtools/generated/api/index.html
.. _tests: https://github.com/elcorto/pwtools/tree/master/pwtools/test
.. _sphinx-autodoc: https://github.com/elcorto/sphinx-autodoc
.. _dcd_code: https://github.com/elcorto/pwtools/blob/master/pwtools/dcd.py
.. _dcd_ase_pr: https://gitlab.com/ase/ase/merge_requests/1109
.. _dcd_ase_code: https://gitlab.com/ase/ase/blob/master/ase/io/cp2k.py

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
.. _install docs: install_
.. _feature overview: features_
.. _the docs: docs_html_
.. _doc/: docs_files_
.. _API documentation: api_
