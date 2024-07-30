.. _features:

Features
========

* Container classes for single unit cells (:class:`~pwtools.crys.Structure`)
  and structure sequences such as molecular dynamics trajectories, relaxation
  runs or NEB paths (:class:`~pwtools.crys.Trajectory`). See
  :ref:`parsers_and_containers`.

* Classes to set up calculations (parameter studies) based on template input
  files for any kind of computational backend (:mod:`~pwtools.batch`). See
  :ref:`param_study`.

* Simple sqlite3 interface with convenience data extraction methods
  (:mod:`~pwtools.sql`).

* Parsing of PWscf (QE_), CPMD_ , CP2K_ and LAMMPS_
  output into Python objects for easy access (:mod:`~pwtools.parse`). See
  :ref:`parsers_and_containers`.

* Structure io: read cif, pdb, write axsf, cif, xyz  (:mod:`~pwtools.io`)

* Pythonic interface to external molecular viewers for interactive use:
  xcrysden_, avogadro_, jmol_, VMD_ (:mod:`~pwtools.visualize`).

* EOS fitting tools (:mod:`~pwtools.eos`)

* Thermodynamic properties in the quasi-harmonic approximation from phonon
  density of states, QHA implementation (:mod:`~pwtools.thermo`). See
  :ref:`qha`.

* MD analysis: radial pair distribution function (own implementation and VMD_
  interface), RMS, RMSD (:mod:`~pwtools.crys`).

* Velocity autocorrelation function and phonon DOS from MD trajectories
  (:mod:`~pwtools.pydos`). See :ref:`pdos_from_vacf`.

* Unit cell related tools: super cell building, coordinate transformation,
  k-grid tools, ... (:mod:`~pwtools.crys`).

* Thin wrappers for spglib_ functions (:mod:`~pwtools.symmetry`)

* Functions and classes to extend numpy/scipy, e.g. N-dim polynomial fitting
  and a number of convenient 1D classes (polynomial, spline) with a common
  API (:mod:`~pwtools.num`).

* N-dim radial basis function interpolation and fitting
  (:mod:`~pwtools.rbf.core`, :mod:`~pwtools.rbf.hyperopt`). See :ref:`rbf`.

* Basic signal processing / fft related tools (:mod:`~pwtools.signal`)

* Tools to handle matplotlib plots in scripts (:mod:`~pwtools.mpl`)

* QE and LAMMPS calculators for ASE (:mod:`~pwtools.calculators`)

* extensive test suite

.. include:: refs.rst
