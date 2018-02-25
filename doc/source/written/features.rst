.. _features:

Features
========

* container classes for single unit cells (:class:`~pwtools.crys.Structure`)
  and structure sequences such as molecular dynamics trajectories, relaxation
  runs or NEB paths (:class:`~pwtools.crys.Trajectory`)

* classes to set up calculations (parameter studies) based on template input
  files for any kind of computational backend (:mod:`~pwtools.batch`)

* simple sqlite3 interface with convenience data extraction methods
  (:mod:`~pwtools.sql`)

* velocity autocorrelation function and phonon DOS from MD trajectories
  (:mod:`~pwtools.pydos`)

* parsing of PWscf (QE_), CPMD_ , CP2K_ and LAMMPS_ 
  output into Python objects for easy access (:mod:`~pwtools.parse`)

* structure io: read cif, pdb, write axsf, cif, xyz  (:mod:`~pwtools.io`)

* pythonic interface to external molecular viewers for interactive use:
  xcrysden_, avogadro_, jmol_, VMD_ (:mod:`~pwtools.visualize`)

* interface to the Elk_ code's EOS fitting tool and own implementation (Vinet
  EOS) (:mod:`~pwtools.eos`)

* thermodynamic properties in the quasi-harmonic approximation from phonon
  density of states, QHA implementation (:mod:`~pwtools.thermo`) 

* MD analyis: radial pair distribution function (own implementation and VMD_
  interface), RMS, RMSD (:mod:`~pwtools.crys`)

* unit cell related tools: super cell building, coordinate transformation,
  k-grid tools, ... (:mod:`~pwtools.crys`)

* functions and classes to extend numpy/scipy, e.g. N-dim polynomial fitting
  (!) and a number of convenient 1D classes (polynomial, spline) with a common
  API (:mod:`~pwtools.num`)

* basic signal processing / fft related tools (:mod:`~pwtools.signal`)

* N-dim radial basis function interpolation and fitting (:mod:`~pwtools.rbf`)

* tools to handle matplotlib plots in scripts (:mod:`~pwtools.mpl`)

* QE and LAMMPS calculators for ASE (:mod:`~pwtools.calculators`)

.. include:: refs.rst
