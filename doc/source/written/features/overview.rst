.. _overview:

Overview
========

* container classes for single unit cells (:class:`~pwtools.crys.Structure`)
  and molecular dynamics trajectories (:class:`~pwtools.crys.Trajectory`)

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
  xcrysden, avogadro, jmol, vmd (:mod:`~pwtools.visualize`)

* interface to the Elk_ code's EOS fitting tool and own implementation (only Vinet
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

* radial basis function interpolation (:mod:`~pwtools.rbf`)

* tools to handle matplotlib plots in scripts (:mod:`~pwtools.mpl`)

* QE and LAMMPS calculators for ASE (:mod:`~pwtools.calculators`)

The very nice ASE_ project is in some way related. It also stores atomic
structure data in Python objects for further manipulation. If needed, one can
convert a :class:`~pwtools.crys.Structure` to an ASE Atoms object. The design
goal of ASE is, however, different from pwtools. ASE provides interfaces to a
large pile of ab initio codes ("calculators"). MD and structure optimization in
ASE is coded in Python, using only the calculator's SCF engine in every step to
get energy and forces. This is a very good idea, but only structure
optimization is really developed and tested, as it seems. MD not so much.
Better use a special MD code here. I'm not sure if ASE provides wave function
extrapolation for Born-Oppenheimer MD [*]. Phonon calculations based on density
functional perturbation theory like PWscf/PH or Abinit are not implemented
(maybe in GPAW_?). However, the supercell method can be used with the related
phonopy_ package. The focus of the pwtools package is to be a handy pre- and
postprocessor providing pythonic access to all input and output quantities of
the used ab initio codes. In ASE, the calculator abtracts the backend code's
input away. With pwtools, you need to know the input file syntax of your
backend code. Once you know that, you use only template files to set up
calculations. Regarding visualization, ASE has some kind of GUI. We have
:mod:`~pwtools.visualize`, which is best used in an interactive Ipython
session.

In fact, appart from :mod:`~pwtools.parse`, which implements parsers for ab
initio code output and :mod:`~pwtools.pwscf`, all other parts of the package
are completely independent from any external simulation code's output.
Especially the parameter study tools in :mod:`~pwtools.batch` can be used for
any kind of (computational) study, since only user-supplied template files are
used. 

[*] Last time I checked, I stumbled over a `mailing list thread`_ where they said
that in LCAO mode, the density would be re-used between steps.

.. _mailing list thread: https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2013-April/002044.html  

.. include:: ../refs.rst
