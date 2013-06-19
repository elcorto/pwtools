.. _features:

Features
========

* container classes for single unit cells (:class:`~pwtools.crys.Structure`)
  and molecular dynamics trajectories (:class:`~pwtools.crys.Trajectory`)

* classes to set up calculations (parameter studies) based on template input
  files for any kind of computational backend (:mod:`~pwtools.batch`)

* simple sqlite3 interface with convenience data extraction methods
  (:mod:`~pwtools.sql`)

* velocity autocorrelation function and phonon DOS from MD trajectories
  (:mod:`~pwtools.pydos`)

* parsing of PWscf [qe]_ and CPMD [cpmd]_ output into Python objects for easy
  access (:mod:`~pwtools.parse`)

* structure io: read cif, cml, pdb, write axsf, cif, xyz  (:mod:`~pwtools.io`)

* interface to external EOS fitting tools (:mod:`~pwtools.eos`)

* thermodynamic properties in the quasi-harmonic approximation from phonon
  density of states (:mod:`~pwtools.thermo`) 

* MD analayis: radial pair distribution function (own implementation and VMD
  [vmd]_ interface), RMS, RMSD (:mod:`~pwtools.crys`)

* unit cell related tools: super cell building, coordinate transformation,
  k-grid tools, ... (:mod:`~pwtools.crys`)

* funcions and classes to extend numpy/scipy (:mod:`~pwtools.num`)

* basic signal processing / fft related tools (:mod:`~pwtools.signal`)

* radial basis function interpolation (:mod:`~pwtools.rbf`)

* tools to handle matplotlib plots in scripts (:mod:`~pwtools.mpl`)


The very nice ASE [ase]_ project is in some way related. It also stores MD
trajectory data in Python objects for further manipulation. It's IO
capabilities are more developed. If needed, one can for instance convert a
:class:`~pwtools.crys.Structure` to an ASE Atoms object and use ASEs file
writers. ASE provides interfaces to a large pile of ab initio codes
("calculators"). MD and structure optimization in ASE is coded in Python, using
only the calculator's SCF engine in every step to get energy and forces.
Unfortunately, it does not  provide wave function extrapolation for
Born-Oppenheimer MD [*] and phonon calculations based on density functional
perturbation theory like PWscf/PH or Abinit (only supercell frozen phonons can
be done with the related [phonopy]_ package). The focus of the pwtools
package is to be a handy pre- and postprocessor providing pythonic access to
all input and output quantities of the used ab initio codes (PWscf, CPMD). 

In fact, appart from :mod:`~pwtools.parse`, which implements parsers for ab
initio code output and :mod:`~pwtools.pwscf`, all other parts of the package
are completely independent from any external code's output. 

[*] Last time I checked, I stumbled over a `mailing list thread`_ where they said
that in LCAO mode, the density would be re-used between steps.

.. _`mailing list thread`: https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2013-April/002044.html   
