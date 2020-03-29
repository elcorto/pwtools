Relation to ASE
===============

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
