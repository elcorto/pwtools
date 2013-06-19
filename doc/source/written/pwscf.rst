Pwscf
=====

Units
-----

From http://www.quantum-espresso.org/input-syntax/INPUT_PW.html:
"All quantities whose dimensions are not explicitly specified are in
RYDBERG ATOMIC UNITS". See also ``constants.py``.

matdyn.x and phonon dos: 
    With dos=.true., matdyn.x calculates the phonon density of states. The
    frequency axis in the output file is f in cm^-1. Note that THIS IS NOT THE
    ANGULAR FREQUENCY omega = 2*pi*f!!! Therefore, if you calculate with this
    frequency as in "hbar*omega", then use f*2*pi!

Atomic coordinates
------------------

In PWscf terms, alat = celldm(1) = lattice constant "a" in a.u. . A length in
a.u. means a unit or Bohr = a0 = 0.52917720859 Angstrom.

To calculate the PDOS, atomic coordinates from MD trajectories have to be in
cartesian coordinates. You may have to transform them *before* using them to
calculate the PDOS. See

``test/test_pdos.py``
``test/test_pdos_coord_trans.py``

This is necessary if you have "ATOMIC_POSITIONS crystal".

The scale (or unit: Bohr, Angstrom, ..., defined by celldm(1) or
CELL_PARAMETERS) does not matter, b/c currently the integral area under the
PDOS curve is normalized in pydos.*_pdos(). But coords MUST be cartesian!

For your convenience, here is a list of all possible formats (from the Pwscf
help)::

    allowed ATOMIC_POSITIONS units:
       alat    : atomic positions are in cartesian coordinates,
                 in units of the lattice parameter "a" (default)

       bohr    : atomic positions are in cartesian coordinate,
                 in atomic units (i.e. Bohr)

       angstrom: atomic positions are in cartesian coordinates,
                 in Angstrom

       crystal : atomic positions are in crystal coordinates, i.e.
                 in relative coordinates of the primitive lattice vectors

Note: crystal coords are also called fractional coordinates (e.g. in Cif
files).

summary::

    ATOMIC_POSITIONS angstrom  -> cartesian angstrom
    ATOMIC_POSITIONS bohr      -> cartesian a.u. 
    ATOMIC_POSITIONS           -> cartesian alat
    ATOMIC_POSITIONS alat      -> cartesian alat
    ATOMIC_POSITIONS crystal   -> crystal alat or crystal a.u. (see below)

The unit of CELL_PARAMETERS is only important for ATOMIC_POSITIONS crystal::

    if celldm(1) present  -> CELL_PARAMETERS in alat -> crystal alat
        => CELL_PARAMETERS = real cell parameter divided by alat
    if not                -> CELL_PARAMETERS in a.u. -> crystal a.u.
        => CELL_PARAMETERS are in Rydberg atomic units, i.e. in Bohr.

Total force on atoms
--------------------

Pwscf writes a "Total Force" after the "Forces acting on atoms" section . This
value is kind of an RMS of the force matrix (f_ij, i=1,natoms j=1,2,3) printed.
According to .../PW/forces.f90, variable "sumfor", the "Total Force" is::

    sqrt(sum_ij f_ij^2)

But this is not normalized to the number of atoms. Use crys.rms() or
crys.rms3d() for MD runs where the RMS of each (f_ij) is::

    sqrt( (sum_ij f_ij^2) / N )

with N = 3*natoms or N=natoms.   
