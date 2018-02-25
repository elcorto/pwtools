Parsing code output and using containers
========================================

.. _parser_classes:

Available parsers
-----------------

A core feature of pwtools is a set of parsers for commonly used atomistic
simulation codes. The parsers are located in :mod:`~pwtools.parse`. These
parsers are available:

    | :class:`~pwtools.parse.CifFile`
    | :class:`~pwtools.parse.PDBFile`
    | :class:`~pwtools.parse.PwSCFOutputFile`
    | :class:`~pwtools.parse.PwMDOutputFile`
    | :class:`~pwtools.parse.PwVCMDOutputFile`
    | :class:`~pwtools.parse.CpmdSCFOutputFile`
    | :class:`~pwtools.parse.CpmdMDOutputFile`
    | :class:`~pwtools.parse.Cp2kSCFOutputFile`
    | :class:`~pwtools.parse.Cp2kMDOutputFile`
    | :class:`~pwtools.parse.Cp2kDcdMDOutputFile`
    | :class:`~pwtools.parse.Cp2kRelaxOutputFile`
    | :class:`~pwtools.parse.LammpsTextMDOutputFile`
    | :class:`~pwtools.parse.LammpsDcdMDOutputFile`

The parsers called ``*OutputFile`` are for parsing simulation code output. The
others parse structure files (cif and pdb).

All parsers have a common API and can be used like ::

    >>> pp = PwSCFOutputFile('pw.out')
    >>> pp.get_cell()
    >>> # or call all get_*() methods at once and access attributes
    >>> pp.parse()
    >>> pp.cell

The problem with using the parsers directly is that there are of course
differences between the codes. For instance, each one uses different units
(Bohr, Angstrom, Ry, Ha, eV, ...). Also, not all structure attributes are
contained in the output of all codes. For example,
:class:`~pwtools.parse.PwSCFOutputFile` will parse Cartesian atomic coordinates into
``pp.coords`` and the unit cell into ``pp.cell``. However, often we need to use
the fractional coordinates ``pp.coords_frac``. This quantity is not present in
the PWscf output and thus ``pp.coords_frac`` will be None. However, we know
that we can manually calculate it from ``coords`` and ``cell``. 

.. _container_classes:

Container classes :class:`~pwtools.crys.Structure` and :class:`~pwtools.crys.Trajectory`
----------------------------------------------------------------------------------------

In order to abstract away the differences between codes as much as possible, we
have implemented unified container classes. These classes are
:class:`~pwtools.crys.Structure` and :class:`~pwtools.crys.Trajectory`. The
former is used to represent a single crystal structure (unit cell, atom
coordinates, total energy, stress tensor, ...). The latter represents a
sequence of structures, for instance an MD or relaxation run. 

They have two important features:

* A defined set of units
  (eV, Angstrom,...), to which all quantities are converted.
* Calculate all missing
  attributes automatically and thus provide a unified API. 

Note that the latter is a convenience feature and will also produce some
redundant data. You may want to :ref:`turn it off <avoid_auto_calc>` 
for parsing/storing big data.

The auto-calculation of missing properties in :class:`~pwtools.crys.Trajectory`
and :class:`~pwtools.crys.Structure` is done by trying to calculate all
properties for which there is a ``get_*`` method. For example, if a parser
finds ``coords`` and ``cell`` in the MD data, then in
:class:`~pwtools.crys.Trajectory` ``coords_frac`` is calculated from that. 

You can of course use these classes to build new structures and trajectories by
hand (just as with ``ase.Atoms``, or you use :func:`~pwtools.crys.atoms2struct`)::

    >>> st = crys.Structure(coords_frac=np.array([[0]*3, [.5]*3]),
                            cryst_const=np.array([3.0]*3 + [60]*3),
                            symbols=['Al','N'])
    >>> tr = crys.Trajectory(coords_frac=rand(1000,20,3),
                             cell=rand(1000,3,3),
                             symbols=['H']*20)

By doing this, the :meth:`~pwtools.crys.Structure.set_all` method is
automatically called, which will calculate all possible attributes from the
input data (for example ``st.coords``, ``st.cell``).

However, some attributes may be undefined. For example, the ``st`` above will
have no ``etot`` or ``stress`` attribute (they are None), since that was not
defined in the input and there is no ways to calculate it, of course, whereas a
Structure returned by :func:`~pwtools.io.read_pw_scf` will have that.

By using the :meth:`~pwtools.base.FlexibleGetters.dump` method, you can store
the object as binary file [using Python's ``pickle`` module] for fast
re-loading later::

    >>> st.dump('struck.pk')
    >>> st_loaded = io.read_pickle('struck.pk')

A Trajectory object can be viewed a list of Structure instances [even though it
is implemented differently due to efficiency: we use 3d numpy arrays], it
supports iteration and slicing, for example::
    
    >>> # extract first and last Structure objects
    >>> st_first = tr[0]
    >>> st_last = tr[-1]
    >>> # slice out a part of the trajectory
    >>> tr_middle = tr[100:500]
    >>> # use every 5t step
    >>> tr[::5]

Structure and Trajectory objects can also be freely concatenated into a new
Trajectory::

    >>> tr_new = crys.concatenate((st1, st2))
    >>> tr_new = crys.concatenate((st, tr))
    >>> tr_new = crys.concatenate((tr1, tr2, st))

.. _high_level_parsing:

High-level parsing functions
----------------------------

The most simple way to parse code output and get a container class is to use
the high-level functions in :mod:`~pwtools.io`.

These return a :class:`~pwtools.crys.Structure`:
    | :func:`~pwtools.io.read_cif`
    | :func:`~pwtools.io.read_pdb`
    | :func:`~pwtools.io.read_pw_scf`
    | :func:`~pwtools.io.read_cpmd_scf`
    | :func:`~pwtools.io.read_cp2k_scf`

These return a :class:`~pwtools.crys.Trajectory`:
    | :func:`~pwtools.io.read_pw_md`
    | :func:`~pwtools.io.read_pw_vcmd`
    | :func:`~pwtools.io.read_cpmd_md`
    | :func:`~pwtools.io.read_cp2k_md`
    | :func:`~pwtools.io.read_cp2k_md_dcd`
    | :func:`~pwtools.io.read_cp2k_relax`
    | :func:`~pwtools.io.read_lammps_md_txt`
    | :func:`~pwtools.io.read_lammps_md_dcd`

For example::

    >>> st = io.read_pw_scf('pw.out') # SCF run
    >>> print st.etot, st.cell
    >>> tr = io.read_pw_md('pw.out') # MD/relax run
    >>> plot(tr.etot)

These functions use the appropriate parser class and transform the result of
the parsing to a :class:`~pwtools.crys.Structure` or
:class:`~pwtools.crys.Trajectory`. For example, what is essentially done is
simply::
    
    >>> # same as tr=io.read_pw_md('pw.out')
    >>> pp = parse.PwMDOutputFile('pw.out')
    >>> tr = pp.get_traj()

    >>> # same as st=io.read_cp2k_scf('cp2k.out')
    >>> pp = parse.Cp2kSCFOutputFile('cp2k.out')
    >>> st = pp.get_struct()

It is important to note that Structure and Trajectory instances built by hand
can be used in exactly the same way as those obtained by using one of the
``io.read_*()`` functions. 

Units
-----

Each parser will (try to) return the "natural" units of each code:

=========== =========   =============== ================== ====================
property    PWscf       CPMD            CP2K               LAMMPS (metal units)
=========== =========   =============== ================== ====================
length      Bohr        Bohr            Angstrom           Angstrom 
energy      Ry          Ha              Ha                 eV 
forces      Ry/Bohr     Ha/Bohr         Ha/Bohr            eV/Angstrom
stress      kbar        kbar            bar[MD], GPa[SCF]  bar
temperature K           K               K                  K 
velocity    -           Bohr/thart (?)  Bohr/thart         Angstrom/ps 
time        tryd        thart           thart              ps
=========== =========   =============== ================== ====================

See :mod:`~pwtools.constants` for `thart` and `tryd`.

For PWscf, we also detect things like "ATOMIC_POSITIONS crystal | alat | bohr"
and transform accordingly. Nevertheless, *always* verify that the units you get
are the ones you expect!

In :class:`~pwtools.crys.Structure` and :class:`~pwtools.crys.Trajectory`, we have 
units eV, Angstrom,...

=========== ==============  ===============================
property    unit            SI
=========== ==============  ===============================
length      Angstrom        (1e-10 m)
energy      eV              (1.602176487e-19 J)
forces      eV / Angstrom
stress      GPa             (not eV/Angstrom**3)
temperature K             
velocity    Angstrom / fs
time        fs              (1e-15 s)
mass        amu             (1.6605387820000001e-27 kg)
=========== ==============  ===============================
