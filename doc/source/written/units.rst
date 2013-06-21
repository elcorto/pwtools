Units
=====

We have a lot of machinery in :mod:`~pwtools.parse` to parse PWscf/CPMD/Cp2K
output files. In each parser ``parse.*OutputFile``,
we try to return the "natural" units of each code:

=========== =========   =============== ==================
what        PWscf       CPMD            CP2K
=========== =========   =============== ==================
length      Bohr        Bohr            Angstrom
energy      Ry          Ha              Ha
forces      Ry/Bohr     Ha/Bohr         Ha/Bohr
stress      kbar        kbar            bar[MD], GPa[SCF]
temperature K           K               K
velocity    **          Bohr/thart (?)  Bohr/thart
time        tryd        thart           thart
=========== =========   =============== ==================

See :mod:`~pwtools.constants` for `thart` and `tryd`.


For PWscf, we also detect things like "ATOMIC_POSITIONS crystal | alat | bohr"
and transform accordingly. Nevertheless, *always* verify that the units you get
are the ones you expect!

You can also use the short-cut methods ``io.read*()``, which will return
crys.Structure or crys.Trajectory with units eV, Angstrom,...

=========== ==============  ===============================
what        unit            SI
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
