import os, types
import numpy as np
from pwtools import parse, common, pwscf, crys, io, constants

# silently fail if ase is missing b/c it is not a dependency
try:
    from ase.calculators.calculator import FileIOCalculator, kpts2mp
except ImportError:
    pass


class PwtoolsQE(FileIOCalculator):
    """QE calculator, only Pwscf (pw.x).

    ATM, we don't write a ``ase.calculators.calculator.Parameters`` class
    (dict) and have no read() method so I guess that restarts don't work. Only
    simple SCF runs for now.
    
    Examples
    --------
    Define a calculator object::

        >>> calc=PwtoolsQE(label='/path/to/calculation/dir/pw',
        ...                kpts=1/0.35, 
        ...                ecutwfc=80,
        ...                conv_thr=1e-8,
        ...                pp='pbe-n-kjpaw_psl.0.1.UPF',
        ...                pseudo_dir='/home/schmerler/soft/share/espresso/pseudo/espresso/',
        ...                calc_name='my_calc', 
        ...                outdir='/scratch/schmerler/', 
        ...                command='mpirun -np 16 pw.x < pw.in > pw.out')
        >>> at=crys.Structure(...).get_ase_atoms(pbc=True)
        >>> at.set_calculator(calc)
        >>> at.get_potential_energy()
    
    Relation to ASE k-grid tools::

        >>> import numpy as np    
        >>> from ase.calculators.calculator import kpts2mp
        >>> from pwtools.crys import kgrid
        >>> st=crys.Structure(cell=np.diag([5,3,3]), 
        ...                   coords=rand(5,3),
        ...                   symbols=['H']*5) 
        >>> at=st.get_ase_atoms(pbc=True)
        >>> crys.kgrid(st.cell, h=0.35)
        array([4, 6, 6])
        >>> kpts2mp(at, kpts=1/0.35)
        array([4, 6, 6])
    
    """

    default_parameters = dict(
        xc=None,
        smearing=None,
        kpts=5.0,
        charge=0.0,
        )
    implemented_properties = ['energy', 'forces', 'stress']
    command = "pw.x -input pw.in | tee pw.out"

    pwin_templ = """
&control
    calculation = 'scf'
    restart_mode = 'from_scratch',
    prefix = '{calc_name}'
    tstress = .true.
    tprnfor = .true.
    pseudo_dir = '{pseudo_dir}',
    outdir = '{outdir}'
    wf_collect = .false.
/
&system
    ibrav = 0,
    nat = {natoms}, 
    ntyp = {ntyp},
    ecutwfc = {ecutwfc},
    ecutrho = {ecutrho},
    nosym = .false.
/
    occupations = 'smearing'
    smearing = 'mv',
    degauss = 0.005

&electrons
    electron_maxstep = {electron_maxstep}   
    diagonalization = '{diagonalization}'
    mixing_mode = '{mixing_mode}'
    mixing_beta = {mixing_beta}
    conv_thr = {conv_thr}
/
ATOMIC_SPECIES
{atspec}
CELL_PARAMETERS angstrom
{cell}
ATOMIC_POSITIONS crystal
{atpos}
K_POINTS automatic
{kpoints} 0 0 0
    """
    # TODO: Automatic collection of keywords, as in crys.Structure. Use a list
    # of allowed ASE and QE keywords for that for check for non-supported ones.
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 xc=None, smearing=None, atoms=None, kpts=None, label='pw',
                 outdir=None, pseudo_dir=None, ecutwfc=80.0,
                 ecutrho=None,diagonalization='david', mixing_mode='plain',
                 mixing_beta=0.3, electron_maxstep=500, conv_thr=1e-10,
                 pp=None, calc_name='pwscf', **kwargs):
        """
        Parameters
        ----------
        kpts : ASE k-points description
            Examples: ``3.5``, ``[6,6,4]``.
            If a float ``x`` is used, then it is the inverse of the
            k-grid spacing `h` per reciprocal axis 
            as in ``kpts=pwtools.crys.kgrid(struct.cell, h=1/x)``
        calc_name : str
            'prefix' in PWscf
        pp : str or sequence 
            Definition of the pseudopotential and thus `xc`, such as
            'pbe-n-kjpaw_psl.0.1.UPF'. If a string, then ``<symbol>.<pp>`` is
            used for all atom symbols. Needs a file of that name in
            `pseudo_dir`. If a list, e.g. ``['Al.PBE.fhi.UPF',
            'N.PBE.fhi.UPF']``, then this is used for each atom type.
        outdir, pseudo_dir, ecutwfc, ecutrho, diagonalization, mixing_mode,
        mixing_beta, electron_maxstep, conv_thr : as in PWscf, see
            http://www.quantum-espresso.org/wp-content/uploads/Doc/INPUT_PW.html
        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        # ASE keywords
        if smearing is not None:
            raise NotImplementedError("smearing not implemented")
        if xc is not None:
            raise StandardError("please use the `pp` keyword instead of `xc`")
        self.kpts = kpts
        
        # QE keywords
        self.pp = pp
        self.outdir = outdir    
        self.pseudo_dir = pseudo_dir
        self.calc_name = calc_name  # prefix
        self.ecutwfc = ecutwfc      # Ry
        self.ecutrho = 4.0*ecutwfc if ecutrho is None else ecutrho
        self.diagonalization = diagonalization
        self.mixing_mode = mixing_mode
        self.conv_thr = conv_thr    # Ry ?
        self.electron_maxstep = electron_maxstep
        self.mixing_beta = mixing_beta

        # hard-coded <label>.(in|out) == <directory>/<prefix>.(in|out)
        self.pwin = os.path.join(self.directory, self.prefix + '.in')
        self.pwout = os.path.join(self.directory, self.prefix + '.out')
        
        assert self.pp is not None, "set pp"
        assert self.pseudo_dir is not None, "set pseudo_dir"
        assert self.outdir is not None, "set outdir"
        assert self.kpts is not None, "set kpts"

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties,
                                     system_changes)
        struct = crys.atoms2struct(atoms)
        self.cell = common.str_arr(struct.cell)
        self.kpoints = pwscf.kpoints_str_pwin(kpts2mp(atoms, self.kpts))
        if isinstance(self.pp, types.StringType):
            pseudos = ["%s.%s" %(sym, self.pp) for sym in struct.symbols_unique]
        else: 
            assert len(self.pp) == struct.ntypat
            pseudos = []
            for sym in struct.symbols_unique:
                for ppi in self.pp:
                    if ppi.startswith(sym):
                        pseudos.append(ppi)
                        break
        assert len(pseudos) == struct.ntypat
        self.atspec = pwscf.atspec_str(symbols=struct.symbols_unique,
                                       masses=struct.mass_unique,
                                       pseudos=pseudos)
        self.atpos = pwscf.atpos_str(symbols=struct.symbols,
                                     coords=struct.coords_frac)
        self.natoms = struct.natoms
        self.ntyp = struct.ntypat
        
        keys = ['pseudo_dir', 'outdir', 'natoms', 'ecutwfc', 'ecutrho', 'atspec',
                'cell', 'atpos', 'kpoints', 'ntyp', 'calc_name', 'mixing_mode',
                'conv_thr', 'mixing_beta', 'diagonalization', 'electron_maxstep']
        txt = self.pwin_templ.format(**dict((key, getattr(self, key)) for key \
                                             in keys))
        common.file_write(self.pwin, txt)
    
    def read_results(self):
        self.results = {}
        assert os.path.exists(self.pwout), "%s missing" %self.pwout
        st = io.read_pw_scf(self.pwout)
        self.results['energy'] = st.etot
        self.results['forces'] = st.forces
        stress = np.empty(6)
        stress[0] = st.stress[0,0]
        stress[1] = st.stress[1,1]
        stress[2] = st.stress[2,2]
        stress[3] = st.stress[1,2]
        stress[4] = st.stress[0,2]
        stress[5] = st.stress[0,1]
        self.results['stress'] = -stress / constants.eV_by_Ang3_to_GPa
