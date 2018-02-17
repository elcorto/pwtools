import os, types
import numpy as np
from pwtools import parse, common, pwscf, crys, io, constants

# silently fail if ase is missing b/c it is not a dependency
try:
    from ase.calculators.calculator import FileIOCalculator, kpts2mp
except ImportError:
    pass

def stress_pwtools2ase(pwstress):
    """
    Convert (3,3) stress tensor (GPa) to Voigt 6-vector for ASE (eV/Ang^3).
    Note that ASE uses another sign convention (too small unit cell = negative
    pressure instead of positive).

    Parameters
    ----------
    pwstress : (3,3)
        symmetric stress tensor (GPa)
    
    Returns
    -------
    [Pxx, Pyy, Pzz, Pyz, Pxz, Pxy] in eV/Ang^3
    """    
    return -crys.tensor2voigt(pwstress) / constants.eV_by_Ang3_to_GPa


# Have an idea for an even more useless name for this class? Then send me a
# mail. Or a patch! :)
class CalculatorBase(object):
    """Base class for creating calculators. 
    
    Provides methods to automatically dispatch constructor input keywords."""
    def init_params_from_input(self, kwds):
        """Set ``self.parameters = self.default_parameters`` and update with input
        keyword arguments `kwds`. For each key in self.parameters, set
        ``self.<key> = <value>``."""
        allowed_keys = list(self.default_parameters.keys())
        input_keys = list(kwds.keys())
        for k in input_keys:
            if k not in allowed_keys:
                raise Exception("key '%s' not allowed, only: \n%s" %(k,
                                    str(allowed_keys)))
        self.parameters = self.default_parameters
        self.parameters.update(kwds)
        self.__dict__.update(self.parameters)

    def fill_infile_templ(self):
        """Replace all placeholders in ``self.infile_templ``. Use all keys in
        ``self.infile_templ_keys`` as possible placeholders."""
        return self.infile_templ.format(**dict((key, getattr(self, key)) for key \
                                        in self.infile_templ_keys))


class Pwscf(FileIOCalculator, CalculatorBase):
    """Pwscf (pw.x) calculator.

    ATM, we don't write a ``ase.calculators.calculator.Parameters`` class
    (dict) and have no read() method so I guess that restarts don't work. Only
    simple SCF runs for now.
    
    Examples
    --------
    Define a calculator object::

        >>> calc=Pwscf(label='/path/to/calculation/dir/pw',
        ...            kpts=1/0.35, 
        ...            ecutwfc=80,
        ...            conv_thr=1e-8,
        ...            pp='pbe-n-kjpaw_psl.0.1.UPF',
        ...            pseudo_dir='/home/schmerler/soft/share/espresso/pseudo/espresso/',
        ...            calc_name='my_calc', 
        ...            outdir='/scratch/schmerler/', 
        ...            command='mpirun -np 16 pw.x < pw.in > pw.out')
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
        restart=None,
        ignore_bad_restart_file=False,
        atoms=None,
        backup=False, 
        calc_name='pwscf', 
        charge=0.0,
        conv_thr=1e-10,
        diagonalization='david', 
        ecutrho=None,
        ecutwfc=80.0,
        electron_maxstep=500, 
        kpts=5.0,
        label='pw',
        mixing_beta=0.3, 
        mixing_mode='plain',
        outdir=None, 
        pp=None, 
        pseudo_dir=None, 
        smearing=None,
        xc=None,
        command = "pw.x -input pw.in | tee pw.out",
        )
    
    implemented_properties = ['energy', 'forces', 'stress']

    infile_templ = """
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
    def __init__(self, **kwds):
        """
        Parameters
        ----------
        All parameters: ``self.parameters.keys()``
        label : str
            Basename of input and output files (e.g. 'pw') or a path to the
            calculation dir *including* the basename ('/path/to/pw', where
            directory='/path/to' and prefix='pw'). In ``self.command``,
            '<prefix>.in' and '<prefix>.out' are used.
        kpts : ASE k-points description
            Examples: ``3.5``, ``[6,6,4]``.
            If a float ``x`` is used, then it is the inverse of the
            k-grid spacing `h` per reciprocal axis 
            as in ``kpts=pwtools.crys.kgrid(struct.cell, h=1/x)``
        calc_name : str
            'prefix' in PWscf
        pp : str or sequence 
            Definition of the pseudopotential file and thus `xc`. If `pp` is a
            string (e.g. 'pbe-n-kjpaw_psl.0.1.UPF'), then the atom symbols are
            used to build the PP file name
            ``'<atom_symbol>.pbe-n-kjpaw_psl.0.1.UPF'`` for each atom type. Needs
            a file of that name in `pseudo_dir`. If `pp` is a list, e.g.
            ``['Al.pbe-n-kjpaw_psl.0.1.UPF', 'N.pbe-n-kjpaw_psl.0.1.UPF']``,
            then this is used for each atom type.
        outdir, pseudo_dir, ecutwfc, ecutrho, diagonalization, mixing_mode,
        mixing_beta, electron_maxstep, conv_thr : as in PWscf, see
            http://www.quantum-espresso.org/wp-content/uploads/Doc/INPUT_PW.html
        backup : bool
            make backup of old pw.in and pw.out if found, uses
            :func:`~pwtools.common.backup`
        """
        self.init_params_from_input(kwds)
        FileIOCalculator.__init__(self, **kwds)
        
        # ASE keywords
        if self.smearing is not None:
            raise NotImplementedError("smearing not implemented")
        if self.xc is not None:
            raise Exception("please use the `pp` keyword instead of `xc`")
        
        self.ecutrho = 4.0*self.ecutwfc if self.ecutrho is None else self.ecutrho

        # hard-coded <label>.(in|out) == <directory>/<prefix>.(in|out)
        self.infile = os.path.join(self.directory, self.prefix + '.in')
        self.outfile = os.path.join(self.directory, self.prefix + '.out')
        
        self.infile_templ_keys = list(self.parameters.keys()) + ['natoms', 'ntyp', 'atpos',
            'atspec', 'cell', 'kpoints']

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
        if isinstance(self.pp, bytes):
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
        
        if self.backup:
            for fn in [self.infile, self.outfile]:
                if os.path.exists(fn):
                    common.backup(fn)
        common.file_write(self.infile, self.fill_infile_templ())
    
    def read_results(self):
        self.results = {}
        assert os.path.exists(self.outfile), "%s missing" %self.outfile
        st = io.read_pw_scf(self.outfile)
        self.results['energy'] = st.etot
        self.results['forces'] = st.forces
        self.results['stress'] = stress_pwtools2ase(st.stress)


class Lammps(FileIOCalculator, CalculatorBase):
    """
    LAMMPS calculator.

    Examples
    --------
    Define a calculator object::

        >>> calc = Lammps(label='/path/to/calculation/dir/lmp',
        ...               pair_style='tersoff',
        ...               pair_coeff='* * /path/to/potential/dir/AlN.tersoff Al N',
        ...               command='lammps < lmp.in > lmp.out 2>&1',
        ...               ) 
        >>> at=crys.Structure(...).get_ase_atoms(pbc=True)
        >>> at.set_calculator(calc)
        >>> at.get_potential_energy()
    """
       
    default_parameters = dict(
        restart=None,
        ignore_bad_restart_file=False,
        atoms=None,
        label='lmp',
        pair_coeff='* * ./AlN.tersoff Al N',
        pair_style='tersoff',
        backup=False,
        command = "lammps < lmp.in > lmp.out", # also writes 'log.lammps'
        )
    implemented_properties = ['energy', 'forces', 'stress']

    infile_templ = """
clear
units metal 
boundary p p p 
atom_style atomic

# lmp.struct written by pwtools
read_data {structfile}

# interactions 
pair_style {pair_style}
pair_coeff {pair_coeff}

# IO
dump dump_txt all custom 1 {dumpfile} id type xu yu zu fx fy fz
dump_modify dump_txt sort id 

thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                    pe pxx pyy pzz pxy pxz pyz
run 0    
    """

    def __init__(self, **kwds):
        """
        Parameters
        ----------
        All parameters: ``self.parameters.keys()``
        """
        
        self.init_params_from_input(kwds)
        FileIOCalculator.__init__(self, **kwds)
        
        self.infile_templ_keys = list(self.parameters.keys()) + \
            ['prefix', 'dumpfile', 'structfile']
        self.infile = os.path.join(self.directory, self.prefix + '.in')
        self.outfile = os.path.join(self.directory, self.prefix + '.out')
        self.dumpfile = os.path.join(self.directory, self.prefix + '.out.dump')
        self.structfile = os.path.join(self.directory, self.prefix + '.struct')
        self.logfile = os.path.join(self.directory, 'log.lammps')

        
    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties,
                                     system_changes)
        if self.backup:
            for fn in [self.infile, self.outfile, self.dumpfile,
                       self.structfile, self.logfile]:
                if os.path.exists(fn):
                    common.backup(fn)
        common.file_write(self.infile, self.fill_infile_templ())
        io.write_lammps(self.structfile, 
                        crys.atoms2struct(atoms), 
                        symbolsbasename=os.path.basename(self.structfile) + \
                            '.symbols')
    
    def read_results(self):
        self.results = {}
        assert os.path.exists(self.outfile), "%s missing" %self.outfile
        st = io.read_lammps_md_txt(self.outfile)[0]
        self.results['energy'] = st.etot
        self.results['forces'] = st.forces
        self.results['stress'] = stress_pwtools2ase(st.stress)

