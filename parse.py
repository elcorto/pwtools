"""
Parser classes for different file formats. Input- and output files.
===================================================================
 
We need the following basic Unix tools installed:

| grep/egrep
| sed
| awk
| tail
| wc 
| ...

The tested egrep versions don't know the ``\s`` character class for
whitespace as sed, Perl, Python or any other sane regex implementation
does. Use ``[ ]`` instead.

Using Parsing classes
---------------------

All parsing classes::

    Pw*OutputFile
    Cpmd*OutputFile

are derived from FlexibleGetters -> FileParser -> {Structure,Trajectory}FileParser

As a general rule: If a getter (self.get_<attr>() or self._get_<attr>_raw()
cannot find anything in the file, it returns None. All getters which depend
on it will also return None.

* After initialization
      pp = SomeParsingClass(<filename>) 
  all attrs whoose name is in pp.attr_lst will be set to None.

* parse() will invoke self.try_set_attr(<attr>), which does 
      self.<attr> = self.get_<attr>() 
  for each <attr> in self.attr_lst, thus setting self.<attr> to a defined
  value: None if nothing was found in the file or not None else

* All getters get_<attr>() will do their parsing action, possibly
  looking for a file self.filename, regardless of the fact that the attribute
  self.<attr> may already be defined (e.g. if parse() has been called before).

* For interactive use (you need <attr> only once), prefer get_<attr>() over
  parse().

* Use dump('foo.pk') only for temporary storage and fast re-reading. Use
  pwtool.common.cpickle_load(foo.pk). See also the FileParser.load() docstring.

* Keep the original parsed output file around (self.filename), avoid
  get_txt().

* Use relative paths in <filename>.

* If loading a dump()'ed pickle file from disk,
      pp=common.cpickle_load(...)
  then use direct attr access
      pp.<attr>
  instead of 
      pp.get_<attr>()
  b/c latter would simply parse self.filname again.

For debugging, we still have many getters which produce redundant
information, e.g.

    | cell + cryst_const
    | _<attr>_raw + <attr> (where <attr> = cell, forces, ...)
    | ...

especially in MD parsers, not so much in StructureFileParser drived
classes. If parse() is used, all this information retrieved and stored.

* All parsers try to return the default units of the program output, e.g. Ry,
  Bohr, tryd for PWscf; Ha, Bohr, thart for Abinit and CPMD.

* Use get_struct() / get_traj() to get a Structure / Trajectory object with
  pwtools standard units (eV, Ang, fs).

Using parse():    

Pro:

* Simplicity. *All* getters are called when parse() is
  invoked. You get it all.
* In theory, you can delete the file pointed to by self.filename, assuming
  all getters have extracted all information that you need.

Con:      

* The object is full of (potentially big) arrays holding redundant
  information. Thus, the dump()'ed file may be large. 
* Parsing may be slow if each getter (of possibly many) is called.

Using get_<attr>():

Pro: 

* You only parse what you really need.

Con:

* self.<attr> will NOT be set, since get_<attr>() only returns <attr> but
  doesn't set self.<attr> = self.get_<attr>(), so dump() would save an
  "empty" file.

Note on get_txt():

If a file parser calls get_txt(), then this has these effects: (1) It can
slow down parsing for big files and (2) the saved binary file (by using
dump()) will have at least the size of the text file. While the original file
could then be deleted in theory, the dump()'ed file becomes unwieldly to work
with.
"""

import re, sys, os
from math import acos, pi, sin, cos, sqrt
from itertools import izip
from cStringIO import StringIO
import types
import warnings

import numpy as np

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    warnings.warn("Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work.")

# XML parser
try:
    from BeautifulSoup import BeautifulStoneSoup
except ImportError:
    warnings.warn("Cannot import BeautifulSoup. " 
    "Parsing XML/HTML/CML files will not work.")

from pwtools import common, constants, regex, crys, atomic_data, num, arrayio
from pwtools.verbose import verbose
from pwtools.base import FlexibleGetters
from pwtools.constants import Ry, Ha, eV, Bohr, Angstrom, thart, Ang, fs
from pwtools.crys import UnitsHandler
com = common


#-----------------------------------------------------------------------------
# General helpers
#-----------------------------------------------------------------------------

def int_from_txt(txt):
    if txt.strip() == '':
        return None
    else:
        return int(txt)

def float_from_txt(txt):
    if txt.strip() == '':
        return None
    else:
        return float(txt)

def nstep_from_txt(txt):
    if txt.strip() == '':
        return 0
    else:
        return int(txt)

def traj_from_txt(txt, shape, axis=-1, dtype=np.float):
    """Used for 3d trajectories where the exact shape must be known, e.g.
    (nstep,N,3,nstep) where N=3 (cell, stress) or N=natoms (coords, forces,
    ...). 
    """
    if txt.strip() == '':
        return None
    else:
        ret = arrayio.readtxt(StringIO(txt), axis=axis, shape=shape, dtype=dtype)
        return ret

def arr1d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        ret = np.atleast_1d(np.loadtxt(StringIO(txt), dtype=dtype))
        return ret

def arr2d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        ret = np.atleast_2d(np.loadtxt(StringIO(txt), dtype=dtype))
        return ret

def axis_lens(seq, axis=0):
    """Return length of `axis` of all arrays in `seq`. 
    
    If an entry in `seq` is None instead of an array, return 0. All arrays must
    have at least axis+1 dimensions, of course (i.e. axis=1 -> all arrays at
    least 2d).

    Examples
    --------
    >>> axis_lens([arange(100), np.array([1,2,3]), None, rand(5,3)])
    [100, 3, 0, 5]
    """
    ret = []
    for xx in seq:
        try:
            # handle None
            ret.append(xx.shape[axis])
        except AttributeError:
            ret.append(0)
    return ret            

#-----------------------------------------------------------------------------
# Parsers
#-----------------------------------------------------------------------------

class FileParser(FlexibleGetters):
    """Base class for file parsers.
    
    All getters are called in the default self.parse() which can be overridden
    in derived classes.
    """
    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : str, name of the file to parse
        """
        FlexibleGetters.__init__(self)
        self.filename = filename
        if self.filename is not None:
            self.fd = open(self.filename, 'r')
            self.basedir = os.path.dirname(self.filename)
        else:
            self.fd = None
            self.basedir = None
        self.parse_called = False    

    def __del__(self):
        """Destructor. If self.fd has not been closed yet (in self.parse()),
        then do it here, eventually."""
        self.close_file()
    
    def _is_file_open(self):
        if self.is_set_attr('fd'):
            return (not self.fd.closed)
        else:
            return False

    def close_file(self):
        if self._is_file_open():
            self.fd.close()
    
    def parse(self):
        self.set_all()
        self.close_file()
        self.parse_called = True

    def get_txt(self):
        if self._is_file_open():
            self.fd.seek(0)    
            return self.fd.read().strip()
        else:
            raise StandardError("self.fd is None or closed")


class StructureFileParser(FileParser, UnitsHandler):
    """Base class for single-structure parsers.
    
    Derived from UnitsHandler, but we actually only use it for
    self.update_units() to set self.units, which are handed over to
    Structure(). The parser itself doesn't apply_units() to itself.
    """
    # XXX Use Container().attr_lst? Must create Container instance first b/c
    # attr_lst is created in Container.__init__() . Maybe not worth the
    # trouble?
    Container = crys.Structure
    cont_attr_lst = [\
        'cell',
        'coords',
        'coords_frac',
        'cryst_const',
        'etot',
        'forces',
        'stress',
        'symbols',
        ]
    default_units = {}        

    def __init__(self, filename=None, units=None):
        FileParser.__init__(self, filename=filename)
        UnitsHandler.__init__(self)
        # Parsers can have default units ...
        self.update_units(self.default_units)
        # ... which are updated with user input. The resulting self.units is
        # passed to Container.
        self.update_units(units)
        # init all attrs to None which go into Structure()
        self.init_attr_lst(self.cont_attr_lst)            
    
    def get_cont(self):
        if not self.parse_called:
            self.parse()
        cont = self.Container(set_all_auto=False, units=self.units)
        for attr_name in self.cont_attr_lst:
            setattr(cont, attr_name, getattr(self, attr_name))
        cont.set_all()
        return cont
   
    def apply_units(self):
        raise NotImplementedError("don't use that in parsers")
    
    def get_struct(self):
        return self.get_cont()

class TrajectoryFileParser(StructureFileParser):
    """Base class for MD-like parsers."""
    Container = crys.Trajectory
    timeaxis = crys.Trajectory(set_all_auto=False).timeaxis
    # XXX use Trajectory().attr_lst ??
    cont_attr_lst = [\
        'cell',
        'coords',
        'coords_frac',
        'cryst_const',
        'ekin',
        'etot',
        'forces',
        'stress',
        'symbols',
        'temperature',
        'timestep',
        ]
    
    def get_struct(self):
        raise NotImplementedError("use get_traj()")

    def get_traj(self):
        return self.get_cont()


class CifFile(StructureFileParser):
    """Parse Cif file. Uses PyCifRW [1]_.

    References
    ----------
    .. [1] http://pycifrw.berlios.de/
    """        
    def __init__(self, filename=None, block=None, *args, **kwds):
        """        
        Parameters
        ----------
        filename : name of the input file
        block : data block name (i.e. 'data_foo' in the Cif file -> 'foo'
            here). If None then the first data block in the file is used.
        
        """
        self.block = block
        StructureFileParser.__init__(self, filename=filename, *args, **kwds)
        # only the ones for which we have getters
        self.attr_lst = [\
            'coords_frac',
            'coords',
            'symbols',
            'cryst_const',
            ]
        self.init_attr_lst()      
    
    def cif_str2float(self, st):
        """'7.3782(7)' -> 7.3782"""
        if '(' in st:
            st = re.match(r'(' + regex.float_re  + r')(\(.*)', st).group(1)
        return float(st)

    def cif_clear_atom_symbol(self, st, rex=re.compile(r'([a-zA-Z]+)([0-9+-]*)')):
        """Remove digits and "+,-" from atom names. 
        
        Examples
        --------
        >>> cif_clear_atom_symbol('Al1')
        'Al'
        """
        return rex.match(st).group(1)
    
    def _get_cif_dct(self):
        # celldm from a,b,c and alpha,beta,gamma
        # alpha = angbe between b,c
        # beta  = angbe between a,c
        # gamma = angbe between a,b
        self.try_set_attr('_cif_block')
        cif_dct = {}
        for x in ['a', 'b', 'c']:
            what = '_cell_length_' + x
            cif_dct[x] = self.cif_str2float(self._cif_block[what])
        for x in ['alpha', 'beta', 'gamma']:
            what = '_cell_angle_' + x
            cif_dct[x] = self.cif_str2float(self._cif_block[what])
        return cif_dct
    
    def _get_cif_block(self):
        cf = pycifrw_CifFile.ReadCif(self.filename)
        if self.block is None:
            cif_block = cf.first_block()
        else:
            cif_block = cf['data_' + self.block]
        return cif_block
    
    def get_coords_frac(self):
        if self.check_set_attr('_cif_block'):
            if self._cif_block.has_key('_atom_site_fract_x'):
                arr = np.array([map(self.cif_str2float, [x,y,z]) for x,y,z in izip(
                                    self._cif_block['_atom_site_fract_x'],
                                    self._cif_block['_atom_site_fract_y'],
                                    self._cif_block['_atom_site_fract_z'])])
                return arr                                    
            else:
                return None
        else:
            return None
        
    def get_coords(self):
        if self.check_set_attr('_cif_block'):
            if self._cif_block.has_key('_atom_site_Cartn_x'):
                arr = np.array([map(self.cif_str2float, [x,y,z]) for x,y,z in izip(
                                    self._cif_block['_atom_site_Cartn_x'],
                                    self._cif_block['_atom_site_Cartn_y'],
                                    self._cif_block['_atom_site_Cartn_z'])])
                return arr                                    
            else:
                return None
        else:
            return None

    def get_symbols(self):
        self.try_set_attr('_cif_block')
        try_lst = ['_atom_site_type_symbol', '_atom_site_label']
        for entry in try_lst:
            if self._cif_block.has_key(entry):
                return map(self.cif_clear_atom_symbol, self._cif_block[entry])
        return None                
    
    def get_cryst_const(self):
        self.try_set_attr('_cif_dct')
        return np.array([self._cif_dct[key] for key in \
            ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
    

class PDBFile(StructureFileParser):
    """Very very simple pdb file parser. 
    
    Extract only ATOM/HETATM and CRYST1 (if present) records. If you want smth
    serious, check biopython.
    
    Notes
    -----
    self.cryst_const :
        If no CRYST1 record is found, this is None.
    
    parsing:
        We use regexes which may not work for more complicated ATOM records. We
        don't use the strict column numbers for each field as stated in the PDB
        spec.
    """
    # Notes:
    #
    # Grep atom symbols and coordinates in Angstrom ([A]) from PDB file.
    # Note that for the atom symbols, we do NOT use the columns 77-78
    # ("Element symbol"), b/c that is apparently not present in all the
    # files which we currently use. Instead, we use the columns 13-16, i.e.
    # "Atom name". Note that in general this is not the element symbol.
    #
    # From the PDB spec v3.20:
    #
    # ATOM record:
    #
    # COLUMNS       DATA  TYPE    FIELD        DEFINITION
    # -------------------------------------------------------------------------------------
    #  1 -  6       Record name   "ATOM  "
    #  7 - 11       Integer       serial       Atom  serial number.
    #  13 - 16      Atom          name         Atom name.
    #  17           Character     altLoc       Alternate location indicator.
    #  18 - 20      Residue name  resName      Residue name.
    #  22           Character     chainID      Chain identifier.
    #  23 - 26      Integer       resSeq       Residue sequence number.
    #  27           AChar         iCode        Code for insertion of residues.
    #  31 - 38      Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    #  39 - 46      Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    #  47 - 54      Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    #  55 - 60      Real(6.2)     occupancy    Occupancy.
    #  61 - 66      Real(6.2)     tempFactor   Temperature  factor.
    #  77 - 78      LString(2)    element      Element symbol, right-justified.
    #  79 - 80      LString(2)    charge       Charge  on the atom.
    #
    # CRYST1 record:
    # 
    # COLUMNS       DATA  TYPE    FIELD          DEFINITION
    # -------------------------------------------------------------
    #  1 -  6       Record name   "CRYST1"
    #  7 - 15       Real(9.3)     a              a (Angstroms).
    # 16 - 24       Real(9.3)     b              b (Angstroms).
    # 25 - 33       Real(9.3)     c              c (Angstroms).
    # 34 - 40       Real(7.2)     alpha          alpha (degrees).
    # 41 - 47       Real(7.2)     beta           beta (degrees).
    # 48 - 54       Real(7.2)     gamma          gamma (degrees).
    # 56 - 66       LString       sGroup         Space  group.
    # 67 - 70       Integer       z              Z value.
    #
    def __init__(self, filename=None, *args, **kwds):
        StructureFileParser.__init__(self, filename=filename, *args, **kwds)
        # only the ones for which we have getters
        self.attr_lst = [\
            'coords',
            'symbols',
            'cryst_const',
            ]
        self.init_attr_lst()      
    
    def _get_coords_data(self):
        if self.check_set_attr('txt'):
            pat = r'(ATOM|HETATM)[\s0-9]+([A-Za-z]+)[\sa-zA-Z0-9]*' + \
                r'[\s0-9]+((\s+'+ regex.float_re + r'){3}?)'
            # array of string type            
            return np.array([[m.group(2)] + m.group(3).split() for m in \
                             re.finditer(pat,self.txt)])
        else:
            return None
    
    def get_symbols(self):
        # list of strings (system:nat,) 
        # Fix atom names, e.g. "AL" -> Al. Note that this is only needed b/c we
        # use the "wrong" column "Atom name".
        self.try_set_attr('_coords_data')
        symbols = []
        for sym in self._coords_data[:,0]:
            if len(sym) == 2:
                symbols.append(sym[0] + sym[1].lower())
            else:
                symbols.append(sym)
        return symbols

    def get_coords(self):
        self.try_set_attr('_coords_data')
        # float array, (system:nat, 3)
        return self._coords_data[:,1:].astype(float)        
    
    def get_cryst_const(self):
        # grep CRYST1 record, extract only crystallographic constants
        # example:
        # CRYST1   52.000   58.600   61.900  90.00  90.00  90.00  P 21 21 21   8
        #          a        b        c       alpha  beta   gamma  |space grp|  z-value
        if self.check_set_attr('txt'):
            pat = r'CRYST1\s+((\s+' + regex.float_re + r'){6}).*'
            match = re.search(pat, self.txt)
            return np.array(match.group(1).split()).astype(float)
        else:
            return None


class CMLFile(StructureFileParser):
    """Parse Chemical Markup Language files (XML-ish format). This file format
    is used by avogadro."""
    def __init__(self, filename=None, *args, **kwds):
        StructureFileParser.__init__(self, filename=filename, *args, **kwds)
        # only the ones for which we have getters
        self.attr_lst = [\
            'coords_frac',
            'symbols',
            'cryst_const',
            ]
        self.init_attr_lst()      
    
    def _get_soup(self):
        return BeautifulStoneSoup(open(self.filename).read())        

    def _get_atomarray(self):
        self.try_set_attr('_soup')
        # ret: list of Tag objects:
        # <atomarray>
        #    <atom id="a1" ...>
        #    <atom id="a2" ...>
        #    ...
        # </atomarray>
        # ==>
        # [<atom id="a1" ...>, <atom id="a2" ...>, ...]
        return self._soup.find('atomarray').findAll('atom')
    
    def get_coords_frac(self):
        self.try_set_attr('_atomarray')
        return np.array([[float(entry.get('xfract')), 
                          float(entry.get('yfract')),
                          float(entry.get('zfract'))] \
                          for entry in self._atomarray])
    
    def get_symbols(self):
        self.try_set_attr('_atomarray')
        return [str(entry.get('elementtype')) for entry in self._atomarray]
    
    def get_cryst_const(self):
        self.try_set_attr('_soup')
        crystal = self._soup.find('crystal')            
        return np.array([crystal.find('scalar', title="a").string,
                         crystal.find('scalar', title="b").string,
                         crystal.find('scalar', title="c").string,
                         crystal.find('scalar', title="alpha").string,
                         crystal.find('scalar', title="beta").string,
                         crystal.find('scalar', title="gamma").string]\
                         ).astype(float)
    

class PwSCFOutputFile(StructureFileParser):
    """Parse a pw.x SCF output file (calculation='scf').
    
    Some getters (_get_<attr>_raw) work for MD-like output, too. Here in the
    SCF case, only the first item along the time axis is returned and should
    only be used on calculation='scf' output.
    
    SCF output files don't have an ATOMIC_POSITIONS block. We need to parse the
    block below, which can be found at the top the file (cartesian, divided by
    alat). From that, we also get symbols::

        Cartesian axes
        
          site n.     atom                  positions (a_0 units)
              1           Al  tau(  1) = (  -0.0000050   0.5773532   0.0000000  )
              2           Al  tau(  2) = (   0.5000050   0.2886722   0.8000643  )
              3           N   tau(  3) = (  -0.0000050   0.5773532   0.6208499  )
              4           N   tau(  4) = (   0.5000050   0.2886722   1.4209142  )
    
    Many quantities in PWscf's output files are always in units of the lattice
    vector "a" (= a_0 = celldm1 = "alat" [Bohr]), i.e. divided by that value,
    which is usually printed in the output in low precision::

         lattice parameter (a_0)   =       5.8789  a.u.
    
    You can parse that value with ``get_alat(use_alat=True)``. We do that by
    default (``use_alat=True``) b/c this is what most people will expect if
    they just call the parser on some file. Then, we multiply all relevent
    quantities with dimension length with the alat value from pw.out
    automatically.

    If ``use_alat=False``, we use ``alat=1.0``, i.e. all length quantities
    which are "in alat units" are returned exactly as found in the file, which
    is the same behavior as in all other parsers. Unit conversion happens only
    when we pass things to Structure / Trajectory using self.units. 

    If you need/want to use another alat (i.e. a value with more precision), 
    then you need to explicitely provide that value and use ``use_alat=False``::

    >>> alat = 1.23456789 # Bohr
    >>> pp = PwSCFOutputFile(..., use_alat=False, units={'length': alat*Bohr/Ang})
    >>> st = pp.get_struct()

    That will overwrite ``default_units['length'] = Bohr/Ang``, which is used to
    convert all PWscf length [Bohr] to [Ang] when passing things to Trajectory. 

    In either case, all quantities with a length unit or derived from such a
    quantitiy, e.g.

        | cell
        | cryst_const
        | coords
        | coords_frac
        | volume
        | ...
    
    will be correct (up to alat's precision).
    
    All getters return PWscf standard units (Ry, Bohr, ...).

    It is a special case for PWscf that a parser class may modify values parsed
    from a file (multiply by alat if use_alat=True, etc) *before* they are
    passed over to Structure / Trajectory  b/c otherwise the numbers would be
    pretty useless, unless you use `units` explicitely. To get an object with
    pwtools standard units (eV, Angstrom, ...), use get_struct().

    Notes
    -----
    Total force: Pwscf writes a "Total Force" after the "Forces acting on
    atoms" section . This value a UNnormalized RMS of the force matrix
    (f_ij, i=1,natoms j=1,2,3) printed. According to .../PW/forces.f90,
    variable "sumfor", the "Total Force" is ``sqrt(sum_ij f_ij^2)``.
    Use ``crys.rms(self.forces)`` (for PwSCFOutputFile) or
    ``crys.rms3d(self.forces, axis=self.timeaxis)`` (for PwMDOutputFile)
    instead.
    
    Verbose force printing: When using van der Waals (``london=.true.``) or
    ``verbosity='high'``, then more than one force block (natoms,3) is printed.
    In that case, we assume the first block to be the sum of all force
    contributions and that will end up in ``self.forces``. Each subsequent
    block is discarded from ``self.forces``. However, you may use
    ``self._forces_raw`` (see ``self._get_forces_raw()``) to obtain all forces,
    which will have the shape (N*natoms). The forces blocks will be in the
    following order:
    
    =====================   =====================     =======================
    ``london=.true.``       ``verbosity='high'``      ``verbosity='high'`` + 
                                                      ``london=.true.`` 
    =====================   =====================     =======================
    sum                     sum                       sum          
    vdw                     non-local                 non-local
    \                       ionic                     ionic
    \                       local                     local
    \                       core                      core
    \                       Hubbard                   Hubbard
    \                       SCF correction            SCF correction
    \                       \                         vdw  
    =====================   =====================     =======================

    Note that this order may change with QE versions, check your output file!
    Tested w/ QE 4.3.2 .
    """
    # self.timeaxis: This is the hardcoded time axis. It must be done
    #     this way b/c getters returning a >2d array cannot determine the shape
    #     of the returned array auttomatically based on the self.timeaxis
    #     setting alone. If you want to change this, then manually fix the
    #     "shape" kwarg to arrayio.readtxt() in all getters which return a 3d array.
    default_units = \
        {'energy': Ry / eV, # Ry -> eV
         'length': Bohr / Angstrom, # Bohr -> Angstrom
         'forces': Ry / eV * Angstrom / Bohr, # Ry / Bohr -> eV / Angstrom
         'stress': 0.1, # kbar -> GPa
        } 
    def __init__(self, filename=None, use_alat=True, **kwds):
        StructureFileParser.__init__(self, filename=filename, **kwds)
        self.timeaxis = crys.Trajectory(set_all_auto=False).timeaxis
        self.attr_lst = [\
            'coords',
            'symbols',
            'stress',
            'etot',
            'forces',
            'nstep_scf',
            'cell',
            'natoms',
            'nkpoints',
            'scf_converged',
            ]
        self.use_alat = use_alat            
        self.init_attr_lst()

    def _get_stress_raw(self):
        verbose("getting _stress_raw")
        key = 'P='
        cmd = "grep %s %s | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e %s -e '--'| \
              awk '{printf $4\"  \"$5\"  \"$6\"\\n\"}'" \
              %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(nstep,3,3),
                             axis=self.timeaxis)              

    def _get_etot_raw(self):
        verbose("getting _etot_raw")
        cmd =  r"grep '^!' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def _get_forces_raw(self):
        verbose("getting _forces_raw")
        self.try_set_attr('natoms')
        natoms = self.natoms
        # nstep: get it from outfile b/c the value in any input file will be
        # wrong if the output file is a concatenation of multiple smaller files
        key = r'Forces\s+acting\s+on\s+atoms.*$'
        cmd = r"egrep '%s' %s | wc -l" %(key.replace(r'\s', r'[ ]'), self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        # Need to split traj_from_txt() up into loadtxt() + arr2d_to_3d() b/c
        # we need to get `nlines` first without an additional "grep ... | wc
        # -l".
        cmd = "grep 'atom.*type.*force' %s \
            | awk '{print $7\" \"$8\" \"$9}'" %self.filename
        arr2d = np.loadtxt(StringIO(com.backtick(cmd)))
        nlines = arr2d.shape[0]
        # nlines_block = number of force lines per step = N*natoms
        nlines_block = nlines / nstep
        assert nlines_block % natoms  == 0, ("nlines_block forces doesn't "
            "match natoms")
        return arrayio.arr2d_to_3d(arr2d,
                                   shape=(nstep,nlines_block,3), 
                                   axis=self.timeaxis)     
    
    def _get_nstep_scf_raw(self):
        verbose("getting _nstep_scf_raw")
        cmd = r"grep 'convergence has been achieved in' %s | awk '{print $6}'" \
            %self.filename
        return arr1d_from_txt(com.backtick(cmd), dtype=int)

    def _get_coords_symbols(self):
        """Grep start coords and symbols from pw.out header. This is always in
        cartesian alat units (i.e. divided by alat) and printed with low
        precision.
        """
        verbose("getting start coords")
        self.try_set_attr('natoms')
        natoms = self.natoms
        # coords
        cmd = r"egrep -A%i 'site.*atom.*positions.*units.*\)' %s | tail -n%i | \
              sed -re 's/.*\((.*)\)/\1/g'" \
              %(natoms, self.filename, natoms)
        coords = arr2d_from_txt(com.backtick(cmd))
        cmd = r"egrep -A%i 'site.*atom.*positions.*units.*\)' %s | tail -n%i | \
              awk '{print $2}'" \
              %(natoms, self.filename, natoms)
        symbols = com.backtick(cmd).strip().split()
        return {'coords': coords, 'symbols': symbols}
    
    def _get_cell_2d(self):
        """Start cell [Bohr] if self.alat in [Bohr].
        
        Grep start cell from pw.out, multiply by alat.
        
        The cell in pw.out is always in alat units (divided by alat) but
        printed with much less precision compared to the input file. If you
        need this information for further calculations, use the input file
        value."""
        verbose("getting start cell parameters")
        if self.check_set_attr('alat'):
            cmd = "egrep -A3 'crystal.*axes.*units.*(a_0|alat)' %s | tail -n3 | \
                   awk '{print $4\" \"$5\" \"$6}'" %(self.filename)
            ret = arr2d_from_txt(com.backtick(cmd))
            return ret * self.alat if ret is not None else ret
        else:
            return None
    
    def get_alat(self, use_alat=None):
        """Lattice parameter "alat" [Bohr]. If use_alat or self.use_alat is
        False, return 1.0, i.e. disbale alat.
        
        Parameters
        ----------
        use_alat : bool
            Set use_alat and override self.use_alat.
        """
        use_alat = self.use_alat if use_alat is None else use_alat
        if use_alat:
            cmd = r"grep -m1 'lattice parameter' %s | \
                sed -re 's/.*=(.*)\s+a\.u\./\1/'" %self.filename
            return float_from_txt(com.backtick(cmd))            
        else:
            return 1.0

    def get_coords(self):
        """Cartesian start coords [Bohr] if self.alat in [Bohr]."""
        verbose("getting coords")
        if self.check_set_attr_lst(['_coords_symbols', 'alat']):
            return self._coords_symbols['coords'] * self.alat
        else:
            return None

    def get_symbols(self):
        verbose("getting symbols")
        if self.check_set_attr('_coords_symbols'):
            return self._coords_symbols['symbols']
        else:
            return None
    
    def get_stress(self):
        """Stress tensor [kbar]."""
        return self.raw_slice_get('stress', sl=0, axis=self.timeaxis)

    def get_etot(self):
        """Total enery [Ry]."""
        return self.raw_slice_get('etot', sl=0, axis=0)
    
    def get_forces(self):
        """Forces [Ry / Bohr]."""
        if self.check_set_attr('natoms'):
            # Assume that the first forces block printed are the forces on the
            # ions. Skip vdw forces or whatever else is printed after that.
            # Users can use self._forces_raw if they want and know what is
            # printed in which order in the output file.
            forces = self.raw_slice_get('forces', sl=0, axis=self.timeaxis)
            return forces[:self.natoms,:]
        else:
            return None

    def get_nstep_scf(self):
        return self.raw_slice_get('nstep_scf', sl=0, axis=0)
    
    def get_cell(self):
        """Start cell [Bohr]."""
        if self.check_set_attr('_cell_2d'):
            return self._cell_2d

    def get_natoms(self):
        verbose("getting natoms")
        cmd = r"grep -m 1 'number.*atoms/cell' %s | \
              sed -re 's/.*=\s+([0-9]+).*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_nkpoints(self):
        verbose("getting nkpoints")
        cmd = r"grep -m 1 'number of k points=' %s | \
            sed -re 's/.*points=\s*([0-9]+)\s*.*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))

    def get_scf_converged(self):
        verbose("getting scf_converged")
        cmd = "grep 'convergence has been achieved in.*iterations' %s" %self.filename
        if com.backtick(cmd).strip() != "":
            return True
        else:
            return False
    

class PwMDOutputFile(TrajectoryFileParser, PwSCFOutputFile):
    """Parse pw.x MD-like output. 
    
    Tested so far: md, relax, vc-relax. For vc-md, see PwVCMDOutputFile. 

    Notes on units for PwSCFOutputFile, esp. alat, apply here as well.

    Additionally: ATOMIC_POSITIONS and CELL_PARAMETERS can have an optional
    "unit": [None (empty string ''), 'bohr', 'angstrom', 'alat']. In each case,
    the quantity is converted to Bohr, which is PWscf's default length, and
    later to Ang if default_units['length'] = Bohr / Ang. In case of 'alat', it
    is assumed that get_alat() returns alat in Bohr. Anything else is up to
    `units` + use_alat=False .
    """
    def __init__(self, filename=None, use_alat=True, **kwds):
        # update default_units *before* calling base class' __init__, where
        # self.units is assembled from default_units
        self.default_units.update({'time': constants.tryd / constants.fs})
        TrajectoryFileParser.__init__(self, filename=filename, **kwds)
        self.attr_lst = [\
            'coords',
            'coords_frac',
            'symbols',
            'stress',
            'etot',
            'ekin',
            'temperature',
            'forces',
            'nstep_scf',
            'cell',
            'natoms',
            'nkpoints',
            'timestep',
            ]
        self.init_attr_lst()
        self.use_alat = use_alat            
    
    
    def _get_block_header_unit(self, key):
        """Parse things like 
            
            ATOMIC_POSITIONS          -> None
            ATOMIC_POSITIONS unit     -> unit
            ATOMIC_POSITIONS (unit)   -> unit
            ATOMIC_POSITIONS {unit}   -> unit
        
        Parameters
        ----------
        key : str (e.g. 'ATOMIC_POSITIONS')
        """
        assert key not in ['', None], "got illegal string"
        cmd = 'grep -m1 %s %s' %(key, self.filename)
        tmp = com.backtick(cmd).strip()
        for sym in ['(', ')', '{', '}']:
            tmp = tmp.replace(sym, '')
        tmp = tmp.split()
        if len(tmp) < 2:
            return None
        else:
            return tmp[1].split('=')[0]
    
    def _get_coords(self):
        """Parse ATOMIC_POSITIONS block. Unit is handled by get_coords_unit()."""
        verbose("getting _coords")
        if self.check_set_attr('natoms'):
            self.try_set_attr('natoms')
            natoms = self.natoms
            # nstep: get it from outfile b/c the value in any input file will be
            # wrong if the output file is a concatenation of multiple smaller files
            key = 'ATOMIC_POSITIONS'
            cmd = 'grep %s %s | wc -l' %(key, self.filename)
            nstep = nstep_from_txt(com.backtick(cmd))
            # coords
            cmd = "grep -A%i '%s' %s | grep -v -e %s -e '--' | \
                  awk '{printf $2\"  \"$3\"  \"$4\"\\n\"}'" \
                  %(natoms, key, self.filename, key)
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(nstep,natoms,3),
                                 axis=self.timeaxis)
        else:
            return None

    def _get_cell(self):
        """Parse CELL_PARAMETERS block."""
        verbose("getting _cell")
        # nstep
        key = 'CELL_PARAMETERS'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        # cell            
        cmd = "grep -A3 %s %s | grep -v -e %s -e '--'" %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(nstep,3,3),
                             axis=self.timeaxis)

    def _match_nstep(self, arr):
        """Get nstep from _coords.shape[0] and return the last nstep steps from
        the array `arr` along self.timeaxis.
        
        Used to match forces,stress,... etc to coords b/c for MDs, the former
        ones are always one step longer b/c of stuff printed in the first SCF
        loop before the MD starts."""
        if arr is not None:
            if self.check_set_attr('_coords'):
                nstep = self._coords.shape[self.timeaxis]
                if arr.shape[self.timeaxis] > nstep:
                    return num.slicetake(arr, slice(-nstep,None,None), 
                                         axis=self.timeaxis)
                else:
                    return arr
            else:
                return None
        else: 
            return None

    def get_coords_unit(self):
        verbose("getting coords_unit")
        return self._get_block_header_unit('ATOMIC_POSITIONS')
    
    def get_cell_unit(self):
        verbose("getting cell_unit")
        return self._get_block_header_unit('CELL_PARAMETERS')

    def get_coords(self):
        """Cartesian coords [Bohr]. If coords_unit='alat', then [Bohr] if
        self.alat in [Bohr]."""
        if self.check_set_attr_lst(['_coords', 'coords_unit', 'alat']):
            if self.coords_unit in ['bohr', None]:
                return self._coords
            elif self.coords_unit == 'alat':
                return self._coords * self.alat
            elif self.coords_unit == 'angstrom':
                return self._coords * Angstrom / Bohr
            else:
                return None
        else:
            return None
    
    def get_cell(self):
        """Cell [Bohr]. Return 3d array from CELL_PARAMETERS or 2d array
        self._cell_2d . If coords_unit='alat', then [Bohr] if
        self.alat in [Bohr]."""
        if self.check_set_attr_lst(['_cell', 'cell_unit', 'alat']):
            if self.cell_unit in ['bohr', None]:
                return self._cell
            elif self.cell_unit == 'alat':
                return self._cell * self.alat
            elif self.cell_unit == 'angstrom':
                return self._cell * Angstrom / Bohr
            else:
                return None
        elif self.check_set_attr('_cell_2d'):                
                return self._cell_2d # Bohr
        else:
            return None
    
    def get_coords_frac(self):
        """Fractional coords."""
        if self.check_set_attr_lst(['_coords', 'coords_unit']):
            if self.coords_unit == 'crystal':
                return self._coords
            else:
                return None
        else:
            return None
    
    def get_ekin(self):
        """Ion kinetic energy [Ry]."""
        verbose("getting ekin")
        cmd = r"grep 'kinetic energy' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))

    def get_temperature(self):
        """Temperature [K]"""
        verbose("getting temperature")
        cmd = r"egrep 'temperature[ ]*=' %s " %self.filename + \
              "| sed -re 's/.*temp.*=\s*(" + regex.float_re + \
              r")\s*K/\1/'"
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_timestep(self):
        """Time step [tryd]."""
        cmd = r"grep -m1 'Time.*step' %s | sed -re \
              's/.*step\s+=\s+(.*)a.u..*/\1/'" %self.filename
        return float_from_txt(com.backtick(cmd))
    
    def get_stress(self):
        """Stress tensor [kbar]."""
        return self._match_nstep(self.raw_return('stress'))

    def get_etot(self):
        """[Ry] """
        return self._match_nstep(self.raw_return('etot'))
    
    def get_forces(self):
        """[Ry / Bohr] """
        if self.check_set_attr('natoms'):
            forces = self._match_nstep(self.raw_return('forces'))
            return forces[:,:self.natoms,:]
        else:
            return None
    
    def get_nstep_scf(self):
        return self.raw_return('nstep_scf')


class PwVCMDOutputFile(PwMDOutputFile):
    def __init__(self, *args, **kwds):
        PwMDOutputFile.__init__(self, *args, **kwds)
        self.set_attr_lst(self.attr_lst + ['econst'])

    def _get_datadct(self):
        verbose("getting _datadct")
        cmd = "grep 'Ekin.*T.*Etot' %s \
              | awk '{print $3\" \"$7\" \"$11}'"%self.filename
        ret_str = com.backtick(cmd)
        if ret_str.strip() == '':
            return None
        else:            
            data = np.atleast_2d(np.loadtxt(StringIO(ret_str)))
            return {'ekin': data[:,0],
                    'temperature': data[:,1],
                    'econst': data[:,2]}
    
    def get_ekin(self):
        verbose("getting ekin")
        self.try_set_attr('_datadct')
        return self._datadct['ekin']
    
    def get_econst(self):
        verbose("getting econst")
        self.try_set_attr('_datadct')
        return self._datadct['econst']
    
    def get_temperature(self):
        verbose("getting temperature")
        self.try_set_attr('_datadct')
        return self._datadct['temperature']


class CpmdSCFOutputFile(StructureFileParser):
    """Parse output from a CPMD "single point calculation" (wave function
    optimization). 
    
    Some extra files are assumed to be in the same directory as self.filename.
    
    extra files::

        GEOMETRY.scale
    
    Notes
    -----
    * The SYSTEM section must have SCALE such that a file GEOMETRY.scale is
      written.
    * To have forces in the output, use PRINT ON FORCES COORDINATES in the CPMD
      section::

        &CPMD
            OPTIMIZE WAVEFUNCTION
            CONVERGENCE ORBITALS
                1.0d-7
            PRINT ON FORCES COORDINATES
            STRESS TENSOR
                1
            ODIIS NO_RESET=10
                20
            MAXITER
                100
            FILEPATH
                /tmp
        &END
        &SYSTEM
            SCALE 
            ....
        &END    
    """
    default_units = \
        {'energy': Ha / eV, # Ha -> eV
         'length': Bohr / Angstrom, # Bohr -> Angstrom
         'forces': Ha / eV * Angstrom / Bohr, # Ha / Bohr -> eV / Angstrom
         'stress': 0.1, # kbar -> GPa
        } 
    def __init__(self, *args, **kwds):
        StructureFileParser.__init__(self, *args, **kwds)
        self.attr_lst = [\
            'cell',
            'stress',
            'etot',
            'coords_frac',
            'symbols',
            'forces',
            'natoms',
            'nkpoints',
            'nstep_scf',
            'scf_converged',
        ]
        self.init_attr_lst()

    def _get_coords_forces(self):
        """Low precision cartesian coords [Bohr] + forces [Ha / Bohr] I guess.
        Only printed in this form if we use 
        &CPMD
            PRINT ON COORDINATES FORCES
        &END
        Forces with more precision are printed in the files TRAJECTORY or
        FTRAJECTORY, but only for MD runs.
        """
        verbose("getting _coords_forces")
        self.try_set_attr('natoms')
        if self.is_set_attr('natoms'):
            cmd = "egrep -A%i 'ATOM[ ]+COORDINATES[ ]+GRADIENTS' %s \
                  | tail -n%i \
                  | awk '{print $3\" \"$4\" \"$5\" \"$6\" \"$7\" \"$8}'" \
                  %(self.natoms, self.filename, self.natoms)
            return arr2d_from_txt(com.backtick(cmd))                  
        else:
            return None
    
    def _get_scale_file(self):
        """Read GEOMETRY.scale file with fractional coords."""
        fn = os.path.join(self.basedir, 'GEOMETRY.scale')
        if os.path.exists(fn):
            cmd = "grep -A3 'CELL MATRIX (BOHR)' %s | tail -n3" %fn
            cell = arr2d_from_txt(com.backtick(cmd))
            self.assert_set_attr('natoms')
            cmd = "grep -A%i 'SCALED ATOMIC COORDINATES' %s | tail -n%i" \
                  %(self.natoms, fn, self.natoms)
            arr = arr2d_from_txt(com.backtick(cmd), dtype=str)
            coords_frac = arr[:,:3].astype(np.float)
            symbols = arr[:,3].tolist()
            return {'coords_frac': coords_frac, 
                    'symbols': symbols,
                    'cell': cell}
        else:
            return None
    
    def _get_cell_2d(self):
        """2d array `cell` [Bohr] for fixed-cell MD or SCF from GEOMETRY.scale
        file."""
        verbose("getting _cell_2d")
        if self.check_set_attr('_scale_file'):
            return self._scale_file['cell']
        else: 
            return None
    
    def get_cell(self):
        """2d cell [Bohr]"""
        verbose("getting cell")
        return self.get_return_attr('_cell_2d')
    
    def get_stress(self):
        """[kbar]"""
        verbose("getting stress")
        cmd = "grep -A3 'TOTAL STRESS TENSOR' %s | tail -n3" %self.filename
        return arr2d_from_txt(com.backtick(cmd))              

    def get_etot(self):
        """[Ha]"""
        verbose("getting etot")
        cmd =  r"grep 'TOTAL ENERGY =' %s | tail -n1 | awk '{print $5}'" %self.filename
        return float_from_txt(com.backtick(cmd))
    
    def get_coords_frac(self):
        verbose("getting coords_frac")
        if self.check_set_attr('_scale_file'):
            return self._scale_file['coords_frac']
        else:            
           return None
 
    def get_symbols(self):
        verbose("getting symbols")
        if self.check_set_attr('_scale_file'):
            return self._scale_file['symbols']
        else:            
           return None
 
    def get_forces(self):
        """[Ha / Bohr]"""
        verbose("getting forces")
        if self.check_set_attr('_coords_forces'):
            return self._coords_forces[:,3:]
        else:
            return None
    
    def get_natoms(self):
        """Number of atoms. Apparently only printed as "NUMBER OF ATOMS ..." in
        the SCF case, not in MD. So we use wc -l on the GEOMETRY file, which
        has `natoms` lines (normally) and 6 colunms. Sometimes (so far seen in
        variable cell calcs) there are some additional lines w/ 3 columns,
        which we skip."""
        verbose("getting natoms")
        fn = os.path.join(self.basedir, 'GEOMETRY')
        if os.path.exists(fn):
            cmd = "egrep '([0-9][ ]+.*){5,}' %s | wc -l" %fn
            return int_from_txt(com.backtick(cmd))
        else:
            return None
    
    def get_nkpoints(self):
        verbose("getting nkpoints")
        cmd = r"grep 'NUMBER OF SPECIAL K POINTS' %s | \
            sed -re 's/.*COORDINATES\):\s*([0-9]+)\s*.*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))

    def get_nstep_scf(self):
        verbose("getting nstep_scf")
        cmd = r"grep -B2 'RESTART INFORMATION WRITTEN' %s | head -n1 \
              | awk '{print $1}'" %self.filename
        return int_from_txt(com.backtick(cmd))
   
    def get_scf_converged(self):
        verbose("getting scf_converged")
        cmd = "grep 'BUT NO CONVERGENCE' %s" %self.filename
        if com.backtick(cmd).strip() == "":
            return True
        else:
            return False


class CpmdMDOutputFile(TrajectoryFileParser, CpmdSCFOutputFile):
    """Parse CPMD MD output. 
    
    Works with BO-MD and CP-MD, fixed and variable cell. Some attrs may be None
    or have different shapes (2d va 3d arrays) depending on what type of MD is
    parsed and what info/files are available.
    
    Notes for the comments below::

        {A,B,C} = A or B or C
        (A) = A is optional
        (A (B)) = A is optional, but only if present, B is optional

    Extra files which will be parsed and MUST be present::

        GEOMETRY.scale
        GEOMETRY
        TRAJECTORY
        ENERGIES
    
    Extra files which will be parsed and MAY be present depending on the type
    of MD::

        (FTRAJECTORY)
        (CELL) 
        (STRESS)
    
    Notes
    -----
    The input should look like that::

        &CPMD
            MOLECULAR DYNAMICS {BO,CP}
            (PARRINELLO-RAHMAN (NPT))
            PRINT ON FORCES COORDINATES
            TRAJECTORY XYZ FORCES
            STRESS TENSOR
                <step>
            ...
        &END

        &SYSTEM
            SCALE
            ...
        &END
    
    Tested with CPMD 3.15.1, the following extra files are always written::

        GEOMETRY.scale
        GEOMETRY
        TRAJECTORY
        ENERGIES

    In the listing below, we show which extra files are written (+) or not (-)
    if the input follows the example above.
    
    Also, the order of columns in the ENERGIES file depends on what type of MD
    we are running. In case of BO-MD it depends on the kind of wavefunction
    optimizer, too! This is most unpleasant. Currently we rely on the fact that
    each tested case has a different number of columns, but this is very
    hackish b/c it is not guaranteed to be unique! Maybe, we should let the
    user set self.energies_order or a keywords mdtype={'cp-npt', 'bo', etc}
    instead of subclassed for each case. 
    
    This is what we tested so far (cpmd 3.15.1). For BO-MD + ODIIS, some
    columns are always 0.0, but all are there (e.g. EKINC is there but 0.0 b/c
    not defined for BO, only CP). For BO-MD, we list the wf optimizer (xxx for
    CP b/c there is none)::

        MOLECULAR DYNAMICS BO
            +FTRAJECTORY
            -CELL
            -STRESS         # why!?
          ODISS        
            NFI EKINC TEMPP EKS ECLASSIC EHAM DIS TCPU 
          LANCZOS DIAGONALIZATION
            NFI TEMPP EKS ECLASSIC DIS TCPU

        MOLECULAR DYNAMICS CP
            +FTRAJECTORY
            -CELL
            +STRESS
          xxx        
            NFI EKINC TEMPP EKS ECLASSIC EHAM DIS TCPU 
        
        MOLECULAR DYNAMICS BO
        PARRINELLO-RAHMAN
            not implemented !
            
        MOLECULAR DYNAMICS CP
        PARRINELLO-RAHMAN
            -FTRAJECTORY    # why!?
            +CELL
            +STRESS
          xxx        
            NFI EKINC EKINH TEMPP EKS ECLASSIC EHAM DIS TCPU

        MOLECULAR DYNAMICS BO
        PARRINELLO-RAHMAN NPT
            -FTRAJECTORY    # why!?
            +CELL
            +STRESS
          ODIIS
            NFI EKINC EKINH TEMPP EKS ECLASSIC EHAM DIS TCPU
        
        MOLECULAR DYNAMICS CP
        PARRINELLO-RAHMAN NPT
            -FTRAJECTORY    # why!?
            +CELL
            +STRESS
          xxx
            NFI EKINC EKINH TEMPP EKS ECLASSIC EHAM DIS TCPU
    """        

    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        filename : file to parse
        """
        self.default_units.update(\
            {'time': constants.thart / constants.fs, # thart -> fs
             'velocity': Bohr / Ang * fs / thart,    # Bohr / thart -> Ang / fs
            })
        TrajectoryFileParser.__init__(self, *args, **kwds)
        self.attr_lst = [\
            'cell',
            'coords',
            'econst',
            'ekin',
            'ekin_cell',
            'ekin_elec',
            'etot',
            'forces',
            'natoms',
            'nstep',
            'stress',
            'symbols',
            'temperature',
            'temperature_cell',
            'timestep',
            'velocity',
        ]
        self.init_attr_lst()
        
        self._energies_order = {\
            9:\
                {'nfi': 0,
                 'ekinc': 1,
                 'ekinh': 2,
                 'tempp': 3,
                 'eks': 4,
                 'eclassic': 5,
                 'eham': 6,
                 'dis': 7,
                 'tcpu': 8},
            8:\
                {'nfi': 0,
                 'ekinc': 1,
                 'tempp': 2,
                 'eks': 3,
                 'eclassic': 4,
                 'eham': 5,
                 'dis': 6,
                 'tcpu': 7},
            6:\
                {'nfi': 0,
                 'tempp': 1,
                 'eks': 2,
                 'eclassic': 3,
                 'dis': 4,
                 'tcpu': 5},
            }                 
    
    def _get_energies_file(self):
        verbose("getting _energies_file")
        fn = os.path.join(self.basedir, 'ENERGIES')
        if os.path.exists(fn):
            arr = np.loadtxt(fn)
            ncols = arr.shape[-1]
            if ncols not in self._energies_order.keys():
                raise StandardError("only %s columns supported in "
                    "ENERGIES file" %str(self._energies_order.keys()))
            else:
                order = self._energies_order[ncols]
            dct = {}
            for key, idx in order.iteritems():
                dct[key] = arr[:,idx]
            del arr
            return dct
        else:
            return None
    
    def _get_coords_vel_forces(self):
        verbose("getting _coords_vel_forces")
        """Parse (F)TRAJECTORY file. Ignore lines which say
        "<<<<<<  NEW DATA  >>>>>>" from restarts.
        """
        # cols (both files):
        #   0:   natoms x nfi (natoms x 1, natoms x 2, ...)
        #   1-3: x,y,z cartesian coords [Bohr]
        #   4-6: x,y,z cartesian velocites [Bohr / thart ] 
        #        thart = Hartree time =  0.024189 fs
        # FTRAJECTORY extra:       
        #   7-9: x,y,z cartesian forces [Ha / Bohr]
        self.assert_set_attr('natoms')
        have_file = False
        have_forces = False
        fn_tr = os.path.join(self.basedir, 'TRAJECTORY')
        fn_ftr = os.path.join(self.basedir, 'FTRAJECTORY')
        if os.path.exists(fn_ftr):
            have_forces = True
            have_file = True
            ncols = 10
            fn = fn_ftr
        elif os.path.exists(fn_tr):            
            have_forces = False
            have_file = True
            ncols = 7
            fn = fn_tr
        if have_file:
            cmd = "grep -v '<<<<' %s | wc -l | awk '{print $1}'" %fn
            nlines = int_from_txt(com.backtick(cmd))
            nstep = float(nlines) / float(self.natoms)
            assert nstep % 1.0 == 0.0, (str(self.__class__) + \
                "nlines is not a multiple of nstep in %s" %fn)
            nstep = int(nstep)
            arr = arrayio.readtxt(fn, axis=self.timeaxis, shape=(nstep, self.natoms, ncols),
                             comments='<<<<')
            dct = {}
            dct['coords'] = arr[...,1:4]
            dct['velocity'] = arr[...,4:7]
            dct['forces'] = arr[...,7:] if have_forces else None
            return dct
        else:           
            return None
    
    def get_ekin(self):
        if self.check_set_attr('_energies_file'):
            return self._energies_file['eclassic']
        else:
            return None

    def get_cell(self):
        verbose("getting cell")
        """Parse CELL file [Bohr]. If CELL is not there, return 2d cell
        from GEOMETRY.scale (self.cell from CpmdSCFOutputFile)."""
        # So far tested CELL files have 6 cols: 
        # 1-3: x,y,z cell vectors
        # 4-6: cell forces? ditch them for now ...
        fn = os.path.join(self.basedir, 'CELL')
        if os.path.exists(fn):
            cmd = "head -n2 %s | tail -n1 | wc | awk '{print $2}'" %fn
            ncols = int_from_txt(com.backtick(cmd))
            cmd = "grep 'CELL PARAMETERS' %s | wc -l" %fn
            nstep = int_from_txt(com.backtick(cmd))
            cmd = "grep -A3 'CELL PARAMETERS' %s | grep -v 'CELL'" %fn
            arr = traj_from_txt(com.backtick(cmd), 
                                shape=(nstep,3,ncols),
                                axis=self.timeaxis)
            return arr[...,:3]                                
        else:
            if self.check_set_attr('_cell_2d'):
                return self._cell_2d
            else:
                return None

    def get_coords(self):
        verbose("getting coords")
        """Cartesian coords [Bohr]."""
        req = '_coords_vel_forces'
        self.try_set_attr(req)
        return self._coords_vel_forces['coords'] if self.is_set_attr(req) \
            else None
    
    def get_econst(self):
        """[Ha]"""
        verbose("getting econst")
        req = ['_energies_file', 'etot']
        self.try_set_attr_lst(req)
        if self.is_set_attr_lst(req):
            if self._energies_file.has_key('eham'):
                return self._energies_file['eham'] 
            else:
                return self.etot
        else:
            return None

    def get_ekinc(self):
        verbose("getting ekinc")
        """Fictitious electron kinetic energy [Ha]."""
        req = '_energies_file'
        self.try_set_attr(req)
        if self.is_set_attr(req) and self._energies_file.has_key('ekinc'):
            return self._energies_file['ekinc']
        else:
            return None

    def get_etot(self):
        verbose("getting etot")
        """Totat energy = EKS = Kohn Sham energy? [Ha]."""
        req = '_energies_file'
        self.try_set_attr(req)
        return self._energies_file['eks'] if self.is_set_attr(req) \
            else None

    def get_ekinh(self):
        verbose("getting ekinh")
        """Fictitious cell kinetic energy [Ha].
        From prcpmd.F: 
            EKINH [J] = 9/2 * kb [J/K] * TEMPH [K]
            EKINH [Ha] = 9/2 * kb [J/K] * TEMPH [K] / Ha
        where TEMPH is the fictitious cell temperature.            
        """
        req = '_energies_file'
        self.try_set_attr(req)
        if self.is_set_attr(req) and self._energies_file.has_key('ekinh'):
            return self._energies_file['ekinh']
        else:
            return None
    
    # alias
    def get_ekin_cell(self):
        verbose("getting ekin_cell")
        return self.get_ekinh()

    # alias
    def get_ekin_elec(self):
        verbose("getting ekin_elec")
        return self.get_ekinc()
    
    def get_temperature_cell(self):
        """[K]"""
        verbose("getting temperature_cell")
        req = 'ekin_cell'
        self.try_set_attr(req)
        if self.is_set_attr(req):
            return self.ekin_cell * 2.0 / 9.0 / constants.kb * constants.Ha
        else:
            return None
    
    def get_forces(self):
        """Cartesian forces [Ha/Bohr]."""
        verbose("getting forces")
        req = '_coords_vel_forces'
        self.try_set_attr(req)
        return self._coords_vel_forces['forces'] if self.is_set_attr(req) \
            else None
    
    def get_nstep(self):
        verbose("getting nstep")
        req = '_energies_file'
        self.try_set_attr(req)
        return int(self._energies_file['nfi'][-1]) if self.is_set_attr(req) \
            else None
    
    def get_stress(self):
        """Stress tensor from STRESS file if available [kbar]"""
        verbose("getting stress")
        fn = os.path.join(self.basedir, 'STRESS')
        if os.path.exists(fn):
            cmd = "grep 'TOTAL STRESS' %s | wc -l" %fn
            nstep = int_from_txt(com.backtick(cmd))
            cmd = "grep -A3 'TOTAL STRESS TENSOR' %s | grep -v TOTAL" %fn
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(nstep,3,3),
                                 axis=self.timeaxis)              
        else:
            return None
    
    def get_temperature(self):
        """[K]"""
        verbose("getting temperature")
        req = '_energies_file'
        self.try_set_attr(req)
        return self._energies_file['tempp'] if self.is_set_attr(req) \
            else None
    
    def get_velocity(self):
        verbose("getting velocity")
        """Cartesian velocity [Bohr / thart]. Not sure about the unit!"""
        if self.check_set_attr('_coords_vel_forces'):
            return self._coords_vel_forces['velocity']
        else:      
            return None
    
    def get_timestep(self):
        """Timestep [thart]."""
        cmd = r"grep 'TIME STEP FOR IONS' %s | \
            sed -re 's/.*IONS:\s+(.*)$/\1/'" %self.filename
        return float_from_txt(com.backtick(cmd))            


# backward compat
PwOutputFile = PwMDOutputFile
