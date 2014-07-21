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
    Cp2k*OutputFile
    Lammps*OutputFile

are derived from FlexibleGetters -> UnitsHandler ->  {Structure,Trajectory}FileParser

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
  pwtool.common.cpickle_load(foo.pk). See also the *FileParser.load() docstring.

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

from pwtools import common, constants, regex, crys, atomic_data, num, \
    arrayio, _dcd
from pwtools.verbose import verbose
from pwtools.base import FlexibleGetters
from pwtools.constants import Ry, Ha, eV, Bohr, Angstrom, thart, Ang, fs, ps
from pwtools.crys import UnitsHandler
com = common
pj = os.path.join


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

def traj_from_txt(txt, shape, axis=0, dtype=np.float, sep=' '):
    """Used for 3d trajectories where the exact shape of the array as written
    by the MD code must be known, e.g. (nstep,N,3) where N=3 (cell, stress) or
    N=natoms (coords, forces, ...). 

    We use np.fromstring for speed, so `txt` can only contain numbers and
    separators (white space), no comments (like "# this is the header"), which
    numpy.loadtxt() would handle.

    Parameters
    ----------
    txt : string
        Text containing numbers (e.g. from ``common.backtick("grep ...")``)
    shape : tuple
        The 3d shape of the array when written along `axis` (see also
        arrayio.writetxt()) to text as 2d chunks and read back in as 3d array.
        Used to reconstruct the array.
    axis : int
        Axis along which the array was written in 2d chunks to text. (see
        also `axis` in arrayio.writetxt()).
        Used to reconstruct the array. 
        Only axis=0 implemented.
    dtype, sep : passed to np.fromstring    
    """
    if txt.strip() == '':
        return None
    else:
        assert len(shape) == 3, ("only 3d arrays supported")
        assert axis == 0, ("only axis=0 implemented")
        # Works only for axis = 0, but this is the only case we have when
        # parsing MD code output. Else, some rollaxis would be needed. E.g. if
        # shape=(natoms,nstep,3) and therefore axis=1, then we would do
        # np.rollaxis(fromstring(...).reshape(shape), axis=1, start=0) ->
        # (nstep,natoms,3)
        return np.fromstring(txt, sep=sep, dtype=dtype).reshape(shape)

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

# wrappers for _dcd functions with file-not-found error handling
def read_dcd_header(filename):
    if os.path.exists(filename):
        return _dcd.read_dcd_header(filename)
    else:
        raise StandardError("file not found: {0}".format(filename))

def read_dcd_data(filename, nstep, natoms):
    if os.path.exists(filename):
        return _dcd.read_dcd_data(filename, nstep, natoms)
    else:
        raise StandardError("file not found: {0}".format(filename))


#-----------------------------------------------------------------------------
# Parsers
#-----------------------------------------------------------------------------

class StructureFileParser(UnitsHandler):
    """Base class for single-structure parsers.
    """
    Container = crys.Structure
    default_units = {}    
    def __init__(self, filename=None, units=None):
        self.parse_called = False    
        self.filename = filename
        # Some parsers do 
        #   self._foo_file = os.path.join(self.basedir,'foo')
        # in their __init__. That should not fail if we create an instance
        # without passing a filename, like parser=SomeParsingClass(). So
        # instead of setting basedir=None by default, we set it to '.' .
        self.basedir = os.path.dirname(self.filename) if (self.filename is not
            None) else '.'
        UnitsHandler.__init__(self)
        # Each derived parser class has `default_units`, with which self.units
        # is updated. Then, self.units is updated from user input if we are
        # called as ``SomeParser(...,units=...)``. Last, self.units is passed
        # to Container(..., units=self.units) and units are applied there.
        # Clear? :)
        self.update_units(self.default_units)
        self.update_units(units)
        self.cont = self.Container(set_all_auto=False, units=self.units)
        self.init_attr_lst(self.cont.attr_lst)            
    
    def parse(self):
        self.set_all()
        self.parse_called = True

    def get_cont(self):
        """Populate and return a Container object."""
        if not self.parse_called:
            self.parse()
        for attr_name in self.cont.attr_lst:
            setattr(self.cont, attr_name, getattr(self, attr_name))
        # we assume that cont.set_all() calls apply_units()
        self.cont.set_all()
        return self.cont
   
    def apply_units(self):
        raise NotImplementedError("don't use that in parsers")
    
    def get_struct(self):
        return self.get_cont()


class TrajectoryFileParser(StructureFileParser):
    """Base class for MD-like parsers."""
    Container = crys.Trajectory
    # timeaxis in Trajectory defined before __init__, so we don't need to
    # instantiate the object
    timeaxis = Container.timeaxis
    
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
        if self.filename is not None:
            self.txt = common.file_read(self.filename)
    
    def _get_coords_data(self):
        pat = r'(ATOM|HETATM)[\s0-9]+([A-Za-z]+)[\sa-zA-Z0-9]*' + \
            r'[\s0-9]+((\s+'+ regex.float_re + r'){3}?)'
        # array of string type            
        return np.array([[m.group(2)] + m.group(3).split() for m in \
                         re.finditer(pat,self.txt)])
    
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
        pat = r'CRYST1\s+((\s+' + regex.float_re + r'){6}).*'
        match = re.search(pat, self.txt)
        return np.array(match.group(1).split()).astype(float)


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
    default: ``PwSCFOutputFile(filename, use_alat=True)`` b/c this is what most
    people will expect if they just call the parser on some file. Then, we
    multiply all relevent quantities with dimension length with the alat value
    from pw.out automatically.

    If ``use_alat=False``, we use ``alat=1.0``, i.e. all length quantities
    which are "in alat units" are returned exactly as found in the file, which
    is the same behavior as in all other parsers. Unit conversion happens only
    when we pass things to Structure / Trajectory using self.units. 

    If you need/want to use another alat (i.e. a value with more precision), 
    then you need to explicitly provide that value and use ``use_alat=False``::

    >>> alat = 1.23456789 # high precision value in Bohr
    >>> pp = PwSCFOutputFile('pw.out', use_alat=False, units={'length': alat*Bohr/Ang})
    >>> st = pp.get_struct()

    ``use_alat=False`` will prevent parsing the low precision value from
    'pw.out'. The option ``units=...`` will overwrite ``default_units['length']
    = Bohr/Ang``, which is used to convert all PWscf length [Bohr] to [Ang]
    when passing things to Trajectory. 

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
    pwtools standard units (eV, Angstrom, ...), use :meth:`get_struct`.

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
        cmd = "grep -c %s %s" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        if nstep > 0:
            cmd = "grep -A3 '%s' %s | grep -v -e %s -e '--'| \
                  awk '{print $4\"  \"$5\"  \"$6}'" \
                  %(key, self.filename, key)
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(nstep,3,3),
                                 axis=self.timeaxis)              
        else:
            return None

    def _get_etot_raw(self):
        verbose("getting _etot_raw")
        cmd =  r"grep '^!' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def _get_forces_raw(self):
        verbose("getting _forces_raw")
        if self.check_set_attr('natoms'):
            # nstep: get it from outfile b/c the value in any input file will be
            # wrong if the output file is a concatenation of multiple smaller files
            key = r'Forces\s+acting\s+on\s+atoms.*$'
            cmd = r"egrep -c '%s' %s" %(key.replace(r'\s', r'[ ]'), self.filename)
            nstep = nstep_from_txt(com.backtick(cmd))
            if nstep > 0:
                # Need to split traj_from_txt() up into loadtxt() + arr2d_to_3d() b/c
                # we need to get `nlines` first without an additional "grep -c
                # ..."
                cmd = "grep 'atom.*type.*force' %s \
                    | awk '{print $7\" \"$8\" \"$9}'" %self.filename
                arr2d = arr2d_from_txt(com.backtick(cmd))
                nlines = arr2d.shape[0]
                # nlines_block = number of force lines per step = N*natoms
                nlines_block = nlines / nstep
                assert nlines_block % self.natoms  == 0, ("nlines_block forces doesn't "
                    "match natoms")
                return arrayio.arr2d_to_3d(arr2d,
                                           shape=(nstep,nlines_block,3), 
                                           axis=self.timeaxis)     
            else:
                return None
        else:
            return None

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
        cmd = r"egrep -m1 -A%i 'site.*atom.*positions.*units.*\)' %s | tail -n%i | \
              sed -re 's/.*\((.*)\)/\1/g'" \
              %(natoms, self.filename, natoms)
        coords = arr2d_from_txt(com.backtick(cmd))
        cmd = r"egrep -m1 -A%i 'site.*atom.*positions.*units.*\)' %s | tail -n%i | \
              awk '{print $2}'" \
              %(natoms, self.filename, natoms)
        symbols = com.backtick(cmd).strip().split()
        return {'coords': coords, 'symbols': symbols}
    
    def _get_cell_2d(self):
        """Start 2d cell in alat units.
        
        Grep start cell from pw.out. Multiplication by alat in
        :meth:`get_cell`.
        
        The cell in pw.out is always in alat units (divided by alat) but
        printed with much less precision compared to the input file. If you
        need this information for further calculations, use the input file
        value."""
        cmd = "egrep -m1 -A3 'crystal.*axes.*units.*(a_0|alat)' %s | tail -n3 | \
               awk '{print $4\" \"$5\" \"$6}'" %(self.filename)
        return arr2d_from_txt(com.backtick(cmd))
    
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
            return None if forces is None else forces[:self.natoms,:]
        else:
            return None

    def get_nstep_scf(self):
        return self.raw_slice_get('nstep_scf', sl=0, axis=0)
    
    def get_cell(self):
        """Start cell [Bohr].
        
        Apply self.alat unit to _cell_2d."""
        if self.check_set_attr_lst(['_cell_2d', 'alat']):
            return self._cell_2d * self.alat
        else:
            return None

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
    
    Notes
    -----
    Units: Notes on units for PwSCFOutputFile, esp. alat, apply here as well.
    Additionally, the ATOMIC_POSITIONS and CELL_PARAMETERS blocks can have an
    optional unit, which we account for. See get_cell(), get_coords() and
    methods called in there. 

        | ATOMIC_POSITIONS <empty> | bohr | angstrom | alat | crystal
        | CELL_PARAMETERS <empty> | (alat=...) | bohr | angstrom | alat
    
    In each case, the quantity is multiplied by alat if applicable and
    converted to Bohr, which is PWscf's default length, and later to Ang by
    default (or whatever self.units['length'] does).
    
    Initial SCF run: A special "feature" of pwscf is that SCF coords+cell
    output is printed differently from MD-like output (where we have
    ATOMIC_POSITIONS and CELL_PARAMETERS blocks). Since this parser uses only
    the latter, the first etot+coords+cell+stress+... is skipped, i.e. the
    complete iteration=0 = initial SCF run. Therefore, if you use
    ``tr=io.read_pw_md('pw.out')``, ``tr[0]`` is actually NOT your start input
    structure! It is the first structure of the MD/relax. This may be a problem
    if you need to accurately calculate differences between initial and final
    relax structs, for instance. Then use::

    >>> st = io.read_pw_scf('pw.out') # parse initial SCF output only: step=0
    >>> tr_md = io.read_pw_md('pw.out') # parse MD-like output: step=1...end
    >>> tr = crys.concatenate((st, tr_md))
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
            
            ATOMIC_POSITIONS            -> None
            ATOMIC_POSITIONS unit       -> unit
            ATOMIC_POSITIONS (unit)     -> unit
            ATOMIC_POSITIONS {unit}     -> unit
            CELL_PARAMETERS (alat=1.23) -> alat

        Parameters
        ----------
        key : str (e.g. 'ATOMIC_POSITIONS')

        Returns
        -------
        str : unit
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
            cmd = 'grep -c %s %s' %(key, self.filename)
            nstep = nstep_from_txt(com.backtick(cmd))
            # coords
            cmd = "grep -A%i '%s' %s | grep -v -e %s -e '--' | \
                  awk '{print $2\"  \"$3\"  \"$4}'" \
                  %(natoms, key, self.filename, key)
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(nstep,natoms,3),
                                 axis=self.timeaxis)
        else:
            return None

    def _get_cell_3d(self):
        """Parse CELL_PARAMETERS block. 
        
        The block unit is ignored here. Only the content of the block is parsed
        (i.e. the CELL_PARAMETERS as they are in the file). See also
        ``_get_block_header_unit()`` and ``get_cell``.
        """
        verbose("getting _cell_3d")
        # nstep
        key = 'CELL_PARAMETERS'
        cmd = 'grep -c %s %s' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        # cell            
        cmd = "grep -A3 %s %s | grep -v -e %s -e '--'" %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(nstep,3,3),
                             axis=self.timeaxis)
    
    def _get_cell_3d_factors(self):
        """Parse CELL_PARAMETERS unit factor printed at each time step.
        
        ::
            CELL_PARAMETERS (alat= 22.75306514)

        Returns
        -------
        alat_values : 1d array (nstep,) or None
            1d array with alat for each time step if 'CELL_PARAMETERS.*alat' is
            found; None if not found or if use_alat=False.
        
        Notes
        -----
        During one run, that alat value doesn't change and is the same as alat
        returned by ``get_alat()``. But it is needed if you parse an output
        file which is a concatenation of multiple smaller files from vc-md runs
        where that alat value has changed. This happens if a vc-md is
        restarted. Then the new alat in the restart run is that of the last
        cell of the old run.
        """
        cmd = 'grep -m1 CELL_PARAMETERS.*alat.*= %s' %self.filename
        if com.backtick(cmd).strip() == '':
            return None
        else:
            if self.use_alat:
                cmd = r"grep CELL_PARAMETERS %s | sed -re 's/.*alat.*=\s*(" \
                    %self.filename + regex.float_re + r")\)*.*/\1/g'"
                return arr1d_from_txt(com.backtick(cmd))
            else:
                return None

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
        self._cell_2d. Beware: complicated units logic ahead!"""
        # From the manual, regarding the unit of CELL_PARAMETERS in the input
        # file:
        # 
        # bohr / angstrom: lattice vectors in bohr radii / angstrom.
        # alat or nothing specified: if a lattice constant (celldm(1)
        # or a) is present, lattice vectors are in units of the lattice
        # constant; otherwise, in bohr radii or angstrom, as specified.
        #
        # .. yo!
        
        # MD-like case
        if self.check_set_attr('_cell_3d'):
            # CELL_PARAMETERS (alat=...) | bohr | angstrom | alat
            if self.check_set_attr('cell_unit'):
                if self.cell_unit == 'bohr':
                    return self._cell_3d
                elif self.cell_unit == 'alat':
                    if self.check_set_attr('_cell_3d_factors'):
                        return self._cell_3d * self._cell_3d_factors[:,None,None]
                    elif self.check_set_attr('alat'):                    
                        return self._cell_3d * self.alat
                    else:
                        return None
                elif self.cell_unit == 'angstrom':
                    return self._cell_3d * Angstrom / Bohr
                else:
                    return None
            # CELL_PARAMETERS <empty> 
            else:
                if self.check_set_attr('alat'):                    
                    return self._cell_3d * self.alat
                else:
                    return None
        # return start cell 2d, will be broadcast to 3d in Trajectory
        elif self.check_set_attr_lst(['_cell_2d', 'alat']):
            return self._cell_2d * self.alat
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
    """Parse only calculation='vc-md'."""
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
        the SCF case, not in MD. So we use "grep -c" on the GEOMETRY file, which
        has `natoms` lines (normally) and 6 colunms. Sometimes (so far seen in
        variable cell calcs) there are some additional lines w/ 3 columns,
        which we skip."""
        verbose("getting natoms")
        fn = os.path.join(self.basedir, 'GEOMETRY')
        if os.path.exists(fn):
            cmd = "egrep -c '([0-9][ ]+.*){5,}' %s" %fn
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
            cmd = "grep -c -v '<<<<' %s" %fn
            nlines = int_from_txt(com.backtick(cmd))
            nstep = float(nlines) / float(self.natoms)
            assert nstep % 1.0 == 0.0, (str(self.__class__) + \
                "nlines is not a multiple of nstep in %s" %fn)
            nstep = int(nstep)
            # Need to use the slower arrayio.readtxt() here instead of
            # traj_from_txt() which uses fast fromstring() b/c we have
            # comments='<<<<'. The other way would be to  use
            # common.backtick("grep -v '<<<<' ...")) the text such that we have
            # only numbers in it and then pass that to traj_from_txt().
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
            cmd = "grep -c 'CELL PARAMETERS' %s" %fn
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
    
    def get_stress(self):
        """Stress tensor from STRESS file if available [kbar]"""
        verbose("getting stress")
        fn = os.path.join(self.basedir, 'STRESS')
        if os.path.exists(fn):
            cmd = "grep -c 'TOTAL STRESS' %s" %fn
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


class Cp2kSCFOutputFile(StructureFileParser):
    """CP2K SCF output parser ("global/run_type energy_force,print_level low"). 

    Notes
    -----
    * Since we mainly use "global/print_level low", we don't bother to
      special-case for "global/print_level medium". Therefore, we don't extract
      cell and coords. SCF runs are only done for convergence tests, so forces,
      etot and stress are important.
    * It seems that with default &print settings, SCF runs write the stress
      tensor in GPa, while for MD, the default is bar. Thank you very much!
    """
    default_units = \
        {'energy': Ha / eV, # Ha -> eV
         'forces': Ha / eV * Angstrom / Bohr, # Ha / Bohr -> eV / Angstrom
        } 
    
    def __init__(self, *args, **kwds):
        StructureFileParser.__init__(self, *args, **kwds)
        self.attr_lst = [\
            'natoms',
            'etot',
            'forces',
            'stress',
            'symbols',
        ]
    
    def _get_run_type(self):
        cmd = r"grep -m1 'GLOBAL.*Run type' {0} | sed \
            -re 's/.*type\s+(.*)\s*/\1/'".format(self.filename)
        return com.backtick(cmd).strip()            

    def _get_natoms_symbols_forces(self):
        cmd = r"sed -nre '1,/ATOMIC FORCES/d; " + \
               "1,/Atom\s+Kind\s+Element/d; " + \
               "/SUM OF ATOMIC FORCES/q;p' %s" %self.filename
        ret = com.backtick(cmd).strip()
        if ret != '':
            arr = np.array([x.split() for x in ret.splitlines()])
            return {'natoms': arr.shape[0],
                    'symbols': arr[:,2].tolist(),
                    'forces': arr[:,3:].astype(float)}
        else:
            return None
    
    def get_natoms(self):
        if self.check_set_attr('_natoms_symbols_forces'):
            return self._natoms_symbols_forces['natoms']
        else:
            return None

    def get_symbols(self):
        if self.check_set_attr('_natoms_symbols_forces'):
            return self._natoms_symbols_forces['symbols']
        else:
            return None

    def get_forces(self):
        """[Ha/Bohr]"""
        if self.check_set_attr('_natoms_symbols_forces'):
            return self._natoms_symbols_forces['forces']
        else:
            return None

    def get_etot(self):
        """[Ha]"""
        cmd = r"sed -nre 's/.*ENERGY.*Total.*energy.*:(.*)/\1/p' %s" %self.filename
        return float_from_txt(com.backtick(cmd))

    
    def get_stress(self):
        """[GPa]"""
        cmd = r"grep -A5 'STRESS TENSOR.*GPa' %s | egrep -v 'X[ ]+Y[ ]+Z' | \
            egrep '^[ ]+(X|Y|Z)'" %self.filename
        ret = com.backtick(cmd).strip()
        arr = np.array([x.split() for x in ret.splitlines()])
        return arr[:,1:].astype(float)


class Cp2kMDOutputFile(TrajectoryFileParser, Cp2kSCFOutputFile):
    """CP2K MD output parser. Tested with cp2k v2.4, "global/run_type
    md,print_level low".
    """
    def __init__(self, *args, **kwds):
        self.default_units['stress'] = 1e-4 # bar -> GPa
        self.default_units['velocity'] = Bohr/thart / Ang*fs # Bohr/thart -> Ang/fs
        TrajectoryFileParser.__init__(self, *args, **kwds)
        self.attr_lst = [\
            'cell',
            'coords',
            'econst',
            'ekin',
            'etot',
            'forces',
            'natoms',
            'stress',
            'symbols',
            'temperature',
            'timestep',
            'velocity',
        ]
        self.init_attr_lst()
        self._cell_file = common.pj(self.basedir, 'PROJECT-1.cell')
        self._ener_file = common.pj(self.basedir, 'PROJECT-1.ener')
        self._stress_file = common.pj(self.basedir, 'PROJECT-1.stress')
        self._pos_file = common.pj(self.basedir, 'PROJECT-pos-1.xyz')
        self._frc_file = common.pj(self.basedir, 'PROJECT-frc-1.xyz')
        self._vel_file = common.pj(self.basedir, 'PROJECT-vel-1.xyz')
    
    @staticmethod
    def _cp2k_repack_arr(arr):
        """Convert arr, which is an unrolled (nstep,3,3) array, back."""
        out = np.empty((arr.shape[0],3,3), dtype=float)
        out[:,0,0] = arr[:,2]
        out[:,0,1] = arr[:,3]
        out[:,0,2] = arr[:,4]
        out[:,1,0] = arr[:,5]
        out[:,1,1] = arr[:,6]
        out[:,1,2] = arr[:,7]
        out[:,2,0] = arr[:,8]
        out[:,2,1] = arr[:,9]
        out[:,2,2] = arr[:,10]
        return out
    
    def _cp2k_xyz2arr(self, fn):
        """Parse cp2k style XYZ files and return the 3d array."""
        cmd = "grep -c 'i = .*E =' %s" %fn
        nstep = nstep_from_txt(com.backtick(cmd))
        natoms = int_from_txt(com.backtick("head -n1 %s" %fn))
        cmd = "awk '!/i =.*E =|^[ ]+[0-9]+/ \
            {print $2\" \"$3\" \"$4}' %s" %fn
        assert self.timeaxis == 0
        return np.fromstring(common.backtick(cmd), sep=' ').reshape(nstep,natoms,3)

    def _get_cell_file_arr(self):
        if os.path.exists(self._cell_file):
            return np.loadtxt(self._cell_file)
        else:            
            return None
    
    def _get_ener_file_arr(self):
        if os.path.exists(self._ener_file):
            return np.loadtxt(self._ener_file)
        else:            
            return None

    def _get_stress_file_arr(self):
        if os.path.exists(self._stress_file):
            return np.loadtxt(self._stress_file)
        else:            
            return None

    def get_natoms(self):
        cmd = r"grep -m1 'Number of atoms:' %s | \
            sed -re 's/.*:(.*)/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd)) 
    
    def get_timestep(self):
        """[fs]"""
        cmd = r"egrep -m1 'MD\| Time Step \[fs\]' %s | \
            sed -re 's/.*\](.*)/\1/'" %self.filename
        return float_from_txt(com.backtick(cmd))            

    def _get_coords_symbols(self):
        """Cartesian [Ang]"""
        if os.path.exists(self._pos_file):
            coords = self._cp2k_xyz2arr(self._pos_file)
            if self.check_set_attr('natoms'):
                cmd = r"grep -m1 -A%i 'i =' %s | \
                    grep -v 'i ='| awk '{print $1}'" %(self.natoms,self._pos_file) 
                symbols = com.backtick(cmd).strip().split()
                return {'coords': coords, 'symbols': symbols}
            else:
                return None
        else:            
            return None
    
    def _get_forces_from_outfile(self):
        if self.check_set_attr('natoms'):
            cmd = r"grep -c 'ATOMIC FORCES in' %s" %self.filename
            nstep = nstep_from_txt(com.backtick(cmd))
            cmd = r"sed -re '/^\s*$/d' {fn} | grep -A{nlines} 'ATOMIC FORCES in' \
                 | egrep -v -e 'ATOM|--|Kind' \
                 | tr -s ' ' | cut -d ' ' -f5-".format(fn=self.filename,
                                                       nlines=self.natoms+1)
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(nstep,self.natoms,3),
                                 axis=self.timeaxis)              
        else:
            return None

    def get_coords(self):
        """Cartesian [Ang]"""
        if self.check_set_attr('_coords_symbols'):
            return self._coords_symbols['coords']
        else:
            return None

    def get_symbols(self):
        if self.check_set_attr('_coords_symbols'):
            return self._coords_symbols['symbols']
        else:
            return None
    
    def get_forces(self):
        """[Ha/Bohr]"""
        if os.path.exists(self._frc_file):
            return self._cp2k_xyz2arr(self._frc_file)
        else:            
            return self._get_forces_from_outfile()
    
    def get_velocity(self):
        """[Bohr/thart]"""
        if os.path.exists(self._vel_file):
            return self._cp2k_xyz2arr(self._vel_file)
        else:            
            return None
    
    def get_stress(self):
        """[bar]"""
        if self.check_set_attr('_stress_file_arr'):
            return self._cp2k_repack_arr(self._stress_file_arr)
        else:
            return None

    def get_ekin(self):
        """[Ha]"""
        if self.check_set_attr('_ener_file_arr'):
            return self._ener_file_arr[:,2]
        else:
            return None

    def get_temperature(self):
        """[K]"""
        if self.check_set_attr('_ener_file_arr'):
            return self._ener_file_arr[:,3]
        else:
            return None
    
    def get_etot(self):
        """[Ha]"""
        if self.check_set_attr('_ener_file_arr'):
            return self._ener_file_arr[:,4]
        else:
            return None
    
    def get_econst(self):
        """[Ha]"""
        if self.check_set_attr('_ener_file_arr'):
            return self._ener_file_arr[:,5]
        else:
            return None
    
    def get_cell(self):
        """[Ang]"""
        if self.check_set_attr('_cell_file_arr'):
            return self._cp2k_repack_arr(self._cell_file_arr)
        else:
            return None

    def get_volume(self):
        """[Ang^3]"""
        if self.check_set_attr('_cell_file_arr'):
            return self._cell_file_arr[:,-1]
        else:
            return None


class Cp2kRelaxOutputFile(Cp2kMDOutputFile):
    """Parse cp2k global/run_type cell_opt. geo_opt might also work, but not
    tested yet."""
    def get_natoms(self):
        if os.path.exists(self._pos_file):
            cmd = r"head -n1 {0}".format(self._pos_file)
            return int_from_txt(com.backtick(cmd))
        else:
            return None
    
    def get_etot(self):
        if os.path.exists(self._pos_file):
            cmd = r"awk '/i =.*E/ {print $6}' %s" %self._pos_file
            return arr1d_from_txt(com.backtick(cmd))
        else:
            return None
        
    def get_cell(self):
        # For cell_opt, cp2k does a final scf calc after the cell optimization.
        # That creates an additional time step in _pos_file, but NOT in
        # _cell_file. Then, coords[-1,...] and coords[-2,...] are the same b/c
        # the final geometry is simply scf'ed again. Also etot is one step
        # longer. Handle that by appending cell[-1,...] to the end of the cell
        # array in order to have equal length. Need that to make get_traj()
        # happy.
        if self.check_set_attr('_cell_file_arr'): 
            cell = self._cp2k_repack_arr(self._cell_file_arr)
            self.assert_set_attr('_run_type')
            if self._run_type == 'CELL_OPT':
                if self.check_set_attr('coords'):
                    offset = self.coords.shape[0] - cell.shape[0] 
                    if offset == 1:
                        return np.concatenate((cell, cell[-1,...][None,...]), 
                                               axis=0)
                    else:
                        raise StandardError("cell and coords have a timestep "
                            "offset != 1, dunno what to do "
                            "(offset={0}, coords: {1}, cell: {2})".format(offset,self.coords.shape, cell.shape))
                else:
                    return None
            # GEO_OPT not tested yet. Simply return cell for now unchanged, if
            # defined.
            else:
                return cell
        else:
            return None


class LammpsTextMDOutputFile(TrajectoryFileParser):
    """Parse LAMMPS text output. 
    
    We parse the default ``log.lammps`` file (`filename`) with ``thermo``
    output and, if present, a custom dump file ``lmp.out.dump`` created by
    something like ``dump 2 all custom 1 lmp.out.dump ...`` Tested with MD and
    structure optimization (``minimize``).
   
    Currently hardcoded file names:
        | `dumpfilename` = ``basedir/lmp.out.dump`` 
        | `symbolsfilename` = ``basedir/lmp.struct.symbols``
    where `basedir` is the dir where `filename` (i.e. ``log.lammps``) lives.
    
    default_units are for lammps metal units.
    
    Examples
    --------
    Example lammps input::
    
        clear
        units metal 
        boundary p p p 
        atom_style atomic

        read_data lmp.struct

        ### interactions 
        pair_style tersoff 
        pair_coeff * * AlN.tersoff Al N

        ### IO
        dump dump_txt all custom 1 lmp.out.dump id type xsu ysu zsu xu yu zu fx fy fz vx vy vz
        dump dump_dcd all dcd 1 lmp.out.dcd
        ##dump dump_xyz all xyz 1 lmp.out.xyz
        ##dump_modify dump_xyz element Al N 
        dump_modify dump_txt sort id 
        dump_modify dump_dcd sort id unwrap yes
        thermo_style custom step temp vol cella cellb cellc cellalpha cellbeta cellgamma &
                            ke pe etotal &
                            press pxx pyy pzz pxy pxz pyz cpu
        thermo_modify flush yes
        thermo 1

        ### init
        velocity all create 300.0 123 rot yes dist gaussian

        ### run
        fix fix_npt all npt temp 1000 1000 0.01 tri 0 0 0.3 tchain 4 pchain 4 &
            mtk yes scaleyz no scalexz no scalexy no flip no
        timestep 2.5e-3
        run 1000

    Notes
    -----
    * columns in `filename` and `dumpfilename`: We automagically extract
      the "header" (e.g. "Step Temp Volume Cella ..." in `filename` or
      "ITEM: ATOMS id type xsu ysu zsu fx fy fz vx vy vz" in
      `dumpfilename`) and map data to these symbols. See `_thermo_dct` and
      `_dump_dct`. Currently, "xsu ysu zsu" is parsed to get
      coords_frac. "xu yu zu" is parsed to get coords. Wrapped coordinates
      (e.g. "xs ys zs" and "x y z") are ignored.
    * multiple runs from one input script (i.e. 2 or more ``run``
      commands): it seems that the last step of the preceeding run is
      printed again by the new run by ``thermo_style``, which results in
      more `nstep` in `filename` as there really are in `dumpfilename`. We
      parse all important stuff (coords, cell, velocity) from
      `dumpfilename`, but `temperature` and `stress` etc. is from
      `filename`. Watch out when plotting.
    * cell: We parse cell from "ITEM: BOX BOUNDS" in `dumpfilename` and the
      cell is always [[x,0,0],[xy,y,0],[xz,yz,z]], i.e. it is aligned in the
      same way in each timestep.
    * atom symbols: First we try to read `symbolsfilename`, which may have been
      written by :func:`pwtools.io.write_lammps`. If that is not found, we try
      to use the ``type`` column in `dumpfilename` together with a type number
      -> atom symbol mapping either from the `order` input keyword or a
      ``dump_modify ... element`` line in `filename` if found.
      """
    def __init__(self, filename='log.lammps', order=None, **kwds):
        """
        Parameters
        ----------
        filename : str
            Text output file, where ``thermo_style [custom]``
            output is written to. The lammps default is ``log.lammps``.
        order : dict, optional
            See :meth:`pwtools.crys.Structure.get_order`. Mapping of atom
            symbols to ``type`` in lammps, i.e. {'Al': 1, 'N': 2}. If None then
            we try to use ``dump_modify ... element`` if present in `filename`,
            where lammps echos the input script. `symbols` is build from the
            ``type`` column in `dumpfilename`.
        """
        self.default_units['stress'] = 1e-4     # bar -> GPa
        self.default_units['velocity'] = fs/ps  # Ang/ps -> Ang/fs
        self.default_units['time'] = ps/fs      # ps -> fs
        TrajectoryFileParser.__init__(self, filename, **kwds)
        self.attr_lst = [\
            'cell',
            'coords_frac',
            'coords',
            'cryst_const',
            'ekin',
            'etot',
            'forces',
            'natoms',
            'stress',
            'symbols',
            'temperature',
            'timestep',
            'velocity',
            'volume',
        ]
        self.init_attr_lst()
        self.order = order
        # Text output file from ``dump <ID> all custom 1 ...``.
        self.dumpfilename = pj(self.basedir, 'lmp.out.dump')
        # written by io.write_lammps()
        self.symbolsfilename = pj(self.basedir, 'lmp.struct.symbols')
   
    @staticmethod
    def _get_from_dct(dct, key):
        if dct.has_key(key):
            return dct[key]
        else:
            return None
    
    @staticmethod 
    def _assert_shape_mod(name, a, b):
        mod = a % b
        msg = ("{name}: shape mismatch: a: {a}, b: {b}, "
               "{a} % {b} = {mod}, "
               "should be 0".format(a=a,b=b,mod=mod,name=name))
        assert mod == 0, msg

    def _get_thermo_dct(self):
        """Parse all text between "Step ..." and "Loop ..." in self.filename::

            Step Temp Volume Cella Cellb Cellc 
                   0            0    37.412297            3            3          4.8 
                   1            0     37.42182    3.0002318    3.0002318      4.80048 
                   2            0    37.431339    3.0004633    3.0004633      4.80096 
            ....
                        
                1000            0    37.431339    3.0004633    3.0004633      4.80096 
            Loop time of 0.115472 on 1 procs (1 MPI x 1 OpenMP) for 1043 steps with 4 atoms

        This is usually generated by:: 
            
            thermo_style custom step temp vol cella cellb cellc
            thermo 1
        
        in the input file. I think if no ``thermo_style`` command is used, it
        still prints a line starting with "Step ...".
        """
        if os.path.exists(self.filename):
            header = com.backtick("grep -m1 Step %s" %self.filename).split()
            # Strip all text except for the data columns. Also works for
            # multiple ``run`` or ``minimize`` commands in one input file,
            # which cause wildly mixed text.
            cmd = r"sed -nre '/^Step/,/^Loop/p' %s | \
                    egrep -v 'Step|Loop'" %self.filename
            arr = arr2d_from_txt(com.backtick(cmd))
            return dict((x, arr[:,ii]) for ii,x in enumerate(header))
        else:
            return None

    def _get_dump_dct(self):
        if self.check_set_attr('natoms') and \
           os.path.exists(self.dumpfilename):
            cmd = r"grep -c 'ITEM: TIMESTEP' %s" %self.dumpfilename
            nstep = nstep_from_txt(com.backtick(cmd))
            header = com.backtick("grep -m1 'ITEM: ATOMS' %s \
                | sed -re 's/.*TOMS //'" %self.dumpfilename).split()
            cmd = r"grep -A%i 'ITEM: ATOMS' %s | \
                    egrep -ve '--|ITEM'" %(self.natoms, self.dumpfilename)
            arr = np.fromstring(com.backtick(cmd), 
                                sep=' ').reshape(nstep*self.natoms,len(header))
            self._assert_shape_mod('dump', arr.shape[0], self.natoms)
            return dict((x, arr[:,ii]) for ii,x in enumerate(header))
        else:
            return None
    
    def _lmp_dump2arr3d(self, dct, keys):
        if self.check_set_attr('natoms'):
            for k in keys:
                if not dct.has_key(k):
                    return None
            nstep = dct[keys[0]].shape[0] / self.natoms
            arr = np.array([dct[keys[0]],
                            dct[keys[1]],
                            dct[keys[2]]]).T.reshape((nstep,
                                                      self.natoms,3))
            return arr
        else:
            return None
    
    def get_natoms(self):
        if os.path.exists(self.dumpfilename):
            cmd = r"grep -A1 -m1 'ITEM: NUMBER OF ATOMS' %s | tail -n1" %self.dumpfilename
            return nstep_from_txt(com.backtick(cmd))
        else:
            return None
    
    def get_stress(self):
        if self.check_set_attr('_thermo_dct'):
            keys = 'Pxx Pyy Pzz Pxy Pxz Pyz'.split()
            for k in keys:
                if not self._thermo_dct.has_key(k):
                    return None
            nstep = self._thermo_dct['Pxx'].shape[0]
            arr = np.zeros((nstep,3,3))
            arr[:,0,0] = self._thermo_dct['Pxx']
            arr[:,1,1] = self._thermo_dct['Pyy']
            arr[:,2,2] = self._thermo_dct['Pzz']
            arr[:,1,0] = self._thermo_dct['Pxy']
            arr[:,2,0] = self._thermo_dct['Pxz']
            arr[:,2,1] = self._thermo_dct['Pyz']
            return arr
        else:
            return None
    
    def get_etot(self):
        """Potetntial energy PotEng [eV]. etot+ekin here is
        TotEng=PotEng+KinEng in lammps. In DFT, the potential energy is usually
        called "total energy".
        """
        if self.check_set_attr('_thermo_dct'):
            return self._get_from_dct(self._thermo_dct, 'PotEng')
        else:
            return None
    
    def get_ekin(self):
        if self.check_set_attr('_thermo_dct'):
            return self._get_from_dct(self._thermo_dct, 'KinEng')
        else:
            return None
    
    def get_temperature(self):
        if self.check_set_attr('_thermo_dct'):
            return self._get_from_dct(self._thermo_dct, 'Temp')
        else:
            return None
    
    def get_volume(self):
        if self.check_set_attr('_thermo_dct'):
            return self._get_from_dct(self._thermo_dct, 'Volume')
        else:
            return None
    
    def get_cryst_const(self):
        if self.check_set_attr('_thermo_dct'):
            keys = 'Cella Cellb Cellc CellAlpha CellBeta CellGamma'.split()
            for k in keys:
                if not self._thermo_dct.has_key(k):
                    return None
            nstep = len(self._thermo_dct['Cella'])
            ret = np.empty((nstep,6))
            for ii,k in enumerate(keys):
                ret[:,ii] = self._thermo_dct[k]
            return ret
        else:
            return None

    def get_coords_frac(self):
        if self.check_set_attr('_dump_dct'):
            keys = 'xsu ysu zsu'.split()
            return self._lmp_dump2arr3d(self._dump_dct, keys)
        else:
            return None
    
    def get_coords(self):
        if self.check_set_attr('_dump_dct'):
            keys = 'xu yu zu'.split()
            return self._lmp_dump2arr3d(self._dump_dct, keys)
        else:
            return None
    
    def get_forces(self):
        if self.check_set_attr('_dump_dct'):
            keys = 'fx fy fz'.split()
            return self._lmp_dump2arr3d(self._dump_dct, keys)
        else:
            return None
    
    def get_velocity(self):
        if self.check_set_attr('_dump_dct'):
            keys = 'vx vy vz'.split()
            return self._lmp_dump2arr3d(self._dump_dct, keys)
        else:
            return None
    
    def get_timestep(self):
        if os.path.exists(self.filename):
            cmd = r"grep -m1 timestep %s | \
                    sed -re 's/.*step (.*)/\1/'" %self.filename
            return float_from_txt(com.backtick(cmd))
        else:
            return None

    def get_symbols(self):
        if os.path.exists(self.symbolsfilename):
            return com.file_read(self.symbolsfilename).split()
        elif self.check_set_attr_lst(['_dump_dct','natoms']):
            if self._dump_dct.has_key('type'):
                if self.order is None:
                   cmd = r"grep -m1 'dump_modify.*element' %s | sed -re \
                           's/.*ment (.*)/\1/'" %self.filename
                   revorder = dict((ii+1,sy) for ii,sy in \
                              enumerate(com.backtick(cmd).split()))        
                else:
                    # {'a':1,'b':2} -> {1:'a',2:'b'}
                    revorder = dict((v,k) for k,v in self.order.iteritems())
                return [revorder[int(ii)] for ii in self._dump_dct['type'][:self.natoms]]
            else:
                return None
        else:
            return None
    
    def get_cell(self):
        if os.path.exists(self.dumpfilename):
            cmd = r"grep -c 'ITEM: BOX BOUNDS' %s" %self.dumpfilename
            nstep = nstep_from_txt(com.backtick(cmd))
            cmd = r"grep -A3 'ITEM: BOX BOUNDS' %s | \
                    egrep -ve '--|ITEM'" %self.dumpfilename
            arr = np.fromstring(com.backtick(cmd), sep=' ').reshape((nstep,3,3))
            cell = np.zeros_like(arr)
            for ii in range(nstep):
                xlo_bound = arr[ii,0,0]
                xhi_bound = arr[ii,0,1]
                ylo_bound = arr[ii,1,0]
                yhi_bound = arr[ii,1,1]
                zlo = arr[ii,2,0]
                zhi = arr[ii,2,1]
                xy =  arr[ii,0,2]
                xz =  arr[ii,1,2]
                yz =  arr[ii,2,2]
                xlo = xlo_bound - min(0.0,xy,xz,xy+xz)
                xhi = xhi_bound - max(0.0,xy,xz,xy+xz)
                ylo = ylo_bound - min(0.0,yz)
                yhi = yhi_bound - max(0.0,yz)
                # [[x,  0,  0],
                #  [xy, y,  0],
                #  [xz, yz, z]]
                cell[ii,0,0] = xhi-xlo
                cell[ii,1,1] = yhi-ylo
                cell[ii,2,2] = zhi-zlo
                cell[ii,1,0] = xy
                cell[ii,2,0] = xz
                cell[ii,2,1] = yz
            return cell                
        else:
            return None


class LammpsDcdMDOutputFile(LammpsTextMDOutputFile):
    """Parse Lammps DCD binary output + ``log.lammps`` text output.

    Hardcodes files:
        | `dcdfilename` = ``basedir/lmp.out.dcd``

    Notes
    -----
    * cell: The DCD file format stores only cryst_const (see ``unitcell`` in
      dcd.f90). ``Trajectory.cell`` calculated from ``Trajectory.cryst_const``
      is aligned in the same way as ``LammpsTextMDOutputFile.cell``. That's why
      the cell and cryst_const obtained from
      :func:`~pwtools.io.read_lammps_md_dcd()` and
      :func:`~pwtools.io.read_lammps_md_txt()` must be identical up to
      numerical noise (about 1e-6 for default lammps text printing precision).
    """
    def __init__(self, *args, **kwds):
        super(LammpsDcdMDOutputFile, self).__init__(*args, **kwds)
        self.attr_lst = [\
            'cryst_const',
            'coords',
            'ekin',
            'etot',
            'natoms',
            'stress',
            'symbols',
            'temperature',
            'timestep',
            'volume',
        ]
        self.init_attr_lst()
        self.dcdfilename = pj(self.basedir, 'lmp.out.dcd')

    def _get_header(self):
        if os.path.exists(self.dcdfilename):
            a, b, c = read_dcd_header(self.dcdfilename)
            return {'nstep': a, 'natoms': b, 'timestep': c}
        else:
            return None
    
    def _get_cryst_const_coords(self):
        if self.check_set_attr_lst(['nstep', 'natoms']) and \
           os.path.exists(self.dcdfilename):
            a, b = read_dcd_data(self.dcdfilename, self.nstep, self.natoms)
            return {'cryst_const': a, 'coords': b}
        else:
            return None
    
    def get_coords(self):
        if self.check_set_attr('_cryst_const_coords'):
            return self._cryst_const_coords['coords']
        else:
            return None

    def get_cryst_const(self):
        if self.check_set_attr('_cryst_const_coords'):
            return self._cryst_const_coords['cryst_const']
        else:
            return None

    def get_natoms(self):
        if self.check_set_attr('_header'):
            return self._header['natoms']
        else:
            return None

    def get_nstep(self):
        if self.check_set_attr('_header'):
            return self._header['nstep']
        else:
            return None

