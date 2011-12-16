"""
parse.py

Parser classes for different file formats. Input- and output files.
 
We need the following basic Unix tools installed:
  grep/egrep
  sed
  awk
  tail
  wc 
  ...

Notes:
* Some functions/methods are Python-only (mostly for historical reasons ..
  code was written once and still works), but most of them actually call
  grep/sed/awk. This may not be pythonic, but hey ... these tools rock and
  the cmd lines are short.
* pwtools.com.backtick() takes "long" to create a child process. So for small
  files, pure-python versions, although they have much more code, are faster. 
  But who cares if the files are small. For big files, grep&friends win + much
  less code here.
* The tested egrep versions don't know the "\s" character class for
  whitespace as sed, Perl, Python or any other sane regex implementation
  does. Use "[ ]" instead.


Using Parsing classes
---------------------

All parsing classes 
    Pw*OutputFile
    Abinit*OutputFile
    Cpmd*OutputFile
are derived from FlexibleGetters -> FileParser.

As a general rule: If a getter (self.get_<attr>() or self._get_<attr>_raw()
cannot find anything in the file, it returns None. All getters which depend
on it will also return None.

* After initialization
      pp = SomeParsingClass(<filename>), all attrs whoose name is in 
  pp.attr_lst will be set to None.

* parse() will invoke self.check_get_attr(<attr>), which does 
      self.<attr> = self.get_<attr>() 
  for each <attr> in self.attr_lst, thus setting self.<attr> to a defined
  value: None if nothing was found in the file or not None else

* All getters get_<attr>() will do their parsing action, possibly
  looking for a file self.filename, regardless of the fact that the attribute
  self.<attr> may already be defined (e.g. if parse() has been called before).

* For interactive use (you need <attr> only once), prefer get_<attr>() over
  parse().

* Use dump(foo.pk) only for temporary storage and fast re-reading. Use
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
    cell + cryst_const + cryst_const_angles_lengths
    forces + forces_rms
    _<attr>_raw + <attr> (where <attr> = cell, forces, ...)
    ...
especially in MD parsers, not so much in StructureFileParser drived
classes. If parse() is used, all this information retrieved and stored.

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
import cPickle
import types

import numpy as np

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Warning: Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work." %__file__)

# XML parser
try:
    from BeautifulSoup import BeautifulStoneSoup
except ImportError:
    print("%s: Warning: Cannot import BeautifulSoup. " 
    "Parsing XML/HTML/CML files will not work." %__file__)

from pwtools import io, common, constants, regex, crys, periodic_table, num
from pwtools.verbose import verbose
from pwtools.pwscf import atpos_str
com = common


#-----------------------------------------------------------------------------
# General helpers
#-----------------------------------------------------------------------------

def next_line(fh):
    """Will raise StopIteration at end of file ."""
    return fh.next().strip()


def scan_until_pat(fh, pat="atomic_positions", err=True, retline=False):
    """
    Go to pattern `pat` in file `fh` and return `fh` at this position. Usually
    this is a header followed by a table in a pw.x input or output file.

    args:
    ----
    fh : file like object
    pat : string, pattern in lower case
    err : bool, raise error at end of file b/c pattern was not found
    retline : bool, return current line
    
    returns:
    --------
    fh : file like object
    flag : int
    {line : current line}
    
    notes:
    ------
    Currently all lines in the file are converted to lowercase before scanning.

    examples:
    ---------
    example file (pw.x input):
        
        [...]
        ATOMIC_POSITIONS alat                              <---- fh at this pos
        Al       4.482670384  -0.021685570   4.283770714         returned
        Al       2.219608875   1.302084775   8.297440557
        Si      -0.015470487  -0.023393016   1.789590196
        Si       2.194751751   1.364416814   5.817547157
        [...]
    """
    # About line returning:
    # cannot use:
    #
    #   >>> fh, flag = scan_until_pat(fh, ...)
    #   # do something with the current line
    #   >>> line = fh.readline()
    #   <type 'exceptions.ValueError'>: Mixing iteration and read methods would
    #   lose data
    # 
    # Must return line at file position instead.
    for line in fh:
        if line.strip().lower().startswith(pat):
            if retline:
                return fh, 1, line
            return fh, 1
    if err:
        raise StandardError("end of file '%s', pattern "
            "'%s' not found" %(fh, pat))
    # nothing found = end of file
    if retline:
        return fh, 0, line
    return fh, 0


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
    """Used for 3d trajectories where the exact shape must be known (N,3,nstep)
    where N=3 (cell, stresstensor) or N=natoms (coords, forces, ...). 
    """
    if txt.strip() == '':
        return None
    else:
        return io.readtxt(StringIO(txt), axis=axis, shape=shape, dtype=dtype)

def arr1d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        return np.atleast_1d(np.loadtxt(StringIO(txt), dtype=dtype))

def arr2d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        return np.atleast_2d(np.loadtxt(StringIO(txt), dtype=dtype))

def axis_lens(seq, axis=0):
    """
    example:
    --------
    >>> axis_lens([arange(100), np.array([1,2,3]), None, rand(5,3)])
    [100, 3, 0, 5]
    """
    ret = []
    for xx in seq:
        try:
            ret.append(xx.shape[axis])
        except AttributeError:
            ret.append(0)
    return ret            

#-----------------------------------------------------------------------------
# Parsers
#-----------------------------------------------------------------------------

# XXX Can this be done using @property ? 
class FlexibleGetters(object):
    """Base class. Implements a mechanism which allows to call getters in
    arbitrary order, even if they depend on each other. The mechanism also
    assured that the code in each getter is only executed once (by using checks
    with self.is_set_attr()).
    
    For each attr, there must exist a getter. We define the convention 
      self.foo  -> self.get_foo() 
      self.bar  -> self.get_bar()  
      self._baz -> self._get_baz() # note the underscores
      ... 
    
    self.attr_lst is an *optional* list of strings, each is the name of a data
    attribute, e.g. ['foo', 'bar', '_baz', ...].       
    Derived classes can override self.attr_lst by using self.set_attr_lst().
    
    Example:
        class MySuperParsingClass(FlexibleGetters):
            def __init__(self):
                self.set_attr_lst(['foo', 'bar', '_baz'])
                self.get_all()
            
            def get_all(self):
                "Sets self.foo, self.bar and self._baz by calling their
                getters"
                for attr in self.attr_lst:
                    self.check_get_attr(attr)
            
            # Getters call each other
            def _get_baz(self):
                return self.calc_baz()
            
            def get_bar(self):
                self.check_get_attr('_baz')
                return None if (not self.is_set_attr('_baz')) else \
                    self.calc_stuff(self._baz)**2.0

            def get_foo(self):
                required = ['bar', '_baz']
                self.check_get_attrs(required)
                if self.is_set_attrs(required):
                    return do_stuff(self._baz, self.bar)
                else:
                    return None
    
    Setting self.attr_lst is optional. It is supposed to be used only in
    get_all(). The check_get_attr() - method works without it, too. 
    """ 
    # Notes for derived classes (long explaination):
    #
    # In this class we define a number of members (self.foo, self.bar,
    # ...) which shall all be set by the get_all() method.
    #
    # There are 3 ways of doing it:
    #
    # 1) Put all code in get_all(). 
    #    Con: One might forget to implement the setting of a member.
    # 
    # 2) Implement get_all() so that for each data member of the API, we have
    #       self.foo = self.get_foo()
    #       self.bar = self.get_bar()
    #       ...
    #    and put the code for each member in a separate getter. This is good
    #    coding style, but often data needs to be shared between getters (e.g.
    #    get_foo() needs bar, which is the result of self.bar =
    #    self.get_bar(). This means that in general the calling order
    #    of the getters is important and is different in each get_all() of each
    #    derived class.
    #    Con: One might forget to call a getter in get_all() and/or in the wrong 
    #         order.
    # 
    # 3) Implement all getters such that they can be called in arbitrary order.
    #    Then in each get_all(), one does exactly the same:
    #
    #        attr_lst = ['foo', 'bar', ...]
    #        for attr in attr_lst:
    #            self.check_get_attr(attr)
    #    
    #    This code (the "getting" of all API members) can then be moved to the
    #    *base* class's get_all() and thereby forcing all derived classes to
    #    conform to the API. 
    #
    #    If again one getter needs a return value of another getter, one has to
    #    transform
    #    
    #       def get_foo(self):
    #           return do_stuff(self.bar)
    #    to 
    #       
    #       def get_foo(self):
    #           self.check_get_attr('bar')                <<<<<<<<<<<<
    #           return do_stuff(self.bar)
    #
    #    If one does
    #        self.foo = self.get_foo()
    #        self.bar = self.get_bar()
    #        ....
    #    then some calls may in fact be redundant b/c e.g. get_foo() has
    #    already been called inside get_bar(). There is NO big overhead in
    #    this approach b/c in each getter we test with check_get_attr() if a
    #    needed other member is already set.
    #    
    #    This way we get a flexible and easily extensible framework to
    #    implement new parsers and modify existing ones (just implement another
    #    getter get_newmember() in each class and extend the list of API
    #    members by 'newmember').
    #
    #    One drawback: Beware of cyclic dependencies (i.e. get_bar ->
    #    get_foo -> get_bar -> ...). Always test the implementation!
    
    def __init__(self):
        self.set_attr_lst([])

    def set_attr_lst(self, attr_lst):
        """Set self.attr_lst and init each attr to None."""
        self.attr_lst = attr_lst
        for attr in self.attr_lst:
            setattr(self, attr, None)

    def dump(self, dump_filename):
        """Pickle (write to binary file) the whole object."""
        # Dumping with protocol "2" is supposed to be the fastest binary format
        # writing method. Probably, this is platform-specific.
        cPickle.dump(self, open(dump_filename, 'wb'), 2)

    def load(self, dump_filename):
        """Load pickled object.
        
        example:
        --------
        # save
        >>> x = FileParser('foo.txt')
        >>> x.parse()
        >>> x.dump('foo.pk')
        # load: method 1
        >>> xx = FileParser()
        >>> xx.load('foo.pk')
        # load: method 2, probably easier :)
        >>> xx = cPickle.load(open('foo.pk'))
        # or 
        >>> xx = common.cpickle_load('foo.pk')
        """
        # this does not work:
        #   self = cPickle.load(...)
        self.__dict__.update(cPickle.load(open(dump_filename, 'rb')).__dict__)
    
    def is_set_attr(self, attr):
        """Check if self has the attribute self.<attr> and if it is _not_ None.

        args:
        -----
        attr : str
            Attrubiute name, e.g. 'foo' for self.foo
        
        returns:
        --------
        True : `attr` is defined and not None
        False : not defined or None
        """
        if hasattr(self, attr): 
            return (getattr(self, attr) is not None)
        else:
            return False
    
    def is_set_attrs(self, attr_lst):
        assert common.is_seq(attr_lst), "attr_lst must be a sequence"
        for attr in attr_lst:
            if not self.is_set_attr(attr):
                return False
        return True                

    def check_get_attr(self, attr):
        """If self.<attr> does not exist or is None, then invoke an
        appropirately named getter as if this command would be executed:
        
        self.foo = self.get_foo()
        self._foo = self._get_foo()
        """
        if not self.is_set_attr(attr):
            if attr.startswith('_'):
                get = '_get'
            else:
                get = 'get_'
            setattr(self, attr, eval('self.%s%s()' %(get, attr))) 
    
    def assert_attr(self, attr):
        if not self.is_set_attr(attr):
            raise AssertionError("attr '%s' is not set" %attr)
    
    def assert_get_attr(self, attr):
        self.check_get_attr(attr)
        self.assert_attr(attr)

    def check_get_attrs(self, attr_lst):
        for attr in attr_lst:
            self.check_get_attr(attr)
    
    def assert_attrs(self, attr_lst):
        for attr in attr_lst:
            self.assert_attr(attr)
    
    def assert_get_attrs(self, attr_lst):
        for attr in attr_lst:
            self.assert_get_attr(attr)
    
    def raw_slice_get(self, attr_name, sl, axis):
        """Shortcut method:
        * call check_get_attr(_<attr_name>_raw) -> set
          self._<attr_name>_raw to None or smth else
        * if set, return self._<attr_name>_raw sliced by `sl` along `axis`,
          else return None
        """
        verbose("getting %s" %attr_name)
        raw_attr_name = '_%s_raw' %attr_name
        self.check_get_attr(raw_attr_name)
        if self.is_set_attr(raw_attr_name):
            arr = getattr(self, raw_attr_name)
            ret = num.slicetake(arr, sl, axis) 
            # slicetake always returns an array, retuen scalar if ret =
            # array([10]) etc
            if (ret.ndim == 1) and (len(ret) == 1):
                return ret[0]
            else:
                return ret
        else:
            return None
    
    def raw_return(self, attr_name):
        """Call check_get_attr(_<attr_name>_raw) and return it if set, else
        None. This is faster but the same the same as 
            raw_slice_get(<attr_name>, sl=slice(None), axis=0)
        or axis=1 or any other valid axis.
        """
        verbose("getting %s" %attr_name)
        raw_attr_name = '_%s_raw' %attr_name
        self.check_get_attr(raw_attr_name)
        if self.is_set_attr(raw_attr_name):
            return getattr(self, raw_attr_name)
        else:
            return None
    

class FileParser(FlexibleGetters):
    """Base class for file parsers.
    
    All getters are called in the default self.parse() which can be overridden
    in derived classes.
    """
    def __init__(self, filename=None):
        """
        args: 
        -----
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
    
    def __del__(self):
        """Destructor. If self.fd has not been closed yet (in self.parse()),
        then do it here, eventually."""
        self.close_file()
    
    def _is_file_open(self):
        return (self.fd is not None) and (not self.fd.closed)

    def close_file(self):
        if self._is_file_open():
            self.fd.close()
    
    def parse(self):
        for attr in self.attr_lst:
            self.check_get_attr(attr)
        self.close_file()

    def get_txt(self):
        if self._is_file_open():
            self.fd.seek(0)    
            return self.fd.read().strip()
        else:
            raise StandardError("self.fd is None or closed")


class StructureFileParser(FileParser):
    """Base class for structure file (pdb, cif, etc) and input file parsers.
    A file parsed by this class is supposed to contain infos about an atomic
    structure. This is like ase.Atoms.
    
    Classes derived from this one must provide the following members. If a
    particular information is not present in the parsed file, the corresponding
    member must be None.
    
    parsing results
    ---------------
    self.coords : ndarray (natoms, 3) with atom coords
    self.symbols : list (natoms,) with strings of atom symbols, must match the
        order of the rows of self.coords
    self.cell : 3x3 array with primitive basis vectors as rows, for
        PWscf, the array is in units of (= divided by) alat == self.cryst_const[0]
    self.cryst_const : array (6,) with crystallographic costants
        [a,b,c,alpha,beta,gamma]
    self.natoms : number of atoms
    
    Note that cell and cryst_const contain the same information (redundancy).

    convenience getters
    -------------------
    get_atpos_str     
    get_mass         
    
    units
    -----
    Unless explicitly stated, we DO NOT DO any unit conversions with the data
    parsed out of the files. It is up to the user (and derived classes) to
    handle that. 
    """
    def __init__(self, filename=None):
        FileParser.__init__(self, filename)
        self.set_attr_lst(['coords', 'symbols', 'cryst_const', 'cell',
                           'natoms'])

    def get_atpos_str(self):
        """Return a string representing the ATOMIC_POSITIONS card in a pw.x
        in/out file."""
        self.check_get_attr('coords')
        self.check_get_attr('symbols')
        return atpos_str(self.symbols, self.coords)
    
    def get_mass(self):
        """1D array of atomic masses in amu (atomic mass unit 1.660538782e-27
        kg as in periodic table). The order is the one from self.symbols."""
        self.check_get_attr('symbols')
        if self.is_set_attr('symbols'):
            return np.array([periodic_table.pt[sym]['mass'] for sym in
                             self.symbols])
        else:
            return None
    

class CifFile(StructureFileParser):
    """Parse Cif file.

    notes:
    ------
    cif parsing:
        We expect PyCifRW [1] to be installed, which provides the CifFile
        module.
    atom positions:
        Cif files contain "fractional" coords, which is just 
        "ATOMIC_POSITIONS crystal" in PWscf, "xred" in Abinit.
    
    refs:
    -----
    [1] http://pycifrw.berlios.de/
    [2] http://www.quantum-espresso.org/input-syntax/INPUT_PW.html#id53713
    """        
    def __init__(self, filename=None, block=None):
        """        
        args:
        -----
        filename : name of the input file
        block : data block name (i.e. 'data_foo' in the Cif file -> 'foo'
            here). If None then the first data block in the file is used.
        
        """
        StructureFileParser.__init__(self, filename)
        self.block = block
    
    def cif_str2float(self, st):
        """'7.3782(7)' -> 7.3782"""
        if '(' in st:
            st = re.match(r'(' + regex.float_re  + r')(\(.*)', st).group(1)
        return float(st)

    def cif_clear_atom_symbol(self, st, rex=re.compile(r'([a-zA-Z]+)([0-9+-]*)')):
        """Remove digits and "+,-" from atom names. 
        
        example:
        -------
        >>> cif_clear_atom_symbol('Al1')
        'Al'
        """
        return rex.match(st).group(1)
    
    def _get_cif_dct(self):
        # celldm from a,b,c and alpha,beta,gamma
        # alpha = angbe between b,c
        # beta  = angbe between a,c
        # gamma = angbe between a,b
        self.check_get_attr('_cif_block')
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
    
    def get_coords(self):
        self.check_get_attr('_cif_block')
        return np.array([map(self.cif_str2float, [x,y,z]) for x,y,z in izip(
                                   self._cif_block['_atom_site_fract_x'],
                                   self._cif_block['_atom_site_fract_y'],
                                   self._cif_block['_atom_site_fract_z'])])
        

    def get_symbols(self):
        self.check_get_attr('_cif_block')
        try_lst = ['_atom_site_type_symbol', '_atom_site_label']
        for entry in try_lst:
            if self._cif_block.has_key(entry):
                return map(self.cif_clear_atom_symbol, self._cif_block[entry])
        return None                
    
    def get_cryst_const(self):
        self.check_get_attr('_cif_dct')
        return np.array([self._cif_dct[key] for key in \
            ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
    
    def get_cell(self):
        self.check_get_attr('cryst_const')
        return crys.cc2cell(self.cryst_const)

    def get_natoms(self):
        self.check_get_attr('symbols')
        return len(self.symbols)
    

class PDBFile(StructureFileParser):
    """Very very simple pdb file parser. Extract only ATOM/HETATM and CRYST1
    (if present) records.
        
    If you want smth serious, check biopython.
    
    members:
    --------
    See StructureFileParser

    notes:
    ------
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
    def __init__(self, filename=None):
        """
        args:
        -----
        filename : name of the input file
        """
        StructureFileParser.__init__(self, filename)
    
    def _get_coords_data(self):
        self.fd.seek(0)
        ret = com.igrep(r'(ATOM|HETATM)[\s0-9]+([A-Za-z]+)[\sa-zA-Z0-9]*'
            r'[\s0-9]+((\s+'+ regex.float_re + r'){3}?)', self.fd)
        # array of string type            
        return np.array([[m.group(2)] + m.group(3).split() for m in ret])
    
    def get_symbols(self):
        # list of strings (system:nat,) 
        # Fix atom names, e.g. "AL" -> Al. Note that this is only needed b/c we
        # use the "wrong" column "Atom name".
        self.check_get_attr('_coords_data')
        symbols = []
        for sym in self._coords_data[:,0]:
            if len(sym) == 2:
                symbols.append(sym[0] + sym[1].lower())
            else:
                symbols.append(sym)
        return symbols

    def get_coords(self):
        self.check_get_attr('_coords_data')
        # float array, (system:nat, 3)
        return self._coords_data[:,1:].astype(float)        
    
    def get_cryst_const(self):
        # grep CRYST1 record, extract only crystallographic constants
        # example:
        # CRYST1   52.000   58.600   61.900  90.00  90.00  90.00  P 21 21 21   8
        #          a        b        c       alpha  beta   gamma  |space grp|  z-value
        self.fd.seek(0)
        ret = com.mgrep(r'CRYST1\s+((\s+'+ regex.float_re + r'){6}).*$', self.fd)
        if len(ret) == 1:
            match = ret[0]
            return np.array(match.group(1).split()).astype(float)
        elif len(ret) == 0:
            return None
        else:
            raise StandardError("found CRYST1 record more then once")
    
    def get_cell(self):
        self.check_get_attr('cryst_const')
        return crys.cc2cell(self.cryst_const)            
    
    def get_natoms(self):
        self.check_get_attr('symbols')
        return len(self.symbols)


class CMLFile(StructureFileParser):
    """Parse Chemical Markup Language files (XML-ish format). This file format
    is used by avogadro."""
    def __init__(self, filename=None):
        StructureFileParser.__init__(self, filename)
    
    def _get_soup(self):
        return BeautifulStoneSoup(open(self.filename).read())        

    def _get_atomarray(self):
        self.check_get_attr('_soup')
        # ret: list of Tag objects:
        # <atomarray>
        #    <atom id="a1" ...>
        #    <atom id="a2" ...>
        #    ...
        # </atomarray>
        # ==>
        # [<atom id="a1" ...>, <atom id="a2" ...>, ...]
        return self._soup.find('atomarray').findAll('atom')
    
    def get_coords(self):
        self.check_get_attr('_atomarray')
        return np.array([[float(entry.get('xfract')), 
                          float(entry.get('yfract')),
                          float(entry.get('zfract'))] \
                          for entry in self._atomarray])
    
    def get_symbols(self):
        self.check_get_attr('_atomarray')
        return [str(entry.get('elementtype')) for entry in self._atomarray]
    
    def get_cryst_const(self):
        self.check_get_attr('_soup')
        crystal = self._soup.find('crystal')            
        return np.array([crystal.find('scalar', title="a").string,
                         crystal.find('scalar', title="b").string,
                         crystal.find('scalar', title="c").string,
                         crystal.find('scalar', title="alpha").string,
                         crystal.find('scalar', title="beta").string,
                         crystal.find('scalar', title="gamma").string]\
                         ).astype(float)
    
    def get_cell(self):
        self.check_get_attr('cryst_const')
        return crys.cc2cell(self.cryst_const)

    def get_natoms(self):
        self.check_get_attr('symbols')
        return len(self.symbols)            



class PwInputFile(StructureFileParser):
    """Parse Pwscf input file.

    members:
    --------
    See StructureFileParser
    
    extra members:
    --------------
    atspec : dict 
        see self.get_atspec()
    atpos : dict 
        see self.get_atpos()
    namelists : dict
        see self.get_namelists()
    kpoints : dict
        see self.get_kpoints()
    mass : 1d array (natoms,)
        Array of masses of all atoms in the order listed in
        ATOMIC_POSITIONS. This is actually self.atpos['mass'].
    
    notes:
    ------
    self.cell is parsed from CELL_PARAMETERS, and no unit conversion is done
    for this, but a conversion is attempted for cryst_const, if alat is found.
    
    PWscf unit conventions are a little wicked. In PWscf, the first lattice
    constant "a" is named alat (=celldm(1)). If that is present in the file,
    then CELL_PARAMETERS in pw.in is in units of alat, i.e. the "normal" cell
    in Bohr or Angstroms divided by alat. If not, then CELL_PARAMETERS is
    assumed to be the "normal" cell, i.e. alat is unity.
    
    Now the magic: If we have an entry in pw.in to determine alat:
    system:celldm(1) or system:A, then self.cell will be multiplied with that
    *only* for the calculation of self.cryst_const. Then [a,b,c] =
    cryst_const[:3] will have the right unit (e.g. Bohr). A warning will be
    issued if neither is found. self.cell will be returned as it is in the
    file, which will in fact be the "normal" cell. 
    
    So, if alat is found, you *cannot* convert back and forth between cell and
    cryst_const by using e.g. crys.cell2cc()/cc2cell(), but only between
    cell*alat and cryst_const !
    """

    def __init__(self, filename=None):
        StructureFileParser.__init__(self, filename)
        # All card names that may follow the namelist section in a pw.x input
        # file. Lowercase here, in pw.in files in general uppercase.
        self.pwin_cardnames = [\
            'atomic_species',
            'atomic_positions',
            'k_points',
            'cell_parameters',
            'occupations',
            'climbing_images',
            'constraints',
            'collective_vars']
        self.set_attr_lst(self.attr_lst + ['atspec', \
                                           'atpos', \
                                           'namelists', \
                                           'kpoints', \
                                           'mass'])
    
    def get_mass(self):
        self.check_get_attr('atpos')
        return self.atpos['mass']

    def get_symbols(self):
        self.check_get_attr('atpos')
        return self.atpos['symbols']

    def get_coords(self):
        self.check_get_attr('atpos')
        return self.atpos['coords']

    def get_natoms(self):
        self.check_get_attr('namelists')
        return int(self.namelists['system']['nat'])

    def get_cryst_const(self):
        self.check_get_attr('cell')
        self.check_get_attr('namelists')
        alat_conv_dct = {'celldm(1)': 1,
                         'a': 1/constants.a0_to_A}
        nl_system = self.namelists['system']
        alat_found = False
        for key, conv_fac in alat_conv_dct.iteritems():
            if nl_system.has_key(key):
                celldm1 = float(nl_system[key])*conv_fac
                alat_found = True
                break
        if not alat_found:
            print("[PwInputFile:get_cryst_const] Warning: no alat found. "
                  "Using celldm1=1.0")
            celldm1 = 1.0
        return None if (self.cell is None) else \
               crys.cell2cc(self.cell*celldm1)

    def get_atspec(self):
        """Parses ATOMIC_SPECIES card in a pw.x input file.

        returns:
        --------
        {'symbols': symbols, 'masses': masses, 'pseudos': pseudos}
        
        symbols : list of strings, (number_of_atomic_species,), 
            ['Si', 'O', 'Al', 'N']
        masses : 1d array, (number_of_atomic_species,)
            array([28.0855, 15.9994, 26.981538, 14.0067])
        pseudos : list of strings, (number_of_atomic_species,)
            ['Si.LDA.fhi.UPF', 'O.LDA.fhi.UPF', 'Al.LDA.fhi.UPF',
            'N.LDA.fhi.UPF']
        
        notes:
        ------
        scan for:

        [...]
        ATOMIC_SPECIES|atomic_species
        [possible empty lines]
        Si  28.0855     Si.LDA.fhi.UPF
        O   15.9994     O.LDA.fhi.UPF   
        Al  26.981538   Al.LDA.fhi.UPF
        N   14.0067     N.LDA.fhi.UPF
        [...]
        """
        self.fd.seek(0)
        verbose('[get_atspec] reading ATOMIC_SPECIES from %s' %self.filename)
        rex = re.compile(r'\s*([a-zA-Z]+)\s+(' + regex.float_re +\
            ')\s+(.*)$')
        self.fd, flag = scan_until_pat(self.fd, 
                                         pat='atomic_species',        
                                         err=False)
        # XXX from here on, could use common.igrep() or re.findall()
        if flag == 0:
            verbose("[get_atspec]: WARNING: start pattern not found")
            return None
        line = next_line(self.fd)
        while line == '':
            line = next_line(self.fd)
        match = rex.match(line)
        lst = []
        # XXX Could use knowledge of namelists['system']['ntyp'] here (=number
        # of lines in this card) if we parse the namelists first
        while match is not None:
            # match.groups: tuple ('Si', '28.0855', 'Si.LDA.fhi.UPF')
            lst.append(list(match.groups()))
            line = next_line(self.fd)
            match = rex.match(line)
        if lst == []:
            return None
        # numpy string array :)
        ar = np.array(lst)
        symbols = ar[:,0].tolist()
        masses = np.asarray(ar[:,1], dtype=float)
        pseudos = ar[:,2].tolist()
        return {'symbols': symbols, 'masses': masses, 'pseudos': pseudos}


    def get_cell(self):
        """Parses CELL_PARAMETERS card in a pw.x input file. Extract primitive
        lattice vectors.

        notes:
        ------
        From the PWscf help:
            
        --------------------------------------------------------------------
        Card: CELL_PARAMETERS { cubic | hexagonal }
        
        Optional card, needed only if ibrav = 0 is specified, ignored
        otherwise!
        
        Flag "cubic" or "hexagonal" specify if you want to look for symmetries
        derived from the cubic symmetry group (default) or from the hexagonal
        symmetry group (assuming c axis as the z axis, a axis along the x
        axis).
        
        v1, v2, v3  REAL

            Crystal lattice vectors:
            v1(1)  v1(2)  v1(3)    ... 1st lattice vector
            v2(1)  v2(2)  v2(3)    ... 2nd lattice vector
            v3(1)  v3(2)  v3(3)    ... 3rd lattice vector

            In alat units if celldm(1) was specified or in a.u. otherwise.
        --------------------------------------------------------------------
        
        In a.u. = Rydberg atomic units (see constants.py).

        This would also work:
            >>> cmd = r"sed -nre '/CELL_PARAMETERS/,+3p' %s | tail -n3" %self.filename
            >>> cell = np.loadtxt(StringIO(com.backtick(cmd)))
        """
        self.fd.seek(0)
        verbose('[get_cell] reading CELL_PARAMETERS from %s' %self.filename)
        rex = re.compile(r'\s*((' + regex.float_re + '\s*){3})\s*')
        self.fd, flag = scan_until_pat(self.fd, pat="cell_parameters",
                                         err=False)
        if flag == 0:
            return None
        line = next_line(self.fd)
        while line == '':
            line = next_line(self.fd)
        match = rex.match(line)
        lst = []
        # XXX Could use <number_of_lines> = 3 instead of regexes
        while match is not None:
            # match.groups(): ('1.3 0 3.0', ' 3.0')
            lst.append(match.group(1).strip().split())
            line = next_line(self.fd)
            match = rex.match(line)
        if lst == []:
            return None
        cell = np.array(lst, dtype=float)
        com.assert_cond(len(cell.shape) == 2, "`cell` is no 2d array")
        com.assert_cond(cell.shape[0] == cell.shape[1], "dimensions of `cell` don't match")
        return cell

    def get_atpos(self):
        """Parse ATOMIC_POSITIONS card in pw.x input file.
        
        returns:
        --------
        {'coords': coords, 'natoms': natoms, 'mass': mass, 'symbols':
        symbols, 'unit': unit}
        
        coords : ndarray,  (natoms, 3)
        natoms : int
        mass : 1d array, (natoms,)
        symbols : list of strings, (natoms,), ['Al', 'A', 'Si', ...]
        unit : string, 'alat', 'crystal', etc.

        notes:
        ------
        scan for:
        
        [...]
        ATOMIC_POSITIONS|atomic_positions [<unit>]
        [0 or more empty lines]
        Al       4.482670384  -0.021685570   4.283770714
        Al       2.219608875   1.302084775   8.297440557
        ...
        Si       2.134975048   1.275864192  -0.207552657
        [empty or nonempty line that does not match the RE]
        [...]

        <unit> is a string: 'alat', 'crystal' etc.
        """
        self.check_get_attr('atspec')
        self.fd.seek(0)
        verbose("[get_atpos] reading ATOMIC_POSITIONS from %s" %self.filename)
        # XXX {3}: will not work for atomic pos. w/ fixed DOFs, i.e.
        #   Al 0.9  0.5   0.1  0 0 1
        rex = re.compile(r'\s*([a-zA-Z]+)((\s+' + regex.float_re + '){3})\s*')
        self.fd, flag, line = scan_until_pat(self.fd, 
                                               pat="atomic_positions", 
                                               retline=True)
        if flag == 0:
            return None
        line = line.strip().lower().split()
        if len(line) > 1:
            unit = re.sub(r'[{\(\)}]', '', line[1])
        else:
            unit = ''
        line = next_line(self.fd)
        while line == '':
            line = next_line(self.fd)
        lst = []
        # XXX Instead of regexes, we could as well use natoms
        # (namelists['system']['nat']).
        match = rex.match(line)
        while match is not None:
            # match.groups():
            # ('Al', '       4.482670384  -0.021685570   4.283770714', '    4.283770714')
            lst.append([match.group(1)] + match.group(2).strip().split())
            line = next_line(self.fd)
            match = rex.match(line)
        if lst == []:
            return None
        ar = np.array(lst)
        symbols = ar[:,0].tolist()
        # same as coords = np.asarray(ar[:,1:], dtype=float)
        coords = ar[:,1:].astype(float)
        natoms = coords.shape[0]
        masses = self.atspec['masses']
        atspec_symbols = self.atspec['symbols']
        mass = np.array([masses[atspec_symbols.index(s)] for s in symbols], dtype=float)
        return {'coords': coords, 'natoms': natoms, 'mass': mass, 'symbols':
            symbols, 'unit': unit}


    def _is_cardname(self, line):
        """Helper for self.get_namelists()."""
        for string in self.pwin_cardnames:
            # matches "occupations", but not "occupations='semaring'"
            if re.match(r'^\s*%s\s*([^=].*$|$)' %string, line.lower()):
                return True
        return False            


    def get_namelists(self):
        """
        Parse "namelist" part of a pw.x input file.

        notes:
        ------
        file to parse:
            [...]
            &namelist1
                FLOATKEY = 1.0d0, stringkey = 'foo' [,]
                boolkey = .true. [,]
                intkey = 10 [,]
                ...
            /
            &namelist2
                floatkey = 2.0d0, stringkey = 'bar' [,]
                boolkey = .false. [,]
                intkey = -10 [,]
                ...
            /
            ...

            [CARD SECTION]
            ATOMIC_SPECIES
            [...]
            
        
        returns: 
        --------
        dict of dicts:
            
            {namelist1:
                {'floatkey': '1.0d0', 'stringkey': "'foo'",
                  boolkey = '.true.', intkey = '10'},
             namelist2:
                {'floatkey': '2.0d0', 'stringkey': "'bar'",
                  boolkey = '.false.', intkey = '-10'},
            }
            
        All keys are converted to lowercase! All values are strings and must be
        converted.
        >>> d = get_namelists(...)
        >>> floatkey  = com.ffloat(d['namelist1']['floatkey'])
        >>> stringkey = d['namelist1']['stringkey']
        >>> intkey    = int(d['namelist1']['intkey'])
        >>> boolkey   = com.tobool(d['namelist1']['boolkey'])
        """
        self.fd.seek(0)
        verbose("[get_namelists] parsing %s" %self.filename)
        dct = {}
        nl_kvps = None
        for line in self.fd:
            # '   A = b, c=d,' -> 'A=b,c=d'
            line = line.strip().strip(',').replace(' ', '')
            # Start of namelist.
            if line.startswith('&'):
                # namelist key value pairs
                nl_kvps = []
                # '&CONTROL' -> 'control'
                nl = line[1:].lower()
            # End of namelist. Enter infos from last namelist into `dct`.           
            elif line == '/':
                # nl = 'control', enter dict for namelist 'control' in `dct` under
                # name 'control'.
                # [['a', 'b'], ['c', 'd']] -> dct = {'control': {'a': 'b', 'c': 'd'}, ...}
                if nl_kvps is not None: 
                    dct[nl] = dict(nl_kvps)
                    nl_kvps = None
                else:
                    dct[nl] = {}
            elif line == '' or line.startswith('!'):
                continue
            # end of namelist part
            elif self._is_cardname(line):
                break
            else:
                # 'A=b,c=d' -> ['A=b', 'c=d'] -> 
                # nl_kvps = [..., ['a', 'b'], ['c', 'd']]
                for p in line.split(','):
                    tmp = p.split('=')
                    tmp[0] = tmp[0].lower()
                    # HACK: remove nested quotes: "'foo'" -> 'foo'
                    if tmp[1][0] == "'" and tmp[1][-1] == "'":
                        tmp[1] = eval(tmp[1])
                    nl_kvps.append(tmp)
        if dct == {}:
            verbose("[get_namelists]: WARNING: nothing found")
            return None
        return dct

    def get_kpoints(self):
        """Parse the K_POINTS card. The format is assumed to be
            K_POINTS <mode>
            [<kpoints>]
        
        example:
        --------
        K_POINTS automatic
        2 2 2 0 0 0
        => 'K_POINTS automatic\n2 2 2 0 0 0'
        => mode = 'automatic'
           kpoints = '2 2 2 0 0 0'
        
        # if mode = 'gamma', then we set kpoints ='gamma'    
        K_POINTS gamma
        => 'K_POINTS gamma'
        => mode = 'gamma'
           kpoints = 'gamma'
        """
        self.check_get_attr('txt')
        rex = re.compile(r'^\s*K_POINTS\s*(.*)\s*\n*(.*)$', re.M)
        m = rex.search(self.txt)
        mode = m.group(1).strip().lower()
        kpoints = m.group(2).strip().lower()
        if mode == 'gamma':
            if kpoints == '':
                kpoints = mode
            else:
                raise StandardError("K_POINTS mode = 'gamma' but kpoints != ''")
        return {'mode': mode,
                'kpoints': kpoints}


class PwSCFOutputFile(FileParser):
    """Parse a pw.x SCF output file. Some getters (_get_<attr>_raw) work for
    MD-like output, too. Here in the SCF case, only the first item along the
    time axis is returned and should only be used on calculation='scf' output.

    notes:
    ------
    total_force : Pwscf writes a "Total Force" after the "Forces acting on
        atoms" section . This value a UNnormalized RMS of the force matrix
        (f_ij, i=1,natoms j=1,2,3) printed. According to .../PW/forces.f90,
        variable "sumfor", the "Total Force" is
            sqrt(sum_ij f_ij^2)
        Use self.forces_rms for a normalized value.            
    """
    # self.time_axis: This is the hardcoded time axis. It must be done
    #     this way b/c getters returning a >2d array cannot determine the shape
    #     of the returned array auttomatically based on the self.time_axis
    #     setting alone. If you want to change this, then manually fix the
    #     "shape" kwarg to io.readtxt() in all getters which return a 3d array.
    def __init__(self, filename=None):
        """
        args:
        -----
        filename : file to parse
        """        
        FileParser.__init__(self, filename)
        self.time_axis = -1
        self.set_attr_lst([\
        'cell', 
        'coords', 
        'etot', 
        'forces',
        'forces_rms',
        'natoms', 
        'nkpoints',
        'nstep_scf', 
        'pressure', 
        'stresstensor', 
        'total_force',
        'volume',
        'scf_converged',
        ])
        
    def _get_stresstensor_raw(self):
        verbose("getting _stresstensor_raw")
        key = 'P='
        cmd = "grep %s %s | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e %s -e '--'| \
              awk '{printf $4\"  \"$5\"  \"$6\"\\n\"}'" \
              %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(3,3,nstep),
                             axis=self.time_axis)              

    def _get_etot_raw(self):
        verbose("getting _etot_raw")
        cmd =  r"grep '^!' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def _get_pressure_raw(self):
        verbose("getting _pressure_paw")
        cmd = r"grep P= %s | awk '{print $6}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
     
    def _get_forces_raw(self):
        verbose("getting _forces_raw")
        self.check_get_attr('natoms')
        natoms = self.natoms
        # nstep: get it from outfile b/c the value in any input file will be
        # wrong if the output file is a concatenation of multiple smaller files
        key = r'Forces\s+acting\s+on\s+atoms.*$'
        cmd = r"egrep '%s' %s | wc -l" %(key.replace(r'\s', r'[ ]'), self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        # forces
        cmd = "grep 'atom.*type.*force' %s \
            | awk '{print $7\" \"$8\" \"$9}'" %self.filename
        return traj_from_txt(com.backtick(cmd), 
                             shape=(natoms,3,nstep),
                             axis=self.time_axis)
    
    def _get_total_force_raw(self):
        verbose("getting _total_force_raw")
        cmd = r"egrep 'Total[ ]+force[ ]*=.*Total' %s \
            | sed -re 's/^.*Total\s+force\s*=\s*(.*)\s*Total.*/\1/'" \
            %self.filename
        return arr1d_from_txt(com.backtick(cmd))

    def _get_forces_rms_raw(self):
        verbose("getting _forces_rms_raw")
        req = ['_forces_raw']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.rms3d(self._forces_raw, axis=self.time_axis) 
        else:
            return None
    
    def _get_nstep_scf_raw(self):
        verbose("getting _nstep_scf_raw")
        cmd = r"grep 'convergence has been achieved in' %s | awk '{print $6}'" \
            %self.filename
        return arr1d_from_txt(com.backtick(cmd), dtype=int)

    def get_stresstensor(self):
        return self.raw_slice_get('stresstensor', sl=0, axis=self.time_axis)

    def get_etot(self):
        return self.raw_slice_get('etot', sl=0, axis=0)
    
    def get_pressure(self):
        return self.raw_slice_get('pressure', sl=0, axis=0)
    
    def get_forces(self):
        return self.raw_slice_get('forces', sl=0, axis=self.time_axis)
    
    def get_total_force(self):
        return self.raw_slice_get('total_force', sl=0, axis=0)
    
    def get_forces_rms(self):
        return self.raw_slice_get('forces_rms', sl=0, axis=0)
    
    def get_nstep_scf(self):
        return self.raw_slice_get('nstep_scf', sl=0, axis=0)
    
    def get_cell(self):
        """Grep start cell from pw.out. This is always in alat
        units, but printed with much less precision compared to the input file.
        If you need this information for further calculations, use the input
        file value."""
        verbose("getting start cell parameters")
        cmd = "grep -A3 'crystal.*axes.*units.*a_0' %s | tail -n3 | \
               awk '{print $4\" \"$5\" \"$6}'" %(self.filename)
        return arr2d_from_txt(com.backtick(cmd))
    
    def get_coords(self):
        """Grep start ATOMIC_POSITIONS from pw.out. This is always in cartesian
        alat units and printed with low precision. The start coords in the
        format given in the input file is just infile.coords. This should be
        the same as
        >>> pp = PwSCFOutputFile(...)
        >>> pp.parse()
        >>> crys.coord_trans(pp.coords, 
        >>>                  old=np.identity(3), 
        >>>                  new=pp.cell, 
        >>>                  align='rows')
        """
        verbose("getting start coords")
        self.check_get_attr('natoms')
        natoms = self.natoms
        cmd = r"grep -A%i 'positions.*a_0.*units' %s | tail -n%i | \
              sed -re 's/.*\((.*)\)/\1/g'" \
              %(natoms, self.filename, natoms)
        return arr2d_from_txt(com.backtick(cmd))

    def get_natoms(self):
        verbose("getting natoms")
        cmd = r"grep 'number.*atoms/cell' %s | head -n1 | \
              sed -re 's/.*=\s+([0-9]+).*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_nkpoints(self):
        verbose("getting nkpoints")
        cmd = r"grep 'number of k points=' %s | head -n1 | \
            sed -re 's/.*points=\s*([0-9]+)\s*.*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))

    def get_volume(self):
        verbose("getting volume")
        ret_str = com.backtick(r"grep 'unit-cell.*volume' %s | head -n1 | sed -re \
                                's/.*volume\s*=\s*(.*?)\s+.*a.u..*/\1/'"
                                %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return float(ret_str)

    def get_scf_converged(self):
        verbose("getting scf_converged")
        cmd = "grep 'convergence has been achieved in.*iterations' %s" %self.filename
        if com.backtick(cmd).strip() != "":
            return True
        else:
            return False
    

class PwMDOutputFile(PwSCFOutputFile):
    """Parse pw.x MD-like output. Tested so far: 
    md, relax, vc-relax. For vc-md, see PwVCMDOutputFile. 
    """
    def __init__(self, *args, **kwargs):
        PwSCFOutputFile.__init__(self, *args, **kwargs)
        self.attr_lst += [\
            'nstep', 
            'ekin', 
            'temperature',
            ]

    def get_nstep(self):
        verbose("getting nstep")
        self.check_get_attr('coords')
        if self.is_set_attr('coords'):
            return self.coords.shape[self.time_axis]
        else:
            return None
    
    def get_volume(self):
        """For vc-relax, vc-md, pw.x prints stuff like
            ,----------
            | unit-cell volume          =    1725.5120 (a.u.)^3
            | new unit-cell volume =   1921.49226 a.u.^3 (   284.73577 Ang^3 )
            | new unit-cell volume =   1873.15813 a.u.^3 (   277.57339 Ang^3 )
            | new unit-cell volume =   1836.54519 a.u.^3 (   272.14792 Ang^3 )
            | ...
            `----------
        i.e. the vol of the start cell and then all new cells until
        convergence. Here, we only grep for the NEW cell values, which should
        occur nstep times.
        """
        verbose("getting volume")
        cmd = r"grep 'new.*volume' %s | sed -re \
               's/.*volume\s*=\s*(.*?)\s+.*a.u..*/\1/'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_ekin(self):
        verbose("getting ekin")
        cmd = r"grep 'kinetic energy' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))

    def get_temperature(self):
        verbose("getting temperature")
        cmd = r"egrep 'temperature[ ]*=' %s " %self.filename + \
              "| sed -re 's/.*temp.*=\s*(" + regex.float_re + \
              r")\s*K/\1/'"
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_coords(self):
        verbose("getting coords")
        self.check_get_attr('natoms')
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
                             shape=(natoms,3,nstep),
                             axis=self.time_axis)              
    
    def get_cell(self):
        verbose("getting cell")
        # nstep
        key = 'CELL_PARAMETERS'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        # cell            
        cmd = "grep -A3 %s %s | grep -v -e %s -e '--'" %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(3,3,nstep),
                             axis=self.time_axis)
    
    def get_stresstensor(self):
        return self.raw_return('stresstensor')

    def get_etot(self):
        return self.raw_return('etot')
    
    def get_pressure(self):
        return self.raw_return('pressure')
    
    def get_forces(self):
        return self.raw_return('forces')
    
    def get_total_force(self):
        return self.raw_return('total_force')
    
    def get_forces_rms(self):
        return self.raw_return('forces_rms')
    
    def get_nstep_scf(self):
        return self.raw_return('nstep_scf')

    def get_scf_converged(self):
        return None
    

class PwVCMDOutputFile(PwMDOutputFile):
    def __init__(self, *args, **kwargs):
        PwMDOutputFile.__init__(self, *args, **kwargs)
        self.set_attr_lst(self.attr_lst + ['_datadct', 'econst'])

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
        self.check_get_attr('_datadct')
        return self._datadct['ekin']
    
    def get_econst(self):
        verbose("getting econst")
        self.check_get_attr('_datadct')
        return self._datadct['econst']
    
    def get_temperature(self):
        verbose("getting temperature")
        self.check_get_attr('_datadct')
        return self._datadct['temperature']


class AbinitSCFOutputFile(FileParser):
    """Parse Abinit SCF output (ionmov = optcell = 0). 
    
    Each trajectory-like quantity of shape (x,y,nstep) or (nstep, z) in
    Abinit{MD,VCMD}OutputFile has shape (x,y) or (z,) here, i.e. one dimension
    less (no time axis). We treat the SCF case explicitly as "scalar". The
    first value along time_axis is returned.
    
    There are two types of getters defined here:
    
    (1) Usual getters which grep for a single value / array / block. These are
    tested with SCF output. If used for MD-like output, they *may* return the
    corresponding value for the fist time step, but this is not guaranteed and
    depends on how stuff is printed. Do not rely on that.
    
    (2) getters for common quantities to be used in derived classes (MD), see
    _get_*_raw(). They return arrays with a time axis, i.e. at least 2d. Here
    in the SCF case, we simply use the first value along the time axis (SCF is
    "scalar").
    """
    # notes:
    # ------
    # rprim : `rprim` lists the basis vecs as rows (like pwscf's
    #     CELL_PARAMETERS) and contrary to the fuzzy description in the Abinit
    #     docs (maybe they think in Fortran). rprim is `cell`, but each row
    #     divided by acell[i], the length of each cell vector.
    #
    #     i = a,b,c
    #     j = 1,2,3 = x,y,z
    #     
    #     basis vecs: 
    #       a = [a1, a2, a3]
    #       b = [b1, b2, b3]
    #       c = [c1, c2, c3]
    #
    #     A = |a|, B = |b|, C = |c|
    #     acell = [A,B,C]
    #
    #     cell = 
    #       [[a1, a2, a3]
    #        [b1, b2, b3]
    #        [c1, c2, c3]]
    #     
    #     rprim =
    #       [[a1/A, a2/A, a3/A]
    #        [b1/B, b2/B, b3/B]
    #        [c1/C, c2/C, c3/C]]
    # 
    def __init__(self, filename=None):
        FileParser.__init__(self, filename)
        self.set_attr_lst([\
            'angles',
            'cell', 
            'coords_frac', 
            'cryst_const',
            'cryst_const_angles_lengths',
            'etot', 
            'forces',
            'forces_rms',
            'lengths',
            'mass',
            'natoms', 
            'nkpt',
            'nstep_scf',
            'pressure', 
            'stresstensor', 
            'symbols',
            'typat',
            'volume',
            'znucl',
            'scf_converged',
            ])

        # Conceptually not needed for SCF, but some quantities are printed and
        # grepped more than once (e.g. stresstensor).
        self.time_axis = -1
    
    def dump(self, *args, **kwargs):
        if kwargs.has_key('slim'):
            slim = kwargs['slim']
            kwargs.pop('slim')
        else:
            slim = True
        if slim:
            if self.is_set_attr('_angles_raw'):
                del self._angles_raw
            if self.is_set_attr('_coords_frac_raw'):
                del self._coords_frac_raw
            if self.is_set_attr('_forces_raw'):
                del self._forces_raw
            if self.is_set_attr('_lengths_raw'):
                del self._lengths_raw
            if self.is_set_attr('_nstep_scf_raw'):
                del self._nstep_scf_raw
            if self.is_set_attr('_stresstensor_raw'):
                del self._stresstensor_raw
            if self.is_set_attr('_velocity_raw'):
                del self._velocity_raw
            if self.is_set_attr('_volume_raw'):
                del self._volume_raw
        FileParser.dump(self, *args, **kwargs)     

    def _get_stresstensor_raw(self):
        # In abinit output:
        # Cartesian components of stress tensor (hartree/bohr^3)
        #  sigma(1 1)= a  sigma(3 2)= d
        #  sigma(2 2)= b  sigma(3 1)= e
        #  sigma(3 3)= c  sigma(2 1)= f
        # 
        # here:
        # arr[...,i] = 
        #   [[a, d],
        #    [b, e],
        #    [c, f]]
        # 
        # indices: 
        #   abinit = strt = arr
        #   
        #   diagonal            lower
        #   1 1 = 0 0 = 0 0     3 2 = 2 1 = 0 1
        #   2 2 = 1 1 = 1 0     3 1 = 2 0 = 1 1
        #   3 3 = 2 2 = 2 0     2 1 = 1 0 = 2 1   
        verbose("getting _stresstensor_raw")
        assert self.time_axis == -1
        key = 'Cartesian components of stress tensor'
        cmd = "grep '%s' %s | grep -v '^-' | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e '%s' -e '--' -e '^-' \
              | awk '{print $3\" \"$6}'" %(key, self.filename, key)
        arr = traj_from_txt(com.backtick(cmd),
                            shape=(3,2,nstep),
                            axis=self.time_axis)
        if arr is None:
            return None
        else:            
            strt = np.empty((3,3,nstep))
            # diagonal
            strt[0,0,:] = arr[0,0,:]
            strt[1,1,:] = arr[1,0,:]
            strt[2,2,:] = arr[2,0,:]
            # lower
            strt[2,1,:] = arr[0,1,:]
            strt[2,0,:] = arr[1,1,:]
            strt[1,0,:] = arr[2,1,:]
            # upper
            strt[0,1,:] = strt[1,0,:]
            strt[0,2,:] = strt[2,0,:]
            strt[1,2,:] = strt[2,1,:]
            return strt
    
    def _get_nstep_scf_raw(self):
        verbose("getting _nstep_scf_raw")
        cmd = r"grep 'At SCF.*converged' %s | sed -re 's/.*step\s*([0-9]+),.*/\1/'" \
            %self.filename
        return arr1d_from_txt(com.backtick(cmd), dtype=int)
    
    def _get_volume_raw(self):
        """Grep for volume. This should catch all various ways in which the
        volume is printed for several ionmov/optcell combinations.
        For SCF, it catches only the start volume, for MD-like output start
        volume + volume at every step.

        This is different from Pw*OutputFile, where we still have
        get_start_volume().
        """
        verbose("getting _volume_raw")
        cmd = r"grep 'ucvol.*=' %s | \
              sed -re 's/.*ucvol.*=\s*(%s)($|\s*.*)/\1/'" %(self.filename,
              regex.float_re)
        return arr1d_from_txt(com.backtick(cmd))
    
    def _get_angles_raw(self, retall=False):
        verbose("getting _angles_raw")
        results = []
        cmds = [\
            r"egrep -A1 'Angles.*\[degrees\].*' %s | \
            grep -v -e '--' -e 'Angles'" %self.filename,
            # The two below are almost the same ("angles" vs "Angles") but
            # egrep -i is terribly slow.
            "egrep 'angles.*=[ ]+[0-9]+' %s | \
             awk '{print $3\" \"$4\" \"$5}'" %self.filename,
            #
            "egrep 'Angles.*=[ ]+[0-9]+' %s | \
             awk '{print $3\" \"$4\" \"$5}'" %self.filename,
            ]             
        for cmd in cmds:
            results.append(arr2d_from_txt(com.backtick(cmd)))
        # Simple heuristic: Use output from the pattern which found the most
        # results.
        ret = results[np.argmax(axis_lens(results, axis=0))]
        if retall:
            return ret, cmds, results
        else:
            del results
            return ret

    def _get_lengths_raw(self, retall=False):
        verbose("getting _lengths_raw")
        results = []
        cmds = [\
            r"egrep -A1 'Lengths[ ]+\[' %s | \
            grep -v -e '--' -e 'Lengths'" %self.filename,
            #
            "grep 'lengths=' %s | \
            awk '{print $2\" \"$3\" \"$4}'" %self.filename,
            #
            "grep 'length scales=' %s | \
            awk '{print $3\" \"$4\" \"$5}'" %self.filename,
            ]             
        for cmd in cmds:
            results.append(arr2d_from_txt(com.backtick(cmd)))
        ret = results[np.argmax(axis_lens(results, axis=0))]
        if retall:
            return ret, cmds, results
        else:
            del results
            return ret
    
    def get_lengths(self):
        return self.raw_slice_get('lengths', sl=0, axis=0)

    def get_angles(self):
        return self.raw_slice_get('angles', sl=0, axis=0)
    
    def get_stresstensor(self):
        return self.raw_slice_get('stresstensor', sl=0, axis=self.time_axis)
    
    def get_nstep_scf(self):
        return self.raw_slice_get('nstep_scf', sl=0, axis=0)
    
    def get_volume(self):
        return self.raw_slice_get('volume', sl=0, axis=0)

    def get_pressure(self):
        """As in PWscf, pressure = 1/3*trace(stresstensor)."""
        verbose("getting pressure")
        self.check_get_attr('stresstensor')
        if self.is_set_attr('stresstensor'):
            return np.trace(self.stresstensor)/3.0
        else:
            return None
     
    def get_coords_frac(self):
        verbose("getting coords_frac")
        req = ['natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            nn = self.natoms - 1
            cmd = "egrep '^[ ]+xred[ ]+' -A%i %s | grep -v -e '--' | head -n%i \
                  | sed -re 's/xred//'" %(nn, self.filename, nn+1)
            return arr2d_from_txt(com.backtick(cmd))
        else:
            return None
    
    # XXX for MD-like, it greps the *last* printed block (the one in the
    #   summary)!
    # fcart is in Ha/Bohr, we don't parse the eV/Angstrom b/c that can be
    # calculated easily
    def get_forces(self):
        verbose("getting forces")
        req = ['natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            nn = self.natoms - 1
            cmd = "egrep '^[ ]+fcart[ ]+' -A%i %s | grep -v -e '--' | tail -n%i \
                  | sed -re 's/fcart//'" %(nn, self.filename, nn+1)
            return arr2d_from_txt(com.backtick(cmd))
        else:
            return None
    
    # easier to calculate than parse 
    def get_forces_rms(self):
        verbose("getting forces_rms")
        req = ['forces']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.rms(self.forces) 
        else:
            return None
    
    def get_cell(self):
        verbose("getting cell")
        req = ['cryst_const']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.cc2cell(self.cryst_const)
        else:
            return None

    def get_cryst_const(self):
        verbose("getting cryst_const")
        return self.get_cryst_const_angles_lengths()

    def get_cryst_const_angles_lengths(self):
        verbose("getting cryst_const_angles_lengths")
        req = ['angles', 'lengths']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return np.concatenate((self.lengths, self.angles)) 
        else:
            return None
    
    def get_natoms(self):
        verbose("getting natoms")
        cmd = r"egrep '^[ ]*natom' %s | tail -n1 | awk '{print $2}'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_typat(self):
        """Parse
        
        typat 1 1 2 1 
            2 1 2 
        xred ...            
        
        i.e. a multi-line entry where the number of lines is not known. Things
        like "xred", "fcart" etc. are easier b/c we know the number of lines,
        which is natoms.

        notes:
        ------
        With re.search(), we read numbers until the next keyword (xred, ...) is
        found. We make use of the fact that we don't need to operate line-wise
        like grep or sed. The text to be searched is the line "typat ..." +
        `natoms` lines of context as a single string (possibly with newlines),
        which is guaranteed to be enough. I'm pretty sure that there is a
        clever awk line to do this even faster. Ideas, anyone?
        """
        # For only one line:
        # cmd = "grep '^[ ]*typat' %s | head -n1 | sed -re 's/typat//'" %self.filename
        # return arr1d_from_txt(com.backtick(cmd))
        verbose("getting typat")
        req = ['natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            cmd = "egrep -A%i '^[ ]+typat' %s | head -n%i" %(self.natoms,
                self.filename, self.natoms)
            rex = re.compile(r"\s+typat\s+([^a-zA-Z][\s0-9]+)")
            match = rex.search(com.backtick(cmd))
            if match is None:
                return None
            else:
                # match.group(1) == '1 1 2 1\n 2 1 2\n'
                return np.array(match.group(1).strip().split()).astype(int)
        else:
            return None

    def get_znucl(self):
        # Does not work for multiline, but that's very unlikely to occur. We
        # would have to use a *very* large number of different species.
        verbose("getting znucl")
        cmd = "grep '^[ ]*znucl' %s | head -n1 | sed -re 's/znucl//'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_symbols(self):
        verbose("getting symbols")
        req = ['znucl', 'typat']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            idxs = (self.typat - 1).astype(int)
            znucl = self.znucl.astype(int)
            # loop over entire periodic table dict :)
            syms = [sym for zz in znucl for sym,dct in \
                    periodic_table.pt.iteritems() if dct['number']==zz]
            return [syms[i] for i in idxs]
        else:
            return None

    def get_mass(self):
        # mass in amu = 1.660538782e-27 kg
        verbose("getting mass")
        self.check_get_attr('symbols')
        if self.is_set_attr('symbols'):
            return np.array([periodic_table.pt[sym]['mass'] for sym in
                             self.symbols])
        else:
            return None
    
    def get_etot(self):
        verbose("getting etot")
        cmd = r"grep '>>>>.*Etotal\s*=' %s | sed -re 's/.*Etotal\s*=//'" %self.filename
        return float_from_txt(com.backtick(cmd))

    def get_nkpt(self):
        verbose("getting nkpt")
        cmd = "grep '^[ ]*nkpt' %s | head -n1 | sed -re 's/nkpt//'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_scf_converged(self):
        verbose("getting scf_converged")
        cmd = "grep 'At SCF step.*converged' %s" %self.filename
        if com.backtick(cmd).strip() != "":
            return True
        else:
            return False
    
    # alias
    def get_nkpoints(self):
        return self.get_nkpt()


class AbinitMDOutputFile(AbinitSCFOutputFile):
    """Parse MD-like output.

    Tested: 
        ionmov 2 + optcell 0 (optimization: only ions)
        ionmov 2 + optcell 1 (?)
        ionmov 2 + optcell 2 (optimization: ions + cell)
        ionmov 8             (md)
    
    attrs set by parse()
    --------------------
    See AbinitSCFOutputFile, plus:
    nstep
    velocity
    ekin_vel
    temperature
    """
    # For ionmov=2, `cell` is printed as rprimd, which can be used.
    #
    # get_cell(): Do not calculate cell from rprim + acell. Our old assumption
    # that acell is constant and rprim is changed in the output is actually
    # only confirmed for ionmov 13 + optcell 2. Instead, some ionmov/optcell
    # combos change acell and leave rprim constant. Calculate cell only from
    # angles and lengths (a.k.a cryst_const) which seem to be printed in all
    # cases or grep directly from "rprimd".
    def __init__(self, filename=None):
        AbinitSCFOutputFile.__init__(self, filename)
        self.time_axis = -1
        attr_lst = self.attr_lst + \
        ['nstep', 
         'velocity',
         'ekin_vel',
         'temperature', 
        ]
        self.set_attr_lst(attr_lst)
    
    def _get_coords_frac_raw(self):
        verbose("getting _coords_frac_raw")
        self.check_get_attr('natoms')
        natoms = self.natoms
        key = 'Reduced.*xred'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A%i '%s' %s | grep -v -e '%s' -e '--'" \
              %(natoms, key, self.filename, key)
        return traj_from_txt(com.backtick(cmd),
                             shape=(natoms,3,nstep),
                             axis=self.time_axis)
    
    def _get_forces_raw(self):
        verbose("getting _forces_raw")
        self.check_get_attr('natoms')
        natoms = self.natoms
        key = 'Cartesian forces.*fcart'
        cmd = "grep '%s' %s | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A%i '%s' %s | grep -v -e '%s' -e '--'" \
              %(natoms, key, self.filename, key)
        return traj_from_txt(com.backtick(cmd),
                             shape=(natoms,3,nstep),
                             axis=self.time_axis)
    
    def _get_velocity_raw(self):
        verbose("getting _velocity_raw")
        self.check_get_attr('natoms')
        natoms = self.natoms
        key = 'Cartesian velocities (vel)'
        cmd = "grep '%s' %s | wc -l" %(key, self.filename)
        nstep_raw = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A%i '%s' %s | grep -v -e '%s' -e '--'" \
              %(natoms, key, self.filename, key)
        return traj_from_txt(com.backtick(cmd),
                             shape=(natoms,3,nstep_raw),
                             axis=self.time_axis)
        
    # XXX For our test data, this is a little different from
    # crys.rms3d(pp.forces, axis=-1, nitems='all'), but normalization (1st
    # value at step 0) seems correct
    def _get_forces_rms_raw(self):
        verbose("getting _forces_rms_raw")
        key = 'Cartesian forces.*fcart.*rms'
        cmd = "grep '%s' %s | awk '{print $7}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_coords_frac(self):
        return self.raw_return('coords_frac')

    def get_forces(self):
        return self.raw_return('forces')
    
    def get_velocity(self):
        return self.raw_return('velocity')
    
    def get_nstep_scf(self):
        return self.raw_return('nstep_scf')

    def get_angles(self):
        return self.raw_return('angles')
    
    def get_lengths(self):
        return self.raw_return('lengths')
    
    def get_volume(self):
        return self.raw_return('volume')

    def get_forces_rms(self):
        return self.raw_return('forces_rms')
    
    def get_cryst_const_angles_lengths(self):
        verbose("getting cryst_const_angles_lengths")
        req = ['angles', 'lengths']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            n1 = self.angles.shape[0]
            n2 = self.lengths.shape[0]
            if n1 != n2: 
                print("warning: nstep different: angles(%i) and lengths(%i),"
                      " using smaller value starting at end" %(n1,n2))
                nn = min(n1,n2)
            else:
                nn = n1
            return np.concatenate((self.lengths[-nn:,:], self.angles[-nn:,:]), 
                                  axis=1)
        else:
            return None
    
    def get_cryst_const(self):
        verbose("getting cryst_const")
        return self.get_cryst_const_angles_lengths()

    def get_cell(self):
        verbose("getting cell")
        req = ['cryst_const']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            assert self.time_axis == -1
            nstep = self.cryst_const.shape[0]
            cell = np.empty((3,3,nstep))
            for ii in range(nstep):
                cell[...,ii] = crys.cc2cell(self.cryst_const[ii,:])
            return cell                
        else:
            return None
    
    def get_nstep(self):
        verbose("getting nstep")
        self.check_get_attr('coords_frac')
        if self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[self.time_axis]
        else:
            return None
 
    def get_etot(self):
        verbose("getting etot")
        key = 'Total energy (etotal).*='
        cmd = r"grep '%s' %s | sed -re 's/.*=\s*(.*)/\1/'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))

    def get_pressure(self):
        verbose("getting pressure")
        self.check_get_attr('stresstensor')
        if self.is_set_attr('stresstensor'):
            assert self.time_axis == -1
            return np.trace(self.stresstensor,axis1=0, axis2=1)/3.0
        else:
            return None
            
    def get_ekin_vel(self):
        """Kinetic energy in Ha. Sum of Ekin_i from all atoms, obtained from
        velocities. Ekin = sum(i=1...natoms) Ekin_i = 3/2 * natoms * kb*T

        notes:
        ------
        This is for verification only. It's the same as self.ekin but due to
        the way stuff is printed in the outfile, the first velocities are zero,
        but the first ekin value is already ekin_vel[1], so
            ekin_vel[1:] == ekin[:-1]
        """
        verbose("getting ekin_vel")
        req = ['velocity', 'mass']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            # self.velocity [a0*Ha/hbar], self.ekin [Ha], self.ekin_vel [Ha]
            vv = self.velocity
            mm = self.mass
            amu = constants.amu
            a0 = constants.a0
            Ha = constants.Ha
            hbar = constants.hbar
            assert self.time_axis == -1
            return (((vv*a0*Ha/hbar)**2.0).sum(axis=1)*mm[:,None]*amu/2.0).sum(axis=0)/Ha
        else:
            return None
    
    def get_temperature(self):
        """Abinit does not print temperature. Not sure if T fluctuates or is
        constant at each step with MTTK (ionmov 13 + optcell 2).
            Ekin = 3/2 * natoms * kb * T
        This can be seen from abinit/src/95_drive/moldyn.f90 (kb_HaK: K->Ha)
          ! v2gauss is twice the kinetic energy
          v2gauss = ...
          ...
          write(message, '(a,d12.5,a,D12.5)' )&
          ' --- Effective temperature',v2gauss/(3*dtset%natom*kb_HaK),' From variance', sigma2
        """      
        verbose("getting temperature")
        req = ['ekin_vel', 'natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self.ekin_vel * constants.Eh / self.natoms / constants.kb * (2.0/3.0)
        else:
            return None
 

class AbinitVCMDOutputFile(AbinitMDOutputFile):
    """Parse ionmov 13 output (NPT MD with the MTTK method).

    This works for ionmov 13 + optcell 0,1,2. With optcell 0 (fixed cell
    actually), some cell related quantities are None. See
    test/test_abinit_md.py

    notes:
    ------
    Due to simple grepping, some arrays may have a time_axis shape of >
    nstep b/c some quantities are repeated in the summary at the file end
    (for instance self.stresstensor). We do NOT truncate these arrays b/c 
    - we do not need to match quantities to timesteps exactly when plotting
      something over, say, 10000 steps and
    - we often deal with files from killed jobs which do not have the summary

    There is no cell-related information in case optcell 0 (cell = constant
    = start cell). We only parse stuff which gets printed repeatatly at each MD
    step. Start cell information is then self.cell[...,0] etc. We do not parse
    "xred" etc at the beginning. Use AbinitSCFOutputFile for the MD output to
    parse start structure quantities.
    """
    # double printing with ionmov 13 
    # ------------------------------
    # This applies to
    #     Cartesian coordinates (xcart) [bohr]
    #     Reduced coordinates (xred)
    #     Cartesian forces (fcart) [Ha/bohr]; max,rms= ... (free atoms)
    #     Reduced forces (fred)
    #     Cartesian velocities (vel) [bohr*Ha/hbar]; max,rms= ... (free atoms)
    #
    # Except for the 1st coord (=start struct, MOLDYN STEP 0), all are printed
    # twice. Take every second item [*] with numpy indexing array[...,::2].
    # nstep = (nstep_raw - 1)/2 + 1 (+1 for step 0)
    #         moldyn_0 *
    #         moldyn_1
    #         moldyn_1 *
    #         moldyn_2
    #         moldyn_2 *
    #         ...
    #         moldyn_{nstep-1}
    #         moldyn_{nstep-1} *
    #
    # ionmov 2,3,8 (others not tested, see AbinitMDOutputFile)
    # -----------------------------------------------------------
    # * coords, forces etc not double-printed, same regex, but don't use
    #   [...,::2] -> we use _get_*_raw()
    # * printed differently: ekin, etot 
    # 
    def __init__(self, filename=None):
        AbinitMDOutputFile.__init__(self, filename)
        self.time_axis = -1
        attr_lst = self.attr_lst + \
        [\
        'ekin', 
        'etot_ekin',
        ]
        self.set_attr_lst(attr_lst)

    def get_etot_ekin(self):
        verbose("getting etot_ekin")
        key = 'KIN+POT.En.'
        cmd = "grep '%s' %s | awk '{print $2}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))
    
    # Could also use 
    #   raw_slice_get('coords', sl=slice(None, None, 2), axis=self.time_axis)
    def get_coords_frac(self):
        verbose("getting coords_frac")
        self.check_get_attr('_coords_frac_raw')
        ret = self._coords_frac_raw
        return None if ret is None else ret[...,::2]                            
    
    def get_forces(self):
        verbose("getting forces")
        self.check_get_attr('_forces_raw')
        ret = self._forces_raw
        return None if ret is None else ret[...,::2]                            
    
    def get_velocity(self):
        verbose("getting velocity")
        self.check_get_attr('_velocity_raw')
        ret = self._velocity_raw
        return None if ret is None else ret[...,::2]                            
    
    def get_forces_rms(self):
        verbose("getting forces_rms")
        self.check_get_attr('_forces_rms_raw')
        ret = self._forces_rms_raw
        return None if ret is None else ret[...,::2]                            
    
    def get_etot(self):
        verbose("getting etot")
        key = 'end of Moldyn step.*POT.En.'
        cmd = "grep '%s' %s | awk '{print $9}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))

    def get_ekin(self):
        """Must be the same as ekin_vel."""
        verbose("getting ekin")
        req = ['etot_ekin', 'etot']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self.etot_ekin - self.etot
        else:
            return None
    

class CpmdSCFOutputFile(FileParser):
    """Parse output from a CPMD "single point calculation" (wave function
    optimization). Some extra files are assumed to be in the same directory as
    self.filename.
    
    extra files:
        GEOMETRY.scale
    
    notes:
    ------
    * The SYSTEM section must have SCALE such that a file GEOMETRY.scale is
      written.
    * To have forces in the output, use PRINT ON FORCES COORDINATES in the CPMD
      section.

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
    def __init__(self, filename=None):
        """
        args:
        -----
        filename : file to parse
        """        
        FileParser.__init__(self, filename)
        self.time_axis = -1
        self.set_attr_lst([\
        'cell', 
        'coords_frac', 
        'symbols',
        'etot', 
        'forces',
        'mass',
        'natoms', 
        'nkpoints',
        'nstep_scf', 
        'pressure', 
        'stresstensor', 
        'volume',
        'scf_converged',
        ])
        
    def _get_coords_forces(self):
        """ Low precision cartesian Bohr coords + forces (Ha / Bohr) I guess.
        Only printed in this form if we use 
        &CPMD
            PRINT ON COORDINATES FORCES
        &END
        Forces with more precision are printed in the files TRAJECTORY or
        FTRAJECTORY, but only for MD runs.
        """
        verbose("getting _coords_forces")
        self.check_get_attr('natoms')
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
            self.assert_get_attr('natoms')
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
    
    def _get_cell_raw(self):
        """2d array `cell` in Bohr for fixed-cell MD or SCF from GEOMETRY.scale
        file."""
        verbose("getting _cell")
        req = '_scale_file'
        self.check_get_attr(req)
        return self._scale_file['cell'] if self.is_set_attr(req) \
            else None

    def get_stresstensor(self):
        # kbar
        verbose("getting stresstensor")
        cmd = "grep -A3 'TOTAL STRESS TENSOR' %s | tail -n3" %self.filename
        return arr2d_from_txt(com.backtick(cmd))              

    def get_etot(self):
        # Ha
        verbose("getting etot")
        cmd =  r"grep 'TOTAL ENERGY =' %s | tail -n1 | awk '{print $5}'" %self.filename
        return float_from_txt(com.backtick(cmd))
    
    def get_pressure(self):
        # kbar
        verbose("getting pressure")
        self.check_get_attr('stresstensor')
        if self.is_set_attr('stresstensor'):
            return np.trace(self.stresstensor) / 3.0
        else:
            return None
     
    def get_cell(self):
        """Cell in Bohr."""
        return self.raw_return('cell')
    
    def get_coords_frac(self):
        verbose("getting coords_frac")
        req = '_scale_file'
        self.check_get_attr(req)
        return self._scale_file['coords_frac'] if self.is_set_attr(req) \
            else None
 
    def get_mass(self):
        # mass in amu = 1.660538782e-27 kg
        verbose("getting mass")
        self.check_get_attr('symbols')
        if self.is_set_attr('symbols'):
            return np.array([periodic_table.pt[sym]['mass'] for sym in
                             self.symbols])
        else:
            return None
    
    def get_symbols(self):
        verbose("getting symbols")
        req = '_scale_file'
        self.check_get_attr(req)
        return self._scale_file['symbols'] if self.is_set_attr(req) \
            else None
 
    def get_forces(self):
        verbose("getting forces")
        self.check_get_attr('_coords_forces')
        if self.is_set_attr('_coords_forces'):
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

    def get_volume(self):
        verbose("getting volume")
        cmd = r"grep 'INITIAL VOLUME' %s" %self.filename + \
              r"| sed -re 's/.*\):\s+(" + regex.float_re + r")\s*/\1/'" 
        return float_from_txt(com.backtick(cmd))

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


class CpmdMDOutputFile(CpmdSCFOutputFile):
    """CPMD MD output. Works with BO-MD and CP-MD, fixed and variable cell.
    Some attrs may be None or have different shapes (2d va 3d arrays) depending
    on what type of MD is parsed and what info/files are available.
    
    Notes for the commemts below: 
        {A,B,C} = A or B or C
        (A) = A is optional
        (A (B)) = A is optional, but only if present, B is optional

    Extra files which will be parsed and MUST be present:
        GEOMETRY.scale
        GEOMETRY
        TRAJECTORY
        ENERGIES
    
    Extra files which will be parsed and MAY be present depending on the type
    of MD:
        (FTRAJECTORY)
        (CELL) 
        (STRESS)
    
    notes:
    ------
    The input should look like that.
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
    
    Tested with CPMD 3.15.1, the following extra files are always written.
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
    CP b/c there is none).

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
    def __init__(self, filename=None):
        """
        args:
        -----
        filename : file to parse
        """        
        CpmdSCFOutputFile.__init__(self, filename)
        self.time_axis = -1
        self.set_attr_lst([\
        'cell', 
        'coords_frac', 
        'econst',
        'ekinc',
        'etot', 
        'forces',
        'mass',
        'natoms', 
        'nstep',
        'pressure', 
        'stresstensor', 
        'symbols',
        'temperature',
        'velocity',
        'volume',
        ])
        
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
        """Parse (F)TRAJECTORY file. Ignore lines which say
        "<<<<<<  NEW DATA  >>>>>>" from restarts.
        """
        # cols (both files):
        #   0:   natoms x nfi (natoms x 1, natoms x 2, ...)
        #   1-3: x,y,z cartesian coords [Bohr]
        #   4-6: x,y,z cartesian velocites [Bohr / th ] 
        #        th = Hartree time =  0.024189 fs
        # FTRAJECTORY extra:       
        #   7-9: x,y,z cartesian forces [Ha / Bohr]
        self.assert_get_attr('natoms')
        assert self.time_axis == -1
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
            arr = io.readtxt(fn, axis=-1, shape=(self.natoms, ncols, nstep),
                            comments='<<<<')
            dct = {}
            dct['coords'] = arr[:,1:4,:]
            dct['velocity'] = arr[:,4:7,:]
            dct['forces'] = arr[:,7:,:] if have_forces else None
            return dct
        else:           
            return None
    
    def get_cell(self):
        """Parse CELL file. Cell in Bohr. If CELL is not there, return 2d cell
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
            assert self.time_axis == -1
            arr = traj_from_txt(com.backtick(cmd), 
                                shape=(3,ncols,nstep),
                                axis=self.time_axis)
            return arr[:,:3,:]                                
        else:
            return self.raw_return('cell') # 2d

    def get_coords(self):
        """Cartesian coords [Bohr]."""
        req = '_coords_vel_forces'
        self.check_get_attr(req)
        return self._coords_vel_forces['coords'] if self.is_set_attr(req) \
            else None
    
    def get_coords_frac(self):
        req = ['coords', 'cell', 'natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            nstep = self.coords.shape[-1]
            assert self.time_axis == -1
            assert self.coords.shape == (self.natoms,3,nstep)
            axis = self.coords.shape.index(3)
            # cell in Bohr
            if self.cell.ndim == 2:
                assert self.cell.shape == (3,3)
                coords_frac = crys.coord_trans(self.coords, 
                                               old=np.identity(3),
                                               new=self.cell,
                                               axis=axis,
                                               align='rows')
                return coords_frac                                               
            else:
                assert self.cell.shape == (3,3,nstep)
                arr = np.array([crys.coord_trans(self.coords[...,ii],
                                                 old=np.identity(3),
                                                 new=self.cell[...,ii],
                                                 axis=axis,
                                                 align='rows') \
                                for ii in range(nstep)])
                # arr: (nstep, natoms, 3) -> (natoms, 3, nstep)
                return np.rollaxis(arr, 0, 3)
        else:
            return None

    def get_econst(self):
        req = ['_energies_file', 'etot']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            if self._energies_file.has_key('eham'):
                return self._energies_file['eham'] 
            else:
                return self.etot
        else:
            return None

    def get_ekinc(self):
        req = '_energies_file'
        self.check_get_attr(req)
        if self.is_set_attr(req) and self._energies_file.has_key('ekinc'):
            return self._energies_file['ekinc']
        else:
            return None

    def get_etot(self):
        req = '_energies_file'
        self.check_get_attr(req)
        return self._energies_file['eks'] if self.is_set_attr(req) \
            else None

    def get_forces(self):
        """Cartesian forces [Ha/Bohr]."""
        req = '_coords_vel_forces'
        self.check_get_attr(req)
        return self._coords_vel_forces['forces'] if self.is_set_attr(req) \
            else None
    
    def get_nstep(self):
        req = '_energies_file'
        self.check_get_attr(req)
        return int(self._energies_file['nfi'][-1]) if self.is_set_attr(req) \
            else None
    
    def get_pressure(self):
        # kbar
        self.check_get_attr('stresstensor')
        if self.is_set_attr('stresstensor'):
            assert self.time_axis == -1
            assert self.stresstensor.ndim == 3
            return np.trace(self.stresstensor,axis1=0, axis2=1)/3.0
        else:
            return None
     
    def get_stresstensor(self):
        """Stress tensor from STRESS file if available."""
        # kbar
        fn = os.path.join(self.basedir, 'STRESS')
        if os.path.exists(fn):
            cmd = "grep 'TOTAL STRESS' %s | wc -l" %fn
            nstep = int_from_txt(com.backtick(cmd))
            cmd = "grep -A3 'TOTAL STRESS TENSOR' %s | grep -v TOTAL" %fn
            return traj_from_txt(com.backtick(cmd), 
                                 shape=(3,3,nstep),
                                 axis=self.time_axis)              
        else:
            return None
    
    def get_temperature(self):
        req = '_energies_file'
        self.check_get_attr(req)
        return self._energies_file['tempp'] if self.is_set_attr(req) \
            else None
    
    def get_velocity(self):
        """Cartesian velocity [Bohr/th]."""
        req = '_coords_vel_forces'
        self.check_get_attr(req)
        return self._coords_vel_forces['velocity'] if self.is_set_attr(req) \
            else None
   
    # "Disable" some inherited methods. Now, C++ style private methods would
    # come in handy.
    def get_nstep_scf(self):
        return None

    def get_scf_converged(self):
        return None
    
    def _get_coords_forces(self):
        return None


class Grep(object):
    """Maximum felxibility!
    
    If the predefined parsers are not enough, use this (or shell scripting
    with grep/sed/awk).
    
    In the constructor, define your a parsing function, e.g. re.<func> from the
    re module or anything else that parses a text string. You must know what it
    will return, e.g. MatchObject for re.search()/re.match() or list for
    re.findall(). Then, optionally define a handle (function), which takes that
    output and returns something (string, list, whatever) for further
    processing.
    
    examples:
    ---------
    
    scalar values - single file
    ---------------------------
    This example uses only "scalar" values, i.e. values that occur only once in
    the parsed file.
    
    # parse "unit-cell volume          =      87.1541 (a.u.)^3"
    >>> rex=r'unit-cell volume\s+=\s+([+-]*[\\.0-9eEdD+-]+)\s*.*'
    >>> re.search(rex, open('pw.out').read()).group(1)
    '87.1541'
    >>> g=Grep(rex); g.grepfile('pw.out')
    '87.1541'

    Here, the default handle is used for re.search, which is to return the
    first match group m.group(1), where m is a MatchObject.
    
    array values - single file
    --------------------------
    The default handle for re.findall is to return what findall()
    returned, namly a list.

    >>> g=Grep(regex=r'.*Total\s+force\s*=\s*(.*?)\s*Total.*',
    ...        func=re.findall)
    >>> g.grepfile('calc/0/pw.out')
    ['1.619173',
     '1.444923',
     '1.164261',
     '0.799796',
     '0.205767',
     '0.064090',
     '0.002981',
     '0.001006']
    
    scalar values - loop over dirs
    ------------------------------
    (1) define Grep objects

    patterns = { 
        "nelec_out"  : Grep(sqltype = 'FLOAT',
                            regex = r'number.*electr.*=\s*(.*)\s*',
                            basename = 'pw.out'),
        "finished": Grep(sqltype='TEXT', 
                         regex = r'end.*bfgs', re.I,
                         handle = lambda m: m if m is None else 'yes' ,
                         basename='pw.out'),
               } 

    (2) Loop over result dirs calc/0 calc/1 ... calc/20 and grep for stuff in
    pw.{in,out}. In the example below, we assume that `sql` is an sqlite db
    with a table named "calc" and that this table already has a column named
    "idx" and maybe some other columns, too.
    
    idx  foo   
    0    'xx'   
    1    'yy'
    ...  ...
    20   'qq'
    
    Note that `sqltype` is an optional kwarg. It is only used to automagically
    add a column to the db table.
    
    sql = SQLiteDB(...)
    for name, grep in patterns.iteritems():
        sql.add_column(name, grep.sqltype)
        for idx in range(0,20):
            dir = os.path.join('calc', str(idx))
            ret = grep.grepdir(dir)
            sql.execute("UPDATE calc SET %s=%s WHERE idx==%s" \
                        %(name, ret, idx))
    
    notes
    -----
    If you define `func`, you have know that self.func is called
    like
        self.func(regex, open(<filename>).read())
    and you have to construct `func` in that way. Keep that in mind if you need
    to pass extra args to func (like re.M if func is a re module function):
        self.func(regex, open(<filename>).read(), re.M)
     -> func = lambda x,y: re.findall(x,y,re.M)
    """
    def __init__(self, regex, basename=None, 
                 handle = None, 
                 func = re.search,
                 sqltype=None):
        """
        args:
        -----
        regex : str or compiled regex
            If string, it will be compiled.
        basename : {None, str}, optional
            'pw.in' for calc/0/pw.in etc.
        handle : callable, optional
            gets the output of re.<func> and returns a string
            or whatever for further processing
        func : function object, optional
            re module function re.<func>
        sqltype : str, optional
            'TEXT', 'FLOAT', ...
        """
        self.sqltype = sqltype
        self.regex = re.compile(regex) if isinstance(regex, str) else regex
        self.basename = basename
        self.func = func
        self.handle = self._get_handle_for_func(func) if handle is None \
                      else handle
    
    def _get_handle_for_func(self, func):
        if func in [re.search, re.match]:
            return lambda m: m.group(1)
        else:            
            return lambda x: x

    def _handle(self, arg):
        """Wrapper for self.handle that takes care of arg=None (common case if
        a MatchObject is None b/c there was no match) so no self.handle()
        has to deal with that."""
        if arg is None:
            print "No match"
            return None
        else:
            return self.handle(arg)
    
    def _grep(self, fn):
        m = self.func(self.regex, open(fn).read())
        return self._handle(m)
    
    def grepdir(self, dir):
        """
        dir : str
            the dir where to search for a file <dir>/self.basename, e.g.
            dir='calc/0' for 'calc/0/pw.in' and self.basename='pw.in'; this is
            useful for looping over a lot of dirs and grepping for the same thing
            in one file in each dir
        """
        assert self.basename is not None, "basename is None!"
        fn = os.path.join(dir, self.basename)
        return self._grep(fn)

    def grepfile(self, fn):
        """
        fn : str
            full filename, e.g. 'calc/0/pw.in'; use either dir + self.basename
            and self.grepdir() or fn directly (this method)
        """
        return self._grep(fn)
    
    def grep(self, dir):
        # backwards compat only
        return self.grepdir(dir)

# backward compat
AbinitMDOptOutputFile = AbinitMDOutputFile
PwOutputFile = PwMDOutputFile
