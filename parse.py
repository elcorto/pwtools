# parse.py
#
# Parser classes for different file formats. Input- and output files.
#  
# We need the following basic Unix tools installed:
#   grep/egrep
#   sed
#   awk
#   tail
#   wc 
#   ...
# 
# Notes:
# * Some functions/methods are Python-only (mostly for historical reasons ..
#   code was written once and still works), but most of them actually call
#   grep/sed/awk. This may not be pythonic, but hey ... these tools rock and
#   the cmd lines are short.
# * pwtools.com.backtick() takes "long" to create a child process. So for small
#   files, pure-python versions, although they have much more code, are faster. 
#   But who cares if the files are small. For big files, grep&friends win + much
#   less code here.
# * The tested egrep's don't know the "\s" character class for whitespace
#   as sed, Perl, Python or any other sane regex implementation does. Use 
#   "[ ]" instead.

import re
import os
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

from pwtools import io, common, constants, regex, crys, periodic_table
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
        CELL_PARAMETERS
           4.477898982  -0.031121661   0.004594438
          -2.231322621   3.882493243  -0.004687159
           0.012261087  -0.006915066  12.050227607
        ATOMIC_SPECIES
        Si 28.0855 Si.LDA.fhi.UPF
        O 15.9994 O.LDA.fhi.UPF   
        Al 26.981538 Al.LDA.fhi.UPF
        N 14.0067 N.LDA.fhi.UPF
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


def scan_until_pat2(fh, rex, err=True, retmatch=False):
    """
    *Slightly* faster than scan_until_pat().

    args:
    ----
    fh : file like object
    rex : Regular Expression Object with compiled pattern for re.match().
    err : bool, raise error at end of file b/c pattern was not found
    retmatch : bool, return current Match Object
    
    returns:
    --------
    fh : file like object
    flag : int
    {match : re Match Object}

    notes:
    ------
    For this function to be fast, you must pass in a compiled re object. 
        rex = re.compile(...)
        fh, flag = scan_until_pat2(fh, rex)
    or 
        fh, flag = scan_until_pat2(fh, re.compile(...))
    BUT, if you call this function often (e.g. in a loop) with the same
    pattern, use the first form, b/c otherwise re.compile(...) is evaluated in
    each loop pass. Then, this would essentially be the same as using
    re.match() directly.
    """
    for line in fh:
        match = rex.match(line)
        if match is not None:
            if retmatch:
                return fh, 1, match
            return fh, 1
    if err:
        raise StandardError("end of file '%s', pattern "
            "not found" %fh)
    # nothing found = end of file, rex.match(line) should be == None
    if retmatch:
        return fh, 0, match
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
        return io.readtxt(StringIO(txt), axis=axis, shape=shape).astype(dtype)

def arr1d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        return np.atleast_1d(np.loadtxt(StringIO(txt))).astype(dtype)

def arr2d_from_txt(txt, dtype=np.float):
    if txt.strip() == '':
        return None
    else:
        return np.atleast_2d(np.loadtxt(StringIO(txt))).astype(dtype)

#-----------------------------------------------------------------------------
# Parsers
#-----------------------------------------------------------------------------

class FlexibleGetters(object):
    """Base class. Implements a mechanism which allows to call getters in
    arbitrary order, even if they depend on each other.
    
    For each attr, there must exist a getter. We define the convention 
      self.foo  -> self.get_foo() 
      self.bar  -> self.get_bar()  
      self._baz -> self._get_baz() # note the underscores
      ... 
    
    self.attr_list is an *optional* list of strings, each is the name of a data
    attribute, e.g. ['foo', 'bar', '_baz', ...].       
    Derived classes can override self.attr_list by using self.set_attr_lst().
    
    Example:
        def __init__(self):
            self.set_attr_lst(['foo', 'bar', '_baz'])
            self.get_all()
        
        def get_all(self):
            for attr in self.attr_lst:
                self.check_get_attr(attr)
        
        # Getters call each other
        def _get_baz(self):
            return self.calc_baz()
        
        def get_bar(self):
            check_get_attr('_baz')
            return self.calc_stuff(self._baz)**2.0

        def get_foo(self):
            self.check_get_attr('bar')
            self.check_get_attr('_baz')
            return do_stuff(self._baz, self.bar)
    
    Setting self.attr_list is optional. It is supposed to be used only in
    get_all(). The check_get_attr() - method works without it, too. 
    """ 
    # Notes for derived classes (long explaination):
    #
    # In this class we define a number of members (self.member1, self.member2,
    # ...) which shall all be set by the get_all() method.
    #
    # There are 3 ways of doing it:
    #
    # 1) Put all code in get_all(). 
    #    Con: One might forget to implement the setting of a member.
    # 
    # 2) Implement get_all() so that for each data member of the API, we have
    #       self.member1 = self.get_member1()
    #       self.member2 = self.get_member2()
    #       ...
    #    and put the code for each member in a separate getter. This is good
    #    coding style, but often data needs to be shared between getters (e.g.
    #    get_member1() needs member2, which is the result of self.member2 =
    #    self.get_member2(). This means that in general the calling order
    #    of the getters is important and is different in each get_all() of each
    #    derived class.
    #    Con: One might forget to call a getter in get_all() and/or in the wrong 
    #         order.
    # 
    # 3) Implement all getters such that they can be called in arbitrary order.
    #    Then in each get_all(), one does exactly the same:
    #
    #        attr_lst = ['member1', 'member2', ...]
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
    #       def get_member1(self):
    #           return do_stuff(self.member2)
    #    to 
    #       
    #       def get_member1(self):
    #           self.check_get_attr('member2')                <<<<<<<<<<<<
    #           return do_stuff(self.member2)
    #
    #    If one does
    #        self.member1 = self.get_member1()
    #        self.member2 = self.get_member2()
    #        ....
    #    then some calls may in fact be redundant b/c e.g. get_member1() has
    #    already been called inside get_member2(). There is NO big overhead in
    #    this approach b/c in each getter we test with check_get_attr() if a
    #    needed other member is already set.
    #    
    #    This way we get a flexible and easily extensible framework to
    #    implement new parsers and modify existing ones (just implement another
    #    getter get_newmember() in each class and extend the list of API
    #    members by 'newmember').
    #
    #    One drawback: Beware of cyclic dependencies (i.e. get_member2 ->
    #    get_member1 -> get_member2 -> ...). Always test the implementation!
    
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
    
    def assert_attrs(self, attr_lst):
        for attr in attr_lst:
            self.assert_attr(attr)
    
    def check_get_attrs(self, attr_lst):
        for attr in attr_lst:
            self.check_get_attr(attr)
    

class FileParser(FlexibleGetters):
    """Base class for file parsers.
    
    All getters are called in the default self.parse() which can, of course, be
    overridden in derived classes.
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
        else:
            self.fd = None
    
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

# Idea:
#
# Structure : single struct -> Atoms in ASE, ATM all classes derived from
#   StructureFileParser have this character (CifFile, etc)
#   
# Trajectory : for trajectories, some arrays are 3d
#   This is like Pw*OutputFile.
# 
# Both as input for crys.write_*, crys.rpdf() etc
#
# Most usefull feature: Must hold attr self.coords_frac, derived classes must
# define getter (how to get fractional coords), i.e. PwOutputFile must know all
# about pwscf's ibrav etc. This means we must implement all 14 bravais lattices
# or map pwscf's ibrav to ASE.lattice
#
# BUT: How to best combine parser class and structure class?


class StructureFileParser(FileParser):
    """Base class for structure file (pdb, cif, etc) and input file parsers.
    A file parsed by this class is supposed to contain infos about an atomic
    structure.
    
    Classes derived from this one must provide the following members. If a
    particular information is not present in the parsed file, the corresponding
    member must be None.
    
    parsing results:
    ----------------
    self.coords : ndarray (natoms, 3) with atom coords
    self.symbols : list (natoms,) with strings of atom symbols, must match the
        order of the rows of self.coords
    self.cell : 3x3 array with primitive basis vectors as rows, for
        PWscf, the array is in units of (= divided by) alat == self.cryst_const[0]
    self.cryst_const : array (6,) with crystallographic costants
        [a,b,c,alpha,beta,gamma]
    self.natoms : number of atoms

    convenience attributes:
    -----------------------
    self.atpos_str : a string representing the ATOMIC_POSITIONS card in a pw.x
        in/out file

    Unless explicitly stated, we DO NOT DO any unit conversions with the data
    parsed out of the files. It is up to the user (and derived classes) to
    handle that. 
    """
    def __init__(self, filename=None):
        FileParser.__init__(self, filename)
        self.set_attr_lst(['coords', 'symbols', 'cryst_const', 'cell',
                           'natoms', 'atpos_str'])

    # Convenience getters
    def get_atpos_str(self):
        self.check_get_attr('coords')
        self.check_get_attr('symbols')
        return atpos_str(self.symbols, self.coords)
    

class CifFile(StructureFileParser):
    """Extract cell parameters and atomic positions from Cif files. This
    data can be directly included in a pw.x input file. 

    members:
    --------
    See StructureFileParser

    extra members:
    --------------
    celldm : array (6,), PWscf celldm, see [2]
        [a, b/a, c/a, cos(alpha), cos(beta), cos(gamma)]
        NOTE: 'a' IS ALWAYS IN BOHR, NOT ANGSTROM!!
    
    notes:
    ------
    cif parsing:
        We expect PyCifRW [1] to be installed, which provides the CifFile
        module.
    cell dimensions:
        We extract
        _cell_length_a
        _cell_length_b
        _cell_length_c
        _cell_angle_alpha
        _cell_angle_beta
        _cell_angle_gamma
        and transform them to pwscf-style celldm. 
    atom positions:
        Cif files contain "fractional" coords, which is just 
        "ATOMIC_POSITIONS crystal" in PWscf.
    
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
        self.set_attr_lst(self.attr_lst + ['celldm'])
    
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
    
    def get_celldm(self):
        self.check_get_attr('_cif_dct')
        celldm = []
        # ibrav 14, celldm(1) ... celldm(6)
        celldm.append(self._cif_dct['a']/constants.a0_to_A) # Angstrom -> Bohr
        celldm.append(self._cif_dct['b']/self._cif_dct['a'])
        celldm.append(self._cif_dct['c']/self._cif_dct['a'])
        celldm.append(cos(self._cif_dct['alpha']*pi/180))
        celldm.append(cos(self._cif_dct['beta']*pi/180))
        celldm.append(cos(self._cif_dct['gamma']*pi/180))
        return np.asarray(celldm)


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
    massvec : 1d array (natoms,)
        Array of masses of all atoms in the order listed in
        ATOMIC_POSITIONS. This is actually self.atpos['massvec'].
    
    notes:
    ------
    self.cell (CELL_PARAMETERS) in pw.in is units of alat
    (=celldm(1)). If we have an entry in pw.in to determine alat:
    system:celldm(1) or sysetm:A, then the cell parameters will be multiplied
    with that *only* for the calculation of self.cryst_const. Then [a,b,c] =
    cryst_const[:3] will have the right unit (Bohr). A warning will be issued
    if neither is found. self.cell will be returned as it is in the
    file.
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
                                           'massvec'])
    
    def get_massvec(self):
        self.check_get_attr('atpos')
        return self.atpos['massvec']

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
                         'A': 1/constants.a0_to_A}
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
        {'coords': coords, 'natoms': natoms, 'massvec': massvec, 'symbols':
        symbols, 'unit': unit}
        
        coords : ndarray,  (natoms, 3)
        natoms : int
        massvec : 1d array, (natoms,)
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
        massvec = np.array([masses[atspec_symbols.index(s)] for s in symbols], dtype=float)
        return {'coords': coords, 'natoms': natoms, 'massvec': massvec, 'symbols':
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


class PwOutputFile(FileParser):
    """Parse a pw.x output file. This class is primarily geared towards
    pw.x MD runs but also works for other calculation types. Tested so far: 
    md, scf, relax, vc-relax. For vc-md, see PwVCMDOutputFile. 

    For scf-like calculations, where we have in fact only 1 "time step", all
    trajectory-like 3d arrays have nstep=1, i.e. coords: (natoms, 3, 1).
    
    members:
    --------
    etot : 1d array (nstep,)
    ekin : 1d array (nstep,)
    stresstensor : 3d array (3, 3, nstep) 
        stress tensor for each step 
    pressure : 1d array (nstep,) 
        This is parsed from the "P= ... kbar" lines and the value is actually
        1/3*trace(stresstensor)
    temperature : 1d array (nstep,)
    coords : 3d array (natoms, 3, nstep)
    start_coords : 2d array
        atomic coords of the start unit cell in cartesian alat units
    cell : 3d array (3, 3, nstep) 
        prim. basis vectors for each step
    start_cell : 2d array
        start prim. basis vectors from the output file in alat units, parsed
        from "crystal axes: (cart. coord. in units of a_0)"
    nstep : scalar 
        number of MD steps
    natoms : scalar 
        number of atoms
    volume : 1d array (nstep,)
    start_volume : float
        the volume of the start unit cell
    total_force : 1d array (nstep,)
        The "total force" parsed from lines "Total force =" after the "Forces
        acting on atoms" block.
    forces : 3d array (natoms, 3, nstep)
    forces_rms : 1d array (nstep,) of RMS of the forces, each forces[...,i] is
        normalized to 3*natoms
    nkpoints : number of kpoints        
    time_axis : the time axis along which all 3d arrays have 2d arrays lined
        up; e.g. `coords` has 2d arrays with atomic coords[:,:,i] for
        i=0,...,nstep-1; time_axis is currently hardcoded to -1, i.e. the last
        axis

    Members, whose corresponding data in the file is not present or cannot
    parsed b/c regexes don't match, are None.
    E.g. if there are no CELL_PARAMETERS printed in the output file, then
    self.cell == None.

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
    # 
    # Possible optimization: Use get_txt() to read in the file to parse and use
    #     only python's re module. Would spare us spawning child processes for
    #     grep/sed/awk. But grep is probably faster for large files and has
    #     some very nice features, too.
    def __init__(self, filename=None):
        """
        args:
        -----
        filename : file to parse
        """        
        FileParser.__init__(self, filename)
        self.time_axis = -1
        self.set_attr_lst([\
        'nstep', 
        'nstep_scf', 
        'etot', 
        'ekin', 
        'stresstensor', 
        'pressure', 
        'temperature', 
        'coords', 
        'cell', 
        'natoms', 
        'volume',
        'total_force',
        'forces',
        'forces_rms',
        'start_volume',
        'start_cell',
        'start_coords',
        'nkpoints',
        ])
        
    def get_nstep(self):
        verbose("getting nstep")
        self.check_get_attr('coords')
        if self.coords is not None:
            return self.coords.shape[self.time_axis]
        else:
            return None
    
    def get_natoms(self):
        verbose("getting natoms")
        cmd = r"grep 'number.*atoms/cell' %s | \
              sed -re 's/.*=\s+([0-9]+).*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_nkpoints(self):
        verbose("getting nkpoints")
        cmd = r"grep 'number of k points=' %s | \
            sed -re 's/.*points=\s*([0-9]+)\s*.*/\1/'" %self.filename
        return int_from_txt(com.backtick(cmd))

    def get_stresstensor(self):
        verbose("getting stress tensor")
        key = 'P='
        cmd = "grep %s %s | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e %s -e '--'| \
              awk '{printf $4\"  \"$5\"  \"$6\"\\n\"}'" \
              %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(3,3,nstep),
                             axis=self.time_axis)              

    def get_etot(self):
        verbose("getting etot")
        cmd =  r"grep '^!' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_ekin(self):
        verbose("getting ekin")
        cmd = r"grep 'kinetic energy' %s | awk '{print $5}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))

    def get_pressure(self):
        verbose("getting pressure")
        cmd = r"grep P= %s | awk '{print $6}'" %self.filename
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
    
    def get_start_coords(self):
        """Grep start ATOMIC_POSITIONS from pw.out. This is always in cartesian
        alat units. The start coords in the format given in the input file is
        just infile.coords. This should be the same as
        >>> p = PwOutputFile(...)
        >>> p.parse()
        >>> crys.coord_trans(p.start_coords, 
        >>>                  old=np.identity(3), 
        >>>                  new=p.start_cell, 
        >>>                  align='rows')
        """
        verbose("getting start coords")
        self.check_get_attr('natoms')
        natoms = self.natoms
        cmd = r"grep -A%i 'positions.*a_0.*units' %s | tail -n%i | \
              sed -re 's/.*\((.*)\)/\1/g'" \
              %(natoms, self.filename, natoms)
        return arr2d_from_txt(com.backtick(cmd))

    def get_forces(self):
        verbose("getting forces")
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
    
    def get_forces_rms(self):
        verbose("getting forces_rms")
        req = ['forces']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.rms3d(self.forces, axis=self.time_axis) 
        else:
            return None

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
    
    def get_start_cell(self):
        """Grep start cell from pw.out. This is always in alat
        units."""
        verbose("getting start cell parameters")
        cmd = "grep -A3 'crystal.*axes.*units.*a_0' %s | tail -n3 | \
               awk '{print $4\" \"$5\" \"$6}'" %(self.filename)
        return arr2d_from_txt(com.backtick(cmd))
    
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
    
    def get_start_volume(self):
        verbose("getting start volume")
        ret_str = com.backtick(r"grep 'unit-cell.*volume' %s | head -n1 | sed -re \
                                's/.*volume\s*=\s*(.*?)\s+.*a.u..*/\1/'"
                                %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return float(ret_str)

    def get_total_force(self):
        verbose("getting total force")
        cmd = r"egrep 'Total[ ]+force[ ]*=.*Total' %s \
            | sed -re 's/^.*Total\s+force\s*=\s*(.*)\s*Total.*/\1/'" \
            %self.filename
        return arr2d_from_txt(com.backtick(cmd))

    def get_nstep_scf(self):
        verbose("getting nstep_scf")
        cmd = r"grep 'convergence has been achieved in' %s | awk '{print $6}'" \
            %self.filename
        return arr1d_from_txt(com.backtick(cmd), dtype=int)

    
class PwVCMDOutputFile(PwOutputFile):
    def __init__(self, *args, **kwargs):
        PwOutputFile.__init__(self, *args, **kwargs)
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
    

class CPOutputFile(PwOutputFile):
    """
    Some notes on parsing cp.x output data.

    For the following explanations:
      infile = PwInputFile('cp.in'); infile.parse()  
      iprint = infile.namelists['control']['iprint']
      isave  = infile.namelists['control']['isave']

    The cp.x code writes several text files to scratch:

    Fortran unit | filename  | content (found out by myself)
    30  <prefix>.con  ?
    31  <prefix>.eig  ?
    32  <prefix>.pol  ?
    33  <prefix>.evp  temperature etc. at iprint steps
    34  <prefix>.vel  atomic velocities
    35  <prefix>.pos  atomic positions (self.coords)
    36  <prefix>.cel  cell parameters (self.cell)
    37  <prefix>.for  forces on atoms 
    38  <prefix>.str  stress tensors  (self.stresstensor)
    39  <prefix>.nos  Nose thetmostat stuff every iprint steps
    40  <prefix>.the  ?
    41  <prefix>.spr  ?
    42  <prefix>.wfc  ? (wafefunctions)?

    Of course, there is no documentation as to what they contain!

    We don't read most of the stuff (coords etc) from these files but instead
    directly from the outfile since we already have the code (inherit from
    PwOutputFile) and it's fast. The text files in scratch have the crappy
    Fortran-style format of e.g.
      [scalar, i.e. time step]
      [matrix, e.g. cell parameters]
      [scalar]
      [matrix]
      ...
    and we don't like to write code that reads this.  

    BUT we need the .evp file b/c:
    The .evp file contains temperature, electron kinetic energy
    etc at every iprint step, .nos some cryptic Nose thermostat variables
    at every iprint step.

    cp.x also prints a line with 12 numbers to stdout (outfile)
    every iprint steps and the header 

      nfi ekinc temph tempp etot enthal econs econt vnhh xnhh0 vnhp xnhp0

    every isave steps. This lists some (but not all!) values from the 
    .evp file and some from the .nos file!

    The important thing is that we cannot grep these lines from the
    outfile b/c this is ambiguous. There may be (in fact there are!)
    other lines with 12 numbers :) So .. we have to load the crappy
    file.
    """
    def __init__(self, filename=None, evpfilename=None):
        
        # XXX This class has not been tested very much yet. Parsing a test
        # cp.out works, but we have not verified if everything is parsed
        # correctly. E.g. we are not sure if self.evp_order is correct. Would
        # have to look at the code b/c this is not documented.
        raise NotImplementedError("If you really want to do CP, use CPMD.")


        PwOutputFile.__init__(self, filename)
        self.evpfilename = evpfilename
        # columns of self.evpfilename
        self.evp_order = \
            ['nfi', 'ekinc', 'temphc', 'tempp', 'etot', 'enthal', 'econs', 
             'econt', 'volume', 'out_press', 'tps']
    
    def parse(self):
        com.assert_cond(self.evpfilename is not None, "self.evpfilename is None")
        self.evp_data = self.load_evp_file()
        PwOutputFile.parse(self)
    
    def load_evp_file(self):
        """Load the file /path/to/scratch/<prefix>.evp. It contains temperature
        etc. at every iprint step. See self.evp_order .        
        """ 
        verbose("loading evp file: %s" %self.evpfilename)
        return np.loadtxt(self.evpfilename)

    def get_stresstensor(self):
        verbose("getting stress tensor")
        key = "Total stress"
        cmd = "grep '%s' %s | wc -l" %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e '%s' -e '--'" %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd), 
                             shape=(3,3,nstep),
                             axis=self.time_axis)
    
    def get_etot(self):
        verbose("getting etot")
        return self.evp_data[:, self.evp_order.index('etot')]
    
    def get_ekin(self):
        verbose("getting ekin")
        cmd = r"egrep 'kinetic[ ]+energy[ ]=' %s | awk '{print $4}'" \
              %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_pressure(self):
        verbose("getting pressure")
        return self.evp_data[:, self.evp_order.index('out_press')]
    
    def get_temperature(self):
        verbose("getting temperature")
        return self.evp_data[:, self.evp_order.index('tempp')]


class AbinitSCFOutputFile(FileParser):
    """ Parse Abinit SCF output (ionmov = optcell = 0).

    PwOutputFile works for SCF and MD-like calculations. In Abinit, too many
    quantities are printed differently in the SCF output. Each trajectory-like
    quantity of shape (X,Y,nstep) in Abinit{MDOpt,VCMD}OutputFile has shape
    (X,Y) here, i.e. only 2d array.
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
            'acell',
            'rprim',
            'cryst_const',
            'typat',
            'znucl',
            'symbols',
            'etot', 
            'stresstensor', 
            'pressure', 
            'mass',
            'coords_frac', 
            'cell', 
            'natoms', 
            'volume',
            'forces_rms',
            'forces',
            'nkpt',
            'nstep_scf',
            ])
    
        # Conceptually not needed for SCF, but some quantities are printed and
        # grepped more than once (stresstensor).
        self.time_axis = -1

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

    def get_stresstensor(self):
        """Return the last printed stresstensor."""
        req = ['_stresstensor_raw']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self._stresstensor_raw[...,-1]
        else:
            return None
    
    def get_nstep_scf(self):
        verbose("getting nstep_scf")
        req = ['_nstep_scf_raw']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self._nstep_scf_raw[-1]
        else:
            return None
    
    def get_pressure(self):
        """As in PWscf, pressure = 1/3*trace(stresstensor)."""
        verbose("getting pressure")
        self.check_get_attr('stresstensor')
        if self.is_set_attr('stresstensor'):
            return np.trace(self.stresstensor)/3.0
        else:
            return None
     
    def get_rprim(self):
        verbose("getting rprim")
        cmd = "egrep '^[ ]+rprim[ ]+' -A2 %s | grep -v -e '--' | head -n3 \
                | sed -re 's/rprim//'" %self.filename
        return arr2d_from_txt(com.backtick(cmd))
    
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
        req = ['rprim', 'acell']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self.rprim*self.acell[:,None]       
        else:
            return None
    
    def get_cryst_const(self):
        verbose("getting cryst_const")
        req = ['cell']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.cell2cc(self.cell)      
        else:
            return None
    
    def get_volume(self):
        verbose("getting volume")
        req = ['cell']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return crys.volume_cell(self.cell)      
        else:
            return None

    def get_acell(self):
        verbose("getting acell")
        cmd = "egrep '^[ ]+acell' %s | head -n1 | awk '{print $2\" \"$3\" \"$4}'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
##    def get_shiftk(self):
##        verbose("getting shiftk")
##        cmd = "egrep '^[ ]+shiftk' %s | head -n1 | awk '{print $2\" \"$3\" \"$4}'" %self.filename
##        return arr1d_from_txt(com.backtick(cmd))
    
    def get_natoms(self):
        verbose("getting natoms")
        cmd = r"egrep '^[ ]*natom' %s | tail -n1 | awk '{print $2}'" %self.filename
        return int_from_txt(com.backtick(cmd))
    
    def get_typat(self):
        """Parse
        
        typat 1 1 2 1 
            2 1 2 
        
        i.e. a multi-line entry where the number of lines is not known. Things
        like "xred", "fcart" etc. are easier b/c we know the number of lines,
        which is natoms.

        notes:
        ------
        We read numbers until the next keyword (typat, xred, ...) is found. We
        make use of the fact that we don't need to operate line-wise like grep.
        The whole file is a single string (maybe slow for large files). I'm
        pretty sure that there is a clever sed line to do this even faster.
        Ideas, anyone?
        """
        # For only one line:
        # cmd = "grep '^[ ]*typat' %s | head -n1 | sed -re 's/typat//'" %self.filename
        # return arr1d_from_txt(com.backtick(cmd))
        verbose("getting typat")
        req = ['txt']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            rex = re.compile(r"\s+typat\s+([^a-zA-Z][\s0-9]+)")
            match = rex.search(self.txt)
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
    
    # alias
    def get_nkpoints(self):
        return self.get_nkpt()


class AbinitMDOptOutputFile(AbinitSCFOutputFile):
    """Parse MD-like optimization output.

    Tested: 
        ionmov 2 + optcell 0 (only ions)
        ionmov 2 + optcell 2 (ions + cell)
    """
    # `cell` is printed as rprimd, so rprim is not needed here. But it is
    # calculated from acell and cell, b/c get_rprim() is otherwise the one from
    # AbinitSCFOutputFile, which greps something different.
    #
    # In AbinitVCMDOutputFile, we calculate cell from acell and rprim. 
    
    def __init__(self, filename=None):
        AbinitSCFOutputFile.__init__(self, filename)
        self.time_axis = -1
        attr_lst = self.attr_lst + \
        ['angles',
         'lengths',
         'nstep', 
        ]
        self.set_attr_lst(attr_lst)
    
    def get_nstep_scf(self):
        verbose("getting nstep_scf")
        req = ['_nstep_scf_raw']
        self.check_get_attrs(req)
        ret = self._nstep_scf_raw
        return None if ret is None else ret

    def get_angles(self):
        verbose("getting angles")
        cmd = r"egrep -A1 'Angles.*=[ ]+.*\[degrees\].*' %s | \
                grep -v -e '--' -e 'Angles'" %self.filename
        return arr2d_from_txt(com.backtick(cmd))
    
    def get_lengths(self):
        verbose("getting lengths")
        cmd = r"egrep -A1 'Lengths[ ]+\[' %s | \
              grep -v -e '--' -e 'Lengths'" %self.filename
        return arr2d_from_txt(com.backtick(cmd))
    
    def get_cryst_const(self):
        verbose("getting cryst_const")
        req = ['angles', 'lengths']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            n1 = self.angles.shape[-1]
            n2 = self.lengths.shape[-1]
            assert n1 == n2, "nstep different: angles(%i) and lengths(%i)" %(n1,n2)
            return np.concatenate((self.lengths, self.angles), axis=1)
        else:
            return None
    
    def get_cell(self):
        verbose("getting cell")
        key = 'rprimd'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A3 '%s' %s | grep -v -e '%s' -e '--'" \
              %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd),
                             shape=(3,3,nstep),
                             axis=self.time_axis)
    
    def get_rprim(self):
        verbose("getting rprim")
        req = ['cell', 'acell']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            assert self.time_axis == -1
            return self.cell/self.acell[:,None,None]       
        else:
            return None

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
    
    def get_coords_frac(self):
        self.check_get_attr('_coords_frac_raw')
        ret = self._coords_frac_raw 
        return None if ret is None else ret

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
    
    def get_forces(self):
        self.check_get_attr('_forces_raw')
        ret = self._forces_raw
        return None if ret is None else ret
    
    # XXX For our test data, this is a little different from
    # crys.rms3d(pp.forces, axis=-1, nitems='all'), but normalization (1st
    # value at step 0) seems correct
    def get_forces_rms(self):
        verbose("getting forces_rms")
        key = 'Cartesian forces.*fcart.*rms'
        cmd = "grep '%s' %s | awk '{print $7}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_nstep(self):
        verbose("getting nstep")
        self.check_get_attr('coords_frac')
        if self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[self.time_axis]
        else:
            return None
 
    def get_volume(self):
        verbose("getting volume")
        cmd = r"grep 'Unitary Cell Volume.*=' %s | \
              sed -re 's/.*ucvol.*=\s*(.*)/\1/'" %self.filename
        return arr1d_from_txt(com.backtick(cmd))
    
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
            
    def get_stresstensor(self):
        req = ['_stresstensor_raw']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self._stresstensor_raw
        else:
            return None
                                 

class AbinitVCMDOutputFile(AbinitMDOptOutputFile):
    """Parse ionmov 13 output (NPT MD with the MTTK method).

    This works for ionmov 13 + optcell 0,1,2. With optcell 0 (fixed cell
    actually), some cell related quantities are None. See
    test/test_abinit_ionmom*.py

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
    # double printing with ionmov 13: This applies to
    #         * coord_frac
    #         * forces
    #     Except for the 1st coord (=start struct, MOLDYN STEP 0), all are
    #     printed twice. Take every second item [*] with numpy indexing
    #     array[...,::2]. nstep = (nstep_raw - 1)/2 + 1 (+1 for step 0)
    #         moldyn_0 *
    #         moldyn_1
    #         moldyn_1 *
    #         moldyn_2
    #         moldyn_2 *
    #         ...
    #         moldyn_{nstep-1}
    #         moldyn_{nstep-1} *
    #
    # ionmov 2,3,8 (others not tested, see AbinitMDOptOutputFile):
    #     * coords, forces etc not double-printed, same regex, but don't use
    #       [...,::2] -> we use _get_{forces,coord_frac}_raw()
    #     * printed differently: ekin, etot 
    # 
    def __init__(self, filename=None):
        AbinitSCFOutputFile.__init__(self, filename)
        self.time_axis = -1
        attr_lst = self.attr_lst + \
        [\
        'etot_ekin',
        'rprim',
        'ekin', 
        'ekin_vel',
        'temperature', 
        'velocity',
        ]
        self.set_attr_lst(attr_lst)
    
    def get_angles(self):
        verbose("getting angles")
        cmd = "grep 'angles.*degrees' %s | awk '{print $3\" \"$4\" \"$5}'" %self.filename
        return arr2d_from_txt(com.backtick(cmd))
    
    def get_lengths(self):
        verbose("getting lengths")
        cmd = "grep 'lengths=' %s | awk '{print $2\" \"$3\" \"$4}'" %self.filename
        return arr2d_from_txt(com.backtick(cmd))
    
    def get_rprim(self):
        verbose("getting rprim")
        key = 'rprim='
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        nstep = nstep_from_txt(com.backtick(cmd))
        cmd = "grep %s -A2 %s | grep -v -e '--' | sed -re 's/%s//'" \
            %(key, self.filename, key)
        return traj_from_txt(com.backtick(cmd),
                             shape=(3,3,nstep),
                             axis=self.time_axis)
    
    def get_etot_ekin(self):
        verbose("getting etot_ekin")
        key = 'KIN+POT.En.'
        cmd = "grep '%s' %s | awk '{print $2}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_cell(self):
        verbose("getting cell")
        req = ['rprim', 'acell']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            assert self.time_axis == -1
            return self.rprim*self.acell[:,None,None]       
        else:
            return None
    
    def get_velocity(self):
        verbose("getting velocity")
        self.check_get_attr('natoms')
        natoms = self.natoms
        key = 'Cartesian velocities (vel)'
        cmd = "grep '%s' %s | wc -l" %(key, self.filename)
        nstep_raw = nstep_from_txt(com.backtick(cmd))
        cmd = "grep -A%i '%s' %s | grep -v -e '%s' -e '--'" \
              %(natoms, key, self.filename, key)
        ret = traj_from_txt(com.backtick(cmd),
                            shape=(natoms,3,nstep_raw),
                            axis=self.time_axis)
        return None if ret is None else ret[...,::2]                            
   
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
    
    def get_volume(self):
        verbose("getting volume")
        key = '^[ ]*ucvol='
        cmd = "grep '%s' %s | awk '{print $2}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))
    
    def get_etot(self):
        verbose("getting etot")
        key = 'end of Moldyn step.*POT.En.'
        cmd = "grep '%s' %s | awk '{print $9}'" %(key, self.filename)
        return arr1d_from_txt(com.backtick(cmd))

    def get_ekin(self):
        verbose("getting ekin")
        req = ['etot_ekin', 'etot']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self.etot_ekin - self.etot
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
    
    # XXX experimental: Abinit does not print temperature. Not sure if T
    # fluctuates or is constant at each step with MTTK (ionmov 13 + optcell 2).
    #
    #   Ekin = 3/2 * natoms * kb * T
    # 
    # This can be seen from abinit/src/95_drive/moldyn.f90 (kb_HaK: K->Ha)
    #   ! v2gauss is twice the kinetic energy
    #   v2gauss = ...
    #   ...
    #   write(message, '(a,d12.5,a,D12.5)' )&
    #   ' --- Effective temperature',v2gauss/(3*dtset%natom*kb_HaK),' From variance', sigma2
    def get_temperature(self):
        verbose("getting temperature")
        req = ['ekin', 'natoms']
        self.check_get_attrs(req)
        if self.is_set_attrs(req):
            return self.ekin * constants.Eh / self.natoms / constants.kb * (2.0/3.0)
        else:
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
