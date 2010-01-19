# parse.py
#
# Parser classes for different file formats. Input- and output files.
#  
# We need the following basic Unix tools installed:
#   grep/egrep
#   GNU sed
#   awk
#   tail
#   wc 
#   ...
# 
# Notes:
# * Some functions/methods are Python-only, but most of them actually call
#   grep/sed/awk. This may not be pythonic, but hey ... these tools rock and
#   the cmd lines are short.
# * pwtools.com.backtick() takes "long" to create a child process. So for small
#   files, pure-python versions, although they have much more code, are faster. 
#   But who cares if the files are small. For big files, grep&friends win + much
#   less code here.
# * Yes, we really need *GNU* sed! (Or, any sed that supports the address
#   syntax "sed '/<patter>/,+<number>'"). Should work on any modern Linux
#   machine. I don't know about all those other exotic Unix flavours
#   you are forced to work on.
# * The tested egrep's don't know the "\s" character class for whitespace
#   as sed, Perl, Python or any other sane regex implementation does. Use 
#   "[ ]" instead.

import re
from math import acos, pi, sin, cos, sqrt
from itertools import izip
from cStringIO import StringIO
import cPickle
import types

import numpy as np
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Cannot import CifFile from the PyCifRW package. " 
    "Some functions in this module will not work." %__file__)

import io
import common as com
import constants
from verbose import verbose
import regex
import crys
from decorators import crys_add_doc


#TODO Get atom masses from periodic_table.py, not from the input file (pw.in).

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


#-----------------------------------------------------------------------------
# Parsers
#-----------------------------------------------------------------------------

class FileParser(object):
    """Base class for file parsers.
    
    Classes derived from this one must at least override self.parse().
    """
    def __init__(self, filename=None):
        """
        args: 
        -----
        filename : str, name of the file to parse
        """
        # Bohr -> Angstrom
        self.a0_to_A = constants.a0_to_A
        
        self.filename = filename
        if self.filename is not None:
            self.file = open(filename)
        else:
            self.file = None
 
    def __del__(self):
        """Destructor. If self.file has not been closed yet (in self.parse()),
        then do it here, eventually."""
        self.close_file()

    def close_file(self):
        if (self.file is not None) and (not self.file.closed):
            self.file.close()
    
    def dump(self, filename):
        # Dumping with protocol "2" is supposed to be the fastest binary format
        # writing method. Probably, this is platform-specific.
        cPickle.dump(self, open(filename, 'wb'), 2)

    def load(self, filename):
        # does not work:
        #   self = cPickle.load(...)
        # 
        # HACK
        # usage:
        # >>> x = FileParser('foo.txt')
        # >>> x.parse()
        # >>> x.dump('foo.pk')
        # # method 1
        # >>> xx = FileParser()
        # >>> xx.load('foo.pk')
        # # method 2, probably easier :)
        # >>> xx = cPickle.load(open('foo.pk'))
        self.__dict__.update(cPickle.load(open(filename, 'rb')).__dict__)
    
    def parse(self):
        pass
    
    def ang_to_bohr(self):
        pass
    
    def bohr_to_ang(self):
        pass

    def to_bohr(self):
        pass
         
    def to_ang(self):
        pass


class StructureFileParser(FileParser):
    """Base class for structure file (pdb, cif, etc) and input file parsers.
    A file parsed by this class is supposed to contain infos about an atomic
    structure.
    
    Classes derived from this one must provide the following members. If a
    particular information is not present in the parsed file, the corresponding
    member must be None.

    self.coords : ndarray (natoms, 3) with atom coords
    self.symbols : list (natoms,) with strings of atom symbols
    self.cell_parameters : 3x3 array with primitive basis vectors as rows
    self.cryst_const : array (6,) with crystallographic costants
        [a,b,c,alpha,beta,gamma]
    self.natoms : number of atoms

    Unless explicitly stated, we DO NOT DO any unit conversions with the data
    parsed out of the files. It is up to the user to handle that. 
    """
    def __init__(self, filename=None):
        FileParser.__init__(self, filename)
        # API
        self.coords = None
        self.symbols = None
        self.cell_parameters = None
        self.cryst_const = None
        self.natoms = None



class CifFile(StructureFileParser):
    def __init__(self, filename=None, block=None):
        """Extract cell parameters and atomic positions from Cif files. This
        data can be directly included in a pw.x input file. 

        args:
        -----
        block : data block name (i.e. 'data_foo' in the Cif file -> 'foo'
            here). If None then the first data block in the file is used.
        
        members:
        --------
        celldm : array (6,), PWscf celldm, see [2]
            [a, b/a, c/a, cos(alpha), cos(beta), cos(gamma)]
            **** NOTE: 'a' is always in Bohr! ****
        symbols : list of strings with atom symbols
        coords : array (natoms, 3), crystal coords
        cif_dct : dct with 'a','b','c' in Angstrom (as parsed from the Cif
            file) and 'alpha', 'beta', 'gamma'
        %(cryst_const_doc)s, same as cif_dct, but as array
        cell_parameters : primitive lattice vectors obtained from cryst_const
            with crys.cc2cp()

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
        StructureFileParser.__init__(self, filename)
        self.block = block
    
    def cif_str2float(self, st):
        """'7.3782(7)' -> 7.3782"""
        if '(' in st:
            st = re.match(r'(' + regex.float_re  + r')(\(.*)', st).group(1)
        return float(st)

    def cif_label(self, st, rex=re.compile(r'([a-zA-Z]+)([0-9]*)')):
        """Remove digits from atom names. 
        
        example:
        -------
        >>> cif_label('Al1')
        'Al'
        """
        return rex.match(st).group(1)
    
    def parse(self):        
        cf = pycifrw_CifFile.ReadCif(self.filename)
        if self.block is None:
            cif_block = cf.first_block()
        else:
            cif_block = cf['data_' + self.block]
        
        # celldm from a,b,c and alpha,beta,gamma
        # alpha = angbe between b,c
        # beta  = angbe between a,c
        # gamma = angbe between a,b
        self.cif_dct = {}
        for x in ['a', 'b', 'c']:
            what = '_cell_length_' + x
            self.cif_dct[x] = self.cif_str2float(cif_block[what])
        for x in ['alpha', 'beta', 'gamma']:
            what = '_cell_angle_' + x
            self.cif_dct[x] = self.cif_str2float(cif_block[what])
        self.celldm = []
        # ibrav 14, celldm(1) ... celldm(6)
        self.celldm.append(self.cif_dct['a']/self.a0_to_A) # Angstrom -> Bohr
        self.celldm.append(self.cif_dct['b']/self.cif_dct['a'])
        self.celldm.append(self.cif_dct['c']/self.cif_dct['a'])
        self.celldm.append(cos(self.cif_dct['alpha']*pi/180))
        self.celldm.append(cos(self.cif_dct['beta']*pi/180))
        self.celldm.append(cos(self.cif_dct['gamma']*pi/180))
        self.celldm = np.asarray(self.celldm)
        
        self.symbols = map(self.cif_label, cif_block['_atom_site_label'])
        
        self.coords = np.array([map(self.cif_str2float, [x,y,z]) for x,y,z in izip(
                                   cif_block['_atom_site_fract_x'],
                                   cif_block['_atom_site_fract_y'],
                                   cif_block['_atom_site_fract_z'])])
        self.cryst_const = np.array([self.cif_dct[key] for key in \
            ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
        self.cell_parameters = crys.cc2cp(self.cryst_const)
        self.natoms = len(self.symbols)
        self.close_file()


class PDBFile(StructureFileParser):
    @crys_add_doc
    def __init__(self, filename=None):
        """
        Very very simple pdb file parser. Extract only ATOM/HETATM and CRYST1
        (if present) records.
        
        If you want smth serious, check biopython. No unit conversion up to
        now.
        
        members:
        --------
        coords : atomic coords in Bohr
        symbols : list of strings with atom symbols
        %(cryst_const_doc)s 
            If no CRYST1 record is found, this is None.
        cell_parameters : primitive lattice vectors obtained from cryst_const
            with crys.cc2cp()

        notes:
        ------
        We use regexes which may not work for more complicated ATOM records. We
        don't use the strict column numbers for each field as stated in the PDB
        spec.
        """
        StructureFileParser.__init__(self, filename)
    
    def parse(self):
        # Grep atom symbols and coordinates in Angstrom ([A]) from PDB file.
        #
        # XXX Note that for the atom symbols, we do NOT use the columns 77-78
        #     ("Element symbol"), b/c that is apparently not present in all the
        #     files which we currently use. Instead, we use the columns 13-16,
        #     i.e. "Atom name". Note that in general this is not the element
        #     symbol.
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
        #
        ret = com.igrep(r'(ATOM|HETATM)[\s0-9]+([A-Za-z]+)[\sa-zA-Z0-9]*'
            r'[\s0-9]+((\s+'+ regex.float_re + r'){3}?)', self.file)
        # array of string type            
        coords_data = np.array([[m.group(2)] + m.group(3).split() for m in ret])
        # list of strings (system:nat,) 
        # Fix atom names, e.g. "AL" -> Al. Note that this is only needed b/c we
        # use the "wrong" column "Atom name".
        self.symbols = []
        for sym in coords_data[:,0]:
            if len(sym) == 2:
                self.symbols.append(sym[0] + sym[1].lower())
            else:
                self.symbols.append(sym)
        # float array, (system:nat, 3)
        self.coords = coords_data[:,1:].astype(float)        
        
        # grep CRYST1 record, extract only crystallographic constants
        # example:
        # CRYST1   52.000   58.600   61.900  90.00  90.00  90.00  P 21 21 21   8
        #          a        b        c       alpha  beta   gamma  |space grp|  z-value
        self.file.seek(0)
        ret = com.mgrep(r'CRYST1\s+((\s+'+ regex.float_re + r'){6}).*$', self.file)
        self.close_file()
        if len(ret) == 1:
            match = ret[0]
            self.cryst_const = np.array(match.group(1).split()).astype(float)
            self.cell_parameters = crys.cc2cp(self.cryst_const)            
        elif len(ret) == 0:
            self.cryst_const = None
        else:
            raise StandardError("found CRYST1 record more then once")       
        self.natoms = len(self.symbols)


class PwInputFile(StructureFileParser):
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

    def parse(self):
        self.atspec = self.get_atomic_species()
        self.atpos = self.get_atomic_positions()
        self.namelists = self.get_namelists()

        # API
        self.coords = self.atpos['coords']
        self.symbols = self.atpos['symbols']
        self.cell_parameters = self.get_cell_parameters()
        if self.cell_parameters is not None:
            self.cryst_const = crys.cp2cc(self.cell_parameters)
        self.natoms = int(self.namelists['system']['nat'])
        self.close_file()
        
    def get_atomic_species(self):
        """Parses ATOMIC_SPECIES card in a pw.x input file.

        returns:
        --------
        {'symbols': symbols, 'masses': masses, 'pseudos': pseudos}
        
        symbols : list of strings, (number_of_atomic_species,), 
            ['Si', 'O', 'Al', 'N']
        masses : 1d arary, (number_of_atomic_species,)
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
        self.file.seek(0)
        verbose('[get_atomic_species] reading ATOMIC_SPECIES from %s' %self.filename)
        # rex: for the pseudo name, we include possible digits 0-9 
        rex = re.compile(r'\s*([a-zA-Z]+)\s+(' + regex.float_re +\
            ')\s+(.*)$')
        self.file, flag = scan_until_pat(self.file, 
                                         pat='atomic_species',        
                                         err=False)
        if flag == 0:
            verbose("[get_atomic_species]: WARNING: start pattern not found")
            return None
        line = next_line(self.file)
        while line == '':
            line = next_line(self.file)
        match = rex.match(line)
        lst = []
        # XXX Could use knowledge of namelists['system']['ntyp'] here (=number
        # of lines in this card) if we parse the namelists first
        while match is not None:
            # match.groups: tuple ('Si', '28.0855', 'Si.LDA.fhi.UPF')
            lst.append(list(match.groups()))
            line = next_line(self.file)
            match = rex.match(line)
        if lst == []:
            verbose("[get_atomic_species]: WARNING: nothing found")
            return None
        # numpy string array :)
        ar = np.array(lst)
        symbols = ar[:,0].tolist()
        masses = np.asarray(ar[:,1], dtype=float)
        pseudos = ar[:,2].tolist()
        return {'symbols': symbols, 'masses': masses, 'pseudos': pseudos}


    def get_cell_parameters(self):
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
        """
        self.file.seek(0)
        verbose('[get_cell_parameters] reading CELL_PARAMETERS from %s' %self.filename)
    ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        rex = re.compile(r'\s*((' + regex.float_re + '\s*){3})\s*')
        self.file, flag = scan_until_pat(self.file, pat="cell_parameters",
                                         err=False)
        if flag == 0:
            verbose("[get_cell_parameters]: WARNING: start pattern not found")
            return None
        line = next_line(self.file)
        while line == '':
            line = next_line(self.file)
        match = rex.match(line)
        lst = []
        # XXX Could use <number_of_lines> = 3 instead of regexes
        while match is not None:
            # match.groups(): ('1.3 0 3.0', ' 3.0')
            lst.append(match.group(1).strip().split())
            line = next_line(self.file)
            match = rex.match(line)
        if lst == []:
            verbose("[get_cell_parameters]: WARNING: nothing found")
            return None
        cp = np.array(lst, dtype=float)
    #----------------------------------------------------------
    ##    cmd = "sed -nre '/CELL_PARAMETERS/,+3p' %s | tail -n3" %self.filename
    ##    cp = np.loadtxt(StringIO(com.backtick(cmd)))
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        com.assert_cond(len(cp.shape) == 2, "`cp` is no 2d array")
        com.assert_cond(cp.shape[0] == cp.shape[1], "dimensions of `cp` don't match")
        return cp

    def get_atomic_positions(self):
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
        self.file.seek(0)
        verbose("[get_atomic_positions] reading ATOMIC_POSITIONS from %s" %self.filename)
        if self.atspec is None:
            self.atspec = self.get_atomic_species()
            # need to seek to start of file here
            self.file.seek(0)
        rex = re.compile(r'\s*([a-zA-Z]+)((\s+' + regex.float_re + '){3})\s*')
        self.file, flag, line = scan_until_pat(self.file, 
                                               pat="atomic_positions", 
                                               retline=True)
        if flag == 0:
            verbose("[get_atomic_positions]: WARNING: start pattern not found")
            return None
        line = line.strip().lower().split()
        if len(line) > 1:
            unit = re.sub(r'[{\(\)}]', '', line[1])
        else:
            unit = ''
        line = next_line(self.file)
        while line == '':
            line = next_line(self.file)
        lst = []
        # XXX Instead of regexes, we could as well use natoms
        # (namelists['system']['nat']).
        match = rex.match(line)
        while match is not None:
            # match.groups():
            # ('Al', '       4.482670384  -0.021685570   4.283770714', '    4.283770714')
            lst.append([match.group(1)] + match.group(2).strip().split())
            line = next_line(self.file)
            match = rex.match(line)
        if lst == []:
            verbose("[get_atomic_positions]: WARNING: nothing found")
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
        self.file.seek(0)
        verbose("[get_namelists] parsing %s" %self.filename)
        dct = {}
        nl_kvps = None
        for line in self.file:
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


class PwOutputFile(FileParser):
    def __init__(self, filename=None, infile=None):
        """Parse a pw.x output file. This class is primarily geared towards
        pw.x MD runs.
        
        args:
        -----
        filename : file to parse
        infile : "parsed" PwInputFile object or filename
            # object
            >>> pwin = PwInputFile('pw.in')
            >>> pwin.parse()
            >>> pwout = PwOutputFile('pw.out', pwin)
            # filename
            >>> pwout = PwOutputFile('pw.out', 'pw.in')
            In case it is a filename, the "pwin" object will be constructed
            here in the way shown above.
        
        members:
        --------
        etot : 1d array (nstep,)
        ekin : 1d array (nstep,)
        stresstensor : 3d array (3, nstep, 3), stress tensor for each step 
            XXX time axis hardcoded!
        pressure : 1d array (nstep,), this is parsed from the "P= ... kbar"
            lines and the value is actually 1/3*trace(stress tensor)
        temperature : 1d array (nstep,)
        coords : 3d array (natoms, nstep, 3) XXX time axis hardcoded!
        cell_parameters : 3d array (3, nstep, 3), prim. basis vectors for each 
            step
        nstep : number of MD steps
        natoms : number of atoms
        
        Members, whose corresponding data in the file is not present, are None.
        E.g. there are no CELL_PARAMETERS printed in the output file, so
        self.cell_parameters == None.

        notes:
        ------
        Why do we need the input file anyway? For pw.x MDs, normally all infos
        are contained in the output file alone. Also often, the output file is
        actually a concatenation of smaller files ("$ cat pw.out.r* >
        pw.out.all"). In that case, things line "nstep" are of course wrong if
        taken from an input file. 
        
        But certain quantinies must be the same (e.g. "nat" == natoms) in
        input- and output file. This can be used to sanity-check the parsed
        results. Also, they are often easier extracted from an input file.
        """

        FileParser.__init__(self, filename)
        self.infile_inp = infile
        com.assert_cond(infile is not None, "infile is None")
        if isinstance(infile, types.StringType):
            self.infile = PwInputFile(infile)
            self.infile.parse()
        elif isinstance(infile, PwInputFile):
            self.infile = infile
        else:
            raise ValueError("infile must be string or instance of "
                             "PwInputFile")

        verbose("parsing %s" %self.filename)
        
        #########################################################################
        # Unlike the old parse_pwout_md, we IGNORE the first coords (the one
        # from input) as well as the first temperature (which is printed is
        # "Starting temperature") in md and vc-md. We grep only for nstep
        # occurrences, NOT nstep+1 !!! 
        #
        # But, w/ the current approach, we get nstep+1 pressure values, the
        # ones for the MD are then self.pressure[1:]
        #########################################################################
        
        # FIXME It MAY be that with the grep-type approch, we grep the last coords
        # twice for relax runs b/c they are printed twice.
        
        # TODO 
        # * Check if this grepping also works for scf runs, where we have in
        #   principle nstep=1
        # * introduce new variable (int): self.time_axis to contruct slice
        #   stuff for reading "3d arrays", dont hard-code axis=1 everywhere
        # * implement array-write methods 
        
    def parse(self):
        # Must be first. self.natoms needed in several functions below
        self.natoms = self.get_natoms()

        self.etot = self.get_etot()
        self.ekin = self.get_ekin()
        self.stresstensor = self.get_stresstensor()
        self.pressure = self.get_pressure()
        self.temperature = self.get_temperature()
        self.coords = self.get_coords()
        self.cell_parameters = self.get_cell_parameters()
        
        # this depends on self.coords
        self.nstep = self.get_nstep()

        # sanity check
        com.assert_cond(self.coords.shape[0] == self.infile.natoms, 
                    "natoms from infile (%s: %i)  and outfile (%s: %i) don't "
                    "match" %(self.infile.filename, self.infile.natoms,
                    self.filename, self.coords.shape[0]))
        if self.infile.namelists['system'].has_key('nstep'):
            if self.nstep != self.infile.namelists['system']['nstep']:
                print("WARNING: nstep from infile (%s) and outfile (%s) "
                      "don't match" %(self.infile.filename, self.filename))
        self.close_file()
    
    def get_nstep(self):
        verbose("getting nstep")
        if self.coords is not None:
            # XXX time axis hardcoded!
            return self.coords.shape[1]
        else:
            return None
    
    def get_natoms(self):
        return self.infile.natoms

    def get_stresstensor(self):
        verbose("getting stress tensor")
        key = 'P='
        cmd = "grep %s %s | wc -l" %(key, self.filename)
        ret_str = com.backtick(cmd)          
        if ret_str.strip() == '':
            nstep = 0
        else:
            nstep = int(ret_str)
        cmd = "sed -nre '/%s/,+3p' %s | grep -v %s | \
              awk '{printf $4\"  \"$5\"  \"$6\"\\n\"}'" \
              %(key, self.filename, key)
        ret_str = com.backtick(cmd)          
        if ret_str.strip() == '':
            return None
        else:
            return io.readtxt(StringIO(ret_str), axis=1, 
                              shape=(3, nstep, 3))

    def get_etot(self):
        verbose("getting etot")
        ret_str = com.backtick(r"grep '^!' %s | awk '{print $5}'" \
                               %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return np.loadtxt(StringIO(ret_str))
    
    def get_ekin(self):
        verbose("getting ekin")
        ret_str = com.backtick(r"grep 'kinetic energy' %s | awk '{print $5}'"\
                               %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return np.loadtxt(StringIO(ret_str))

    def get_pressure(self):
        verbose("getting pressure")
        ret_str = com.backtick(r"grep P= %s | awk '{print $6}'" %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return np.loadtxt(StringIO(ret_str))
     
    def get_temperature(self):
        verbose("getting temperature")
        cmd = r"egrep 'temperature[ ]*=' %s " %self.filename + \
              "| sed -re 's/.*temp.*=\s*(" + regex.float_re + \
              r")\s*K/\1/'"
        ret_str = com.backtick(cmd)
        if ret_str.strip() == '':
            return None
        else:            
            return np.loadtxt(StringIO(ret_str))
    
    def get_coords(self):
        verbose("getting atomic positions")
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
##        # natoms
##        #
##        # tail ... b/c this line appears multiple times if the output file
##        # is a concatenation of multiple smaller files
##        cmd = r"egrep 'number[ ]+of[ ]+atoms' %s | \
##                sed -re 's/.*=(.*)$/\1/' | tail -n1" %self.filename
##        ret_str = com.backtick(cmd)
##        if ret_str.strip() == '':
##            natoms = 0
##        else:            
##            natoms = int(ret_str)
#--------------------------------------------------
        natoms = self.natoms
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # nstep 
        #
        # get it from outfile b/c the value in any input file will be
        # wrong if the output file is a concatenation of multiple smaller files
        key = 'ATOMIC_POSITIONS'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        ret_str = com.backtick(cmd)
        if ret_str.strip() == '':
            nstep = 0
        else:            
            nstep = int(ret_str)
        # coords
        cmd = "sed -nre '/%s/,+%ip' %s | grep -v %s | \
              awk '{printf $2\"  \"$3\"  \"$4\"\\n\"}'" \
              %(key, natoms, self.filename, key)
        ret_str = com.backtick(cmd)          
        if ret_str.strip() == '':
            return None
        else:
            return io.readtxt(StringIO(ret_str), axis=1, 
                              shape=(natoms, nstep, 3))

    def get_cell_parameters(self):
        verbose("getting cell parameters")
        # nstep
        key = 'CELL_PARARAMETERS'
        cmd = 'grep %s %s | wc -l' %(key, self.filename)
        ret_str = com.backtick(cmd)
        if ret_str.strip() == '':
            nstep = 0
        else:            
            nstep = int(ret_str)
        # cell_parameters            
        cmd = "sed -nre '/%s/,+3p' %s | grep -v %s" %(key, self.filename, key)
        ret_str = com.backtick(cmd)
        if ret_str.strip() == '':
            return None
        else:
            # XXX time axis hardcoded!
            return io.readtxt(StringIO(ret_str), axis=1, shape=(3, nstep, 3))


class CPOutputFile(PwOutputFile):
    """
    Some notes on parsing cp.x output data.

    For the following explanations:
      iprint = self.infile.namelists['control']['iprint']
      isave  = self.infile.namelists['control']['isave']

    The cp.x code writes several text files to scratch:

    Fortran unit | filename  | content (found out by myself)
    30  <prefix>.con  ?
    31  <prefix>.eig  ?
    32  <prefix>.pol  ?
    33  <prefix>.evp  temperature etc. at iprint steps
    34  <prefix>.vel  atomic velocities
    35  <prefix>.pos  atomic positions (self.coords)
    36  <prefix>.cel  cell parameters (self.cell_parameters)
    37  <prefix>.for  forces on atoms 
    38  <prefix>.str  stress tensors  (self.stresstensor)
    39  <prefix>.nos  Nose thetmostat stuff every iprint steps
    40  <prefix>.the  ?
    41  <prefix>.spr  ?
    42  <prefix>.wfc  ? (wafefunctions)?

    Of course, there is no docu as to what they contain! But why should
    there be, this would be useful then.

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
    .evp file and some from the .nos file. Useless shit! 

    The important thing is that we cannot grep these lines from the
    outfile b/c this is ambiguous. There may be (in fact there are!)
    other lines with 12 numbers :) So .. we have to load the crappy
    file.
    """
    def __init__(self, filename=None, infile=None, evpfilename=None):
        PwOutputFile.__init__(self, filename, infile)
        self.evpfilename = evpfilename
        # columns of self.evpfilename
        self.evp_order = \
            ['nfi', 'ekinc', 'temphc', 'tempp', 'etot', 'enthal', 'econs', 
             'econt', 'volume', 'out_press', 'tps']
    
    # from base class:
    #   parse()
    #   get_nstep() 
    #   get_natoms()
    #   get_coords()
    #   get_cell_parameters()
    
    def parse(self):
        com.assert_cond(self.evpfilename is not None, "self.evpfilename is None")
        self.evp_data = self.load_evp_file()
        PwOutputFile.parse(self)
    
    def load_evp_file(self):
        """
        Load the file /path/to/scratch/<prefix>.evp. It contains temperature
        etc. at every iprint step. See self.evp_order .        
        """ 
        verbose("loading evp file: %s" %self.evpfilename)
        return np.loadtxt(self.evpfilename)

    def get_stresstensor(self):
        verbose("getting stress tensor")
        key = "Total stress"
        cmd = "grep '%s' %s | wc -l" %(key, self.filename)
        ret_str = com.backtick(cmd)          
        if ret_str.strip() == '':
            nstep = 0
        else:
            nstep = int(ret_str)
        cmd = "sed -nre '/%s/,+3p' %s | grep -v '%s'" %(key, self.filename, key)
        ret_str = com.backtick(cmd)          
        if ret_str.strip() == '':
            return None
        else:
            return io.readtxt(StringIO(ret_str), axis=1, 
                              shape=(3, nstep, 3))
    
    def get_etot(self):
        verbose("getting etot")
        return self.evp_data[:, self.evp_order.index('etot')]
    
    def get_ekin(self):
        verbose("getting ekin")
        ret_str = com.backtick(r"egrep 'kinetic[ ]+energy[ ]=' %s | "\
                                "awk '{print $4}'" %self.filename)
        if ret_str.strip() == '':
            return None
        else:            
            return np.loadtxt(StringIO(ret_str))
    
    def get_pressure(self):
        verbose("getting pressure")
##        if self.stresstensor is not None:
##            # FIXME: axis=1 as time axis hard-coded here! 
##            return np.trace(self.stresstensor, axis1=0, axis2=2)/3.0
##        else:
##            raise StandardError("self.stresstensor is None")  
        return self.evp_data[:, self.evp_order.index('out_press')]
    
    def get_temperature(self):
        verbose("getting temperature")
        return self.evp_data[:, self.evp_order.index('tempp')]
    

