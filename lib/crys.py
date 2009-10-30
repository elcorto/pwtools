# crys.py
#
# Crystal and unit-cell related tools. Converters between file formats.
#

from math import acos, pi, sin, cos, sqrt
from itertools import izip
import re

import numpy as np

try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Cannot import CifFile from the PyCifRW package. " 
    "Some functions in this module will not work." %__file__)

from common import assert_cond
import common
import constants as con
import regex

#-----------------------------------------------------------------------------

def _add_doc(func):
    """Decorator to add common docstrings to functions in this module."""
    dct = {}
    dct['align_doc'] = \
    """align: str
        'rows' : basis vecs are the rows of `cp`
        'cols' : basis vecs are the columns of `cp`"""
    dct['cp_doc'] = \
    """cp: array_like, shape (3,3)
        Matrix with basis vectors."""
    dct['cryst_const_doc'] = \
    """cryst_const: array_like, shape (6,)
        [a, b, c, alpha, beta, gamma]"""
    dct['notes_cp_crys_const'] = \
    """We use PWscf notation.
    CELL_PARAMETERS == (matrix of) primitime basis vectors elsewhere
    crystallographic constants a,b,c,alpha,beta,gamma == cell parameters 
        elsewhere"""
    # Use dictionary string replacement:
    # >>> '%(lala)i %(xxx)s' %{'lala': 3, 'xxx': 'grrr'}
    # '3 grrr'
    func.__doc__ = func.__doc__ % dct 
    return func

#-----------------------------------------------------------------------------
# misc math
#-----------------------------------------------------------------------------

# np.linalg.norm handles also complex arguments, but we don't need that here. 
##norm = np.linalg.norm
def norm(a):
    """2-norm for real vectors."""
    assert_cond(len(a.shape) == 1, "input must be 1d array")
    # math.sqrt is faster then np.sqrt for scalar args
    return sqrt(np.dot(a,a))

#-----------------------------------------------------------------------------

def angle(x,y):
    """Angle between vectors `x' and `y' in degrees.
    
    args:
    -----
    x,y : 1d numpy arrays
    """
    # Numpy's `acos' is "acrcos", but we take the one from math for scalar
    # args.
    return acos(np.dot(x,y)/norm(x)/norm(y))*180.0/pi

#------------------------------------------------------------------------------

def floor_eps(arg, copy=True):
    """sin(180 * pi/180.) == 1e-17, we want 0.0"""
    eps = np.finfo(np.float).eps
    if np.isscalar(arg):
        return 0.0 if abs(arg) < eps else arg
    else:
        if copy:
            _arg = np.asarray(arg).copy()
        else:            
            _arg = np.asarray(arg)
        _arg[np.abs(_arg) < eps] = 0.0
        return _arg

#------------------------------------------------------------------------------

def deg2rad(x):
    return x * pi/180.0

#-----------------------------------------------------------------------------
# crystallographic constants and basis vectors
#-----------------------------------------------------------------------------

@_add_doc
def volume_cp(cp):
    """Volume of the unit cell from CELL_PARAMETERS. Calculates the triple
    product 
        np.dot(np.cross(a,b), c) 
    of the basis vectors a,b,c contained in `cp`. Note that (mathematically)
    the vectors can be either the rows or the cols of `cp`.

    
    args:
    -----
    %(cp_doc)s

    returns:
    --------
    volume, unit: [a]**3

    example:
    --------
    >>> a = [1,0,0]; b = [2,3,0]; c = [1,2,3.];
    >>> m = np.array([a,b,c])
    >>> volume(m)
    9.0
    >>> m = rand(3,3)
    >>> volume(m)
    0.34119414123070052
    >>> volume(m.T)
    0.34119414123070052

    notes:
    ------
    %(notes_cp_crys_const)s
    """    
    assert_cond(cp.shape == (3,3), "input must be (3,3) array")
    return np.dot(np.cross(cp[0,:], cp[1,:]), cp[2,:])        

#-----------------------------------------------------------------------------

@_add_doc
def volume_cc(cryst_const):
    """Volume of the unit cell from crystallographic constants [1].
    
    args:
    -----
    %(cryst_const_doc)s
    
    returns:
    --------
    volume, unit: [a]**3
    
    notes:
    ------
    %(notes_cp_crys_const)s

    refs:
    -----
    [1] http://en.wikipedia.org/wiki/Parallelepiped
    """
    a = cryst_const[0]
    b = cryst_const[1]
    c = cryst_const[2]
    alpha = cryst_const[3]*pi/180
    beta = cryst_const[4]*pi/180
    gamma = cryst_const[5]*pi/180
    return a*b*c*sqrt(1+ 2*cos(alpha)*cos(beta)*cos(gamma) -\
          cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 )


#-----------------------------------------------------------------------------

@_add_doc
def cp2cc(cp, align='rows'):
    """From CELL_PARAMETERS to crystallographic constants a, b, c, alpha, beta,
    gamma.
    
    args:
    -----
    %(cp_doc)s
    %(align_doc)s

    returns:
    --------
    %(cryst_const_doc)s, 
        unit: [a]**3

    notes:
    ------
    %(notes_cp_crys_const)s
    """
    cp = np.asarray(cp)
    assert_cond(cp.shape == (3,3), "cp must be (3,3) array")
    if align == 'cols':
        cp = cp.T
##    print "spat volume:", volume(cp)
    cryst_const = np.empty((6,), dtype=float)
    # a = |a|, b = |b|, c = |c|
    cryst_const[:3] = np.sqrt((cp**2.0).sum(axis=1))
    va = cp[0,:]
    vb = cp[1,:]
    vc = cp[2,:]
    # alpha
    cryst_const[3] = angle(vb,vc)
    # beta
    cryst_const[4] = angle(va,vc)
    # gamma
    cryst_const[5] = angle(va,vb)
    return cryst_const

#-----------------------------------------------------------------------------

@_add_doc
def cc2cp(cryst_const):
    """From crystallographic constants a, b, c, alpha, beta,
    gamma to CELL_PARAMETERS.
    
    args:
    -----
    %(cryst_const_doc)s
    
    returns:
    --------
    %(cp_doc)s
        Basis vecs are the rows.
        unit: [a]**3
    
    notes:
    ------
    * %(notes_cp_crys_const)s
    
    * Basis vectors fulfilling the crystallographic constants are arbitrary
      w.r.t. their orientation in space. We choose the common convention that
        va : along x axis
        vb : in the x-y plane
      Then, vc is fixed. 
      cp = [[-- va --],
            [-- vb --],
            [-- vc --]]
    """
    a = cryst_const[0]
    b = cryst_const[1]
    c = cryst_const[2]
    alpha = cryst_const[3]*pi/180
    beta = cryst_const[4]*pi/180
    gamma = cryst_const[5]*pi/180
    
    va = np.array([a,0,0])
    vb = np.array([b*cos(gamma), b*sin(gamma), 0])
    
    # vc must be calculated ...
    # projection onto x axis (va)
    cx = c*cos(beta)
    
    # Now need cy and cz ...

##    #
##    # Maxima solution
##    #   
##    # volume of the unit cell 
##    vol = volume_cc(cryst_const)
##    print "Maxima: vol", vol
##    cz = vol / (a*b*sin(gamma))
##    print "Maxima: cz", cz
##    cy = sqrt(a**2 * b**2 * c**2 * sin(beta)**2 * sin(gamma)**2 - \
##        vol**2) / (a*b*sin(gamma))
##    print "Maxima: cy", cy
##    cy = sqrt(c**2 - cx**2 - cz**2)
##    print "Pythagoras: cy", cy
    
    # PWscf , WIEN2K's sgroup, results are the same as with Maxima but the
    # formulas are shorter :)
    cy = c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma)
##    print "sgroup: cy", cy
    cz = sqrt(c**2 - cy**2 - cx**2)
##    print "sgroup: cz", cz
    vc = np.array([cx, cy, cz])
    return np.array([va, vb, vc])


#-----------------------------------------------------------------------------
# super cell building
#-----------------------------------------------------------------------------

def scell_mask(dim1, dim2, dim3):
    """Build a mask for the creation of a dim1 x dim2 x dim3 supercell (for 3d
    coordinates).  Return all possible permutations with repitition of the
    integers n1, n2,  n3, and n1, n2, n3 = 0, ..., dim1-1, dim2-1, dim3-1 .

    args:
    -----
    dim1, dim2, dim3 : int

    returns:
    --------
    mask : 2d array, shape (dim1*dim2*dim3, 3)

    example:
    --------
    >>> # 2x2x2 supercell
    >>> scell_mask(2,2,2)
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  1.,  1.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  1.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  1.]])
    >>> # a "plane" of 4 cells           
    >>> scell_mask(2,2,1)
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.]])
    
    notes:
    ------
    If dim1 == dim2 == dim3 == n, then we have a permutation with repetition
    (german: Variation mit Wiederholung):  select r elements out of n with
    rep. In gerneral, n >= r or n < r possible. There are always n**r
    possibilities.
    Here r = 3 always (select x,y,z direction):
    example:
    n=2 : {0,1}   <=> 2x2x2 supercell: 
      all 3-tuples out of {0,1}   -> n**r = 2**3 = 8
    n=3 : {0,1,2} <=> 3x3x3 supercell:
      all 3-tuples out of {0,1,2} -> n**r = 3**3 = 27
    Computationally, we need `r` nested loops (or recursion of depth 3), one
    per dim.  
    """
    b = [] 
    for n1 in range(dim1):
        for n2 in range(dim2):
            for n3 in range(dim3):
                b.append([n1,n2,n3])
    return np.array(b, dtype=float)

#-----------------------------------------------------------------------------

@_add_doc
def raw_scell(R0, cp, mask, symbols, align='rows'):
    """Build supercell based on `mask`.

    args:
    -----
    R0 : 2d array, (natoms, 3) with atomic positions in *crystal* coordinates
        (i.e. in units of the basis vecs in `cp`), these represent the initial
        single unit cell
    %(cp_doc)s        
    mask : what scell_mask() returns, (N, 3)
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of R0
    %(align_doc)s

    returns:
    --------
    (symbols_sc, Rsc)
    symbols_sc : list of strings with atom symbols, (N*natoms,)
    Rsc : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t `cp`, i.e. the numbers
        are not in [0,1], but in [0, max(dims)].

    notes:
    ------
    This is tested for R0 in 'crystal' coords, i.e. in units of `cp`.
    """
    if align == 'cols':
        cp = cp.T
    symbols_sc = []
    Rsc = np.empty((mask.shape[0]*R0.shape[0], 3), dtype=float)
    k = 0
    for iatom in range(R0.shape[0]):
        for j in range(mask.shape[0]):
            # Build supercell. Place each atom N=dim1*dim2*dim3 times in the
            # supercell, i.e. copy unit cell N times. Actually, N-1, since
            # n1=n2=n3=0 is the unit cell itself.
            # mask[j,:] = [n1, n2, n3], ni = integers (floats actually, but
            #   mod(ni, floor(ni)) == 0.0)
            # cp = [[-- a1 --]
            #       [-- a2 --]
            #       [-- a3 --]]
            # dot(...) = n1*a1 + n2*a2 + n3*a3
            # R0[i,:] = r_i = position vect of atom i in the unit cell
            # r_i_in_supercell = r_i + n1*a1 + n2*a2 + n3*a3
            #   for all permutations (see scell_mask()) of n1, n2, n3.
            #   ni = 0, ..., dimi-1
            Rsc[k,:] = R0[iatom,:] + np.dot(mask[j,:], cp)
            symbols_sc.append(symbols[iatom])
            k += 1
    return symbols_sc, Rsc

#-----------------------------------------------------------------------------

@_add_doc
def scell(coords, cp, dims, symbols, align='rows'):
    """Convenience function. Uses raw_scell() and scell_mask(). It scales the
    unit cell to the dims of the super cell and returns crystal atomic
    positions w.r.t this cell.
    
    args:
    -----
    coords : 2d array, (natoms, 3) with atomic positions in *crystal* coordinates
        (i.e. in units of the basis vecs in `cp`), these represent the initial
        single unit cell
    %(cp_doc)s
    dims : tuple (nx, ny, nz) for a N = nx * ny * nz supercell
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of R0
    %(align_doc)s

    returns:
    --------
    (symbols_sc, Rsc_scaled, cp_sc)
    symbols_sc : list of strings with atom symbols for the supercell, (N*natoms,)
    Rsc_scaled : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t `cp_sc`, i.e. the numbers
        are in [0,1].
    cp_sc : array (3,3), basis vecs of the super cell        
    """
    mask = scell_mask(*tuple(dims))
    # in crystal coords w.r.t the *old* cell, i.e. the entries are in
    # [0,(max(dims))], not [0,1]
    symbols_sc, Rsc = raw_scell(coords, cp, mask, symbols, align=align)
    # scale cp acording to super cell dims
    cp_sc = cp * np.asarray(dims)[:,np.newaxis]
    # Rescale crystal coords to cp_sc (coord_trans actually) -> all values in
    # [0,1] again
    Rsc[:,0] /= dims[0]
    Rsc[:,1] /= dims[1]
    Rsc[:,2] /= dims[2]
    return (symbols_sc, Rsc, cp_sc)

#-----------------------------------------------------------------------------
# file parsers / converters
#-----------------------------------------------------------------------------

@_add_doc
def wien_sgroup_input(lat_symbol, symbols, atpos_crystal, cryst_const):
    """Generate input for WIEN2K's sgroup tool.

    args:
    -----
    lat_symbol : str, e.g. 'P'
    symbols : list of strings with atom symbols, (atpos_crystal.shape[0],)
    atpos_crystal : array_like (natoms, 3), crystal ("fractional") atomic
        coordinates
    %(cryst_const_doc)s

    notes:
    ------
    From sgroup's README:

    / ------------------------------------------------------------
    / in input file symbol "/" means a comment
    / and trailing characters are ignored by the program

    / empty lines are allowed

    P  /type of lattice; choices are P,F,I,C,A

    /  parameters of cell:
    /  lengths of the basis vectors and
    /  angles (degree unit is used)  alpha=b^c  beta=a^c  gamma=a^b
    /   |a|  |b|   |c|               alpha  beta  gamma

       1.0   1.1   1.2                90.   91.    92.

    /Number of atoms in the cell
    4

    /List of atoms
    0.1 0.2 0.3  / <-- Atom positions in units of the vectors a b c
    Al           / <-- name of this atom

    0.1 0.2 0.4  /....
    Al1

    0.2 0.2 0.3
    Fe

    0.1 0.3 0.3
    Fe

    / ------------------------------------------------------------------
    """
    atpos_crystal = np.asarray(atpos_crystal)
    assert_cond(len(symbols) == atpos_crystal.shape[0], 
        "len(symbols) != atpos_crystal.shape[0]")
    empty = '\n\n'
    txt = "/ lattice type symbol\n%s" %lat_symbol
    txt += empty
    txt += "/ a b c alpha beta gamma\n"
    txt += " ".join(["%.15g"]*6) % tuple(cryst_const)
    txt += empty
    txt += "/ number of atoms\n%i" %len(symbols)
    txt += empty
    txt += "/ atom list (crystal cooords)\n"
    fmt = ' '.join(['%.15g']*3)
    for sym, coord in izip(symbols, atpos_crystal):
        txt += fmt % tuple(coord) + '\n' + sym + '\n'
    return txt

#-----------------------------------------------------------------------------

class StructureFileParser(object):
    def __init__(self, fn):
        """Abstract base class for all structure file parsing classes. Can
        house common code like unit conversion etc.
        
        Not used atm.
        """
        # Current lenth unit. After parsing, it is the unit of legth of the
        # parsed file (i.e. Angstrom for Cif and PDB). The variable is altered by 
        # self.ang_to_bohr() etc. 
        #
        # With this state variable, we essentially
        # implement a physical quantity: number * unit ...
        self.length_unit = None
        self.fn = fn
    
    def parse(self):
        pass
    
    def ang_to_bohr(self):
        pass
    
    def bohr_to_and(self):
        pass

    def to_bohr(self):
        pass
         
#-----------------------------------------------------------------------------

class CifFile(object):
    def __init__(self, fn, block=None):
        """Extract cell parameters and atomic positions from Cif files. This
        data can be directly included in a pw.x input file. 

        args:
        -----
        fn : str, name of the *cif file
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
        
        Since we return also `cryst_const`, one could also easily obtain the
        CELL_PARAMETERS by pwtools.crys.cc2cp(cryst_const) and wouldn't need
        celldm(1..6) at all.

        refs:
        -----
        [1] http://pycifrw.berlios.de/
        [2] http://www.quantum-espresso.org/input-syntax/INPUT_PW.html#id53713
        """
        self.fn = fn
        self.a0_to_A = con.a0_to_A
        self.block = block
        self.parse()
    
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
        cf = pycifrw_CifFile.ReadCif(self.fn)
        if self.block is None:
            cif_block = cf.first_block()
        else:
            cif_block = cf['data_' + name]
        
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
        self.celldm.append(cos(deg2rad(self.cif_dct['alpha'])))
        self.celldm.append(cos(deg2rad(self.cif_dct['beta'])))
        self.celldm.append(cos(deg2rad(self.cif_dct['gamma'])))
        self.celldm = np.asarray(self.celldm)
        
        self.symbols = map(self.cif_label, cif_block['_atom_site_label'])
        
        self.coords = np.array([map(self.cif_str2float, [x,y,z]) for x,y,z in izip(
                                   cif_block['_atom_site_fract_x'],
                                   cif_block['_atom_site_fract_y'],
                                   cif_block['_atom_site_fract_z'])])
        self.cryst_const = np.array([self.cif_dct[key] for key in \
            ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
        
#------------------------------------------------------------------------------

class PDBFile(object):
    @_add_doc
    def __init__(self, fn):
        """
        Very very simple pdb file parser. Extract only ATOM/HETATM and CRYST1
        (if present) records.
        
        If you want smth serious, check biopython. No unit conversion up to
        now.
        
        args:
        -----
        fn : filename

        members:
        --------
        coords : atomic coords in Bohr
        symbols : list of strings with atom symbols
        %(cryst_const_doc)s 
            If no CRYST1 record is found, this is None.
        
        notes:
        ------
        We use regexes which may not work for more complicated ATOM records. We
        don't use the strict column numbers for each field as stated in the PDB
        spec.
        """
        self.fn = fn
        ##self.a0_to_A = con.a0_to_A
        self.parse()
    
    def parse(self):
        # Grep atom symbols and coordinates in Angstrom ([A]) from PDB file.
        fh = open(self.fn)
        ret = common.igrep(r'(ATOM|HETATM)[\s0-9]+([A-Za-z]+)[\sa-zA-Z0-9]*'
            r'[\s0-9]+((\s+'+ regex.float_re + r'){3}?)', fh)
        # array of string type            
        coords_data = np.array([[m.group(2)] + m.group(3).split() for m in ret])
        # list of strings (system:nat,), fix atom names, e.g. "AL" -> Al
        self.symbols = []
        for sym in coords_data[:,0]:
            if len(sym) == 2:
                self.symbols.append(sym[0] + sym[1].lower())
            else:
                self.symbols.append(sym)
        # float array, (system:nat, 3) in Bohr
        ##self.coords = coords_data[:,1:].astype(float) /self.a0_to_A        
        self.coords = coords_data[:,1:].astype(float)        
        
        # grep CRYST1 record, extract only crystallographic constants
        # example:
        # CRYST1   52.000   58.600   61.900  90.00  90.00  90.00  P 21 21 21   8
        #          a        b        c       alpha  beta   gamma  |space grp|  z-value
        fh.seek(0)
        ret = common.mgrep(r'CRYST1\s+((\s+'+ regex.float_re + r'){6}).*$', fh)
        fh.close()
        if len(ret) == 1:
            match = ret[0]
            self.cryst_const = np.array(match.group(1).split()).astype(float)
            ##self.cryst_const[:3] /= self.a0_to_A
        elif len(ret) == 0:
            self.cryst_const = None
        else:
            raise StandardError("found CRYST1 record more then once")       

