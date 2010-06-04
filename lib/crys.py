# crys.py
#
# Crystal and unit-cell related tools.
#

from math import acos, pi, sin, cos, sqrt
from itertools import izip

import numpy as np

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Warning: Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work." %__file__)

from common import assert_cond
import common
import constants as con
from decorators import crys_add_doc

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


def angle(x,y):
    """Angle between vectors `x' and `y' in degrees.
    
    args:
    -----
    x,y : 1d numpy arrays
    """
    # Numpy's `acos' is "acrcos", but we take the one from math for scalar
    # args.
    return acos(np.dot(x,y)/norm(x)/norm(y))*180.0/pi


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


def deg2rad(x):
    return x * pi/180.0

#-----------------------------------------------------------------------------
# crystallographic constants and basis vectors
#-----------------------------------------------------------------------------

@crys_add_doc
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
    >>> volume_cp(m)
    9.0
    >>> m = rand(3,3)
    >>> volume_cp(m)
    0.34119414123070052
    >>> volume_cp(m.T)
    0.34119414123070052

    notes:
    ------
    %(notes_cp_crys_const)s
    """    
    assert_cond(cp.shape == (3,3), "input must be (3,3) array")
    return np.dot(np.cross(cp[0,:], cp[1,:]), cp[2,:])        


@crys_add_doc
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


@crys_add_doc
def cp2cc(cp, align='rows'):
    """From CELL_PARAMETERS to crystallographic constants a, b, c, alpha, beta,
    gamma. 
    This mapping is unique in the sense that multiple `cp`s will have
    the same` cryst_const`, i.e. the representation of the cell in
    `cryst_const` is independent from the spacial orientation of the cell
    w.r.t. a cartesian coord sys.
    
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


@crys_add_doc
def cc2cp(cryst_const):
    """From crystallographic constants a, b, c, alpha, beta,
    gamma to CELL_PARAMETERS.
    This mapping not NOT unique in the sense that one set of `cryst_const` can
    have arbitrarily many representations in terms of `cp`s. We stick to a
    common convention. See notes below.
    
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


@crys_add_doc
def recip_cp(cp, align='rows'):
    """Reciprocal lattice vectors.
        {a,b,c}* = 2*pi / V * {b,c,a} x {c, a, b}
    
    The volume is calculated using `cp`, so make sure that all units match.

    args:
    -----
    %(cp_doc)s
    %(align_doc)s

    returns:
    --------
    Shape (3,3) numpy array with reciprocal vectors as rows.

    notes:
    ------
    %(notes_cp_crys_const)s

    The unit of the recip. vecs is 1/[cp] and the unit of the volume is
    [cp]**3.
    """
    cp = np.asarray(cp)
    assert_cond(cp.shape == (3,3), "cp must be (3,3) array")
    if align == 'cols':
        cp = cp.T
    cp_recip = np.empty_like(cp)
    vol = volume_cp(cp)
    a = cp[0,:]
    b = cp[1,:]
    c = cp[2,:]
    cp_recip[0,:] = 2*pi/vol * np.cross(b,c)
    cp_recip[1,:] = 2*pi/vol * np.cross(c,a)
    cp_recip[2,:] = 2*pi/vol * np.cross(a,b)
    return cp_recip

#-----------------------------------------------------------------------------
# super cell building
#-----------------------------------------------------------------------------

def scell_mask(dim1, dim2, dim3):
    """Build a mask for the creation of a dim1 x dim2 x dim3 supercell (for 3d
    coordinates).  Return all possible permutations with repitition of the
    integers n1, n2, n3, and n1, n2, n3 = 0, ..., dim1-1, dim2-1, dim3-1 .

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


@crys_add_doc
def raw_scell(R0, mask, symbols):
    """Build supercell based on `mask`. This function does only translate the
    atomic coords. it does NOT do anything with the crystal axes. See scell()
    for that.

    args:
    -----
    R0 : 2d array, (natoms, 3) with atomic positions in *crystal* (fractional)
        coordinates (i.e. in units of the basis vecs in `cp`, for instance in .cif
        files _atom_site_fract_*), these represent the initial single unit cell
    mask : what scell_mask() returns, (N, 3)
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of R0

    returns:
    --------
    {symbols, coords)
    symbols : list of strings with atom symbols, (N*natoms,)
    coords : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t the original cell, 
        i.e. the numbers are not in [0,1], but in [0, max(dims)].

    notes:
    ------
    This is tested for R0 in 'crystal' coords, i.e. in units of `cp`.
    """
    # Build supercell 
    # ---------------
    # Place each atom N = dim1*dim2*dim3 times in the
    # supercell, i.e. copy unit cell N times. Actually, N-1, since
    # n1=n2=n3=0 is the unit cell itself.
    #
    # mask[j,:] = [n1, n2, n3], ni = integers (floats actually, but
    #   mod(ni, floor(ni)) == 0.0)
    #
    # original cell:
    # R0[i,:] = r_i = position vect of atom i in the unit cell in *crystal*
    #           coords!!
    # 
    # super cell:
    # r*_i = r_i + [n1, n2, n3]
    #   for all permutations (see scell_mask()) of n1, n2, n3.
    #   ni = 0, ..., dim_i - 1, i = 1,2,3
    symbols_sc = []
    Rsc = np.empty((mask.shape[0]*R0.shape[0], 3), dtype=float)
    k = 0
    for iatom in range(R0.shape[0]):
        for j in range(mask.shape[0]):
            Rsc[k,:] = R0[iatom,:] + mask[j,:]
            symbols_sc.append(symbols[iatom])
            k += 1
    return {'symbols': symbols_sc, 'coords': Rsc}


@crys_add_doc
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
    {symbols, coords, cell_parameters}
    symbols : list of strings with atom symbols for the supercell, (N*natoms,)
    ocoords : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t `cp_sc`, i.e. the numbers
        are in [0,1].
    cell_parameters : array (3,3), basis vecs of the super cell        
    """
    cp = np.asarray(cp)
    assert_cond(cp.shape == (3,3), "cp must be (3,3) array")
    if align == 'cols':
        cp = cp.T
    mask = scell_mask(*tuple(dims))
    # Rsc : crystal coords w.r.t the *old* cell, i.e. the entries are in
    # [0,(max(dims))], not [0,1]
    sc = raw_scell(coords, mask, symbols)
    # scale cp acording to super cell dims
    cp_sc = cp * np.asarray(dims)[:,np.newaxis]
    # Rescale crystal coords to new bigger cell_parameters (coord_trans
    # actually) -> all values in [0,1] again
    sc['coords'][:,0] /= dims[0]
    sc['coords'][:,1] /= dims[1]
    sc['coords'][:,2] /= dims[2]
    return {'symbols': sc['symbols'], 'coords': sc['coords'], 
            'cell_parameters': cp_sc}

#-----------------------------------------------------------------------------
# file parsers / converters
#-----------------------------------------------------------------------------

@crys_add_doc
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


@crys_add_doc
def write_cif(filename, coords, symbols, cryst_const, fac=con.a0_to_A, conv=False):
    """Q'n'D Cif writer. Should be a method of parse.StructureFileParser ....
    stay tuned.
    
    args:
    -----
    filename : str
        name of output .cif file
    coords : array (natoms, 3)
        crystal (fractional) coords
    symbols : list of strings
        atom symbols
    %(cryst_const_doc)s
    fac : conv factor Bohr -> Ang (.cif wants Angstrom)
    conv: bool
        Convert cryst_const[:3] to Ang
    """
    cf = pycifrw_CifFile.CifFile()
    block = pycifrw_CifFile.CifBlock()
    symbols = list(symbols)

    # Bohr -> A
    if conv:
        # nasty trick, make local var with same name, otherwise, 'cryst_const'
        # in global scope (module level= gets changed!
        cryst_const = cryst_const.copy()
        cryst_const[:3] *= fac
    # cell
    #
    # dunno why I have to use str() here, assigning floats does not work
    block['_cell_length_a'] = str(cryst_const[0])
    block['_cell_length_b'] = str(cryst_const[1])
    block['_cell_length_c'] = str(cryst_const[2])
    block['_cell_angle_alpha'] = str(cryst_const[3])
    block['_cell_angle_beta'] = str(cryst_const[4])
    block['_cell_angle_gamma'] = str(cryst_const[5])
    block['_symmetry_space_group_name_H-M'] = 'P 1'
    block['_symmetry_Int_Tables_number'] = 1
    # assigning a list produces a "loop_"
    block['_symmetry_equiv_pos_as_xyz'] = ['x,y,z']
    
    # atoms
    #
    # _atom_site_label: We just use symbols, which is then =
    #   _atom_site_type_symbol, but we *could* use that to number atoms of each
    #   specie, e.g. Si1, Si2, ..., Al1, Al2, ...
    data_names = ['_atom_site_label', 
                  '_atom_site_fract_x',
                  '_atom_site_fract_y',
                  '_atom_site_fract_z',
                  '_atom_site_type_symbol']
    data = [symbols, 
            coords[:,0].tolist(), 
            coords[:,1].tolist(), 
            coords[:,2].tolist(),
            symbols]
    # "loop_" with multiple columns            
    block.AddCifItem([[data_names], [data]])                
    cf['pwtools'] = block
    common.file_write(filename, str(cf))


#XXX implement PBC!
def rmsd(coords_cart, ref_idx=0):
    """Root mean square distance of an MD trajectory of a whole atomic
    structure. For now, use the 3d array R (as used in pydos.py) with time 
    axis = 1.
    
    args:
    -----
    coords_cart : 3d array with atom coords, time axis = 1
    ref_idx : time index of the reference structure (i.e. 0 to compare with the
        start structure).
    """
    R = coords_cart.copy()
##    for j in range(R.shape[1]):
##        R[:,j,:] -= R[:,ref_idx,:]
    R -= R[:,ref_idx,:][:,np.newaxis,:]
    R = R**2.0
    N = float(R.shape[0])
    return np.sqrt(R.sum(axis=2).sum(axis=0)/N)


#XXX check ME!!!
def max_displacement(coords_cart):
    R = coords_cart
    md = np.empty((R.shape[0], R.shape[2]), dtype=float)
    # iatom
    for i in range(R.shape[0]):
        # x,y,z
        for j in range(R.shape[2]):
            md[i,j] = R[i,:,j].max() - R[i,:,j].min()
    return md            

