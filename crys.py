# crys.py
#
# Crystal and unit-cell related tools. Some MD analysis tools, too.
#

from math import acos, pi, sin, cos, sqrt
from itertools import izip

import numpy as np
from scipy.linalg import inv
from scipy.integrate import cumtrapz

# Cif parser
try:
    import CifFile as pycifrw_CifFile
except ImportError:
    print("%s: Warning: Cannot import CifFile from the PyCifRW package. " 
    "Parsing Cif files will not work." %__file__)

from pwtools.common import assert_cond
import pwtools.common as common
import pwtools.constants as con
from pwtools.decorators import crys_add_doc
from pwtools.pwscf import atpos_str

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
    >>> volume_cp(m.T)
    9.0
    >>> m = rand(3,3)
    >>> volume_cp(m)
    0.11844733769775126
    >>> volume_cp(m.T)
    0.11844733769775123
    >>> np.linalg.det(m)
    0.11844733769775125
    >>> np.linalg.det(m.T)
    0.11844733769775125

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
    the same `cryst_const`, i.e. the representation of the cell in
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
    # vc must be calculated:
    # cx: projection onto x axis (va)
    cx = c*cos(beta)
    # Now need cy and cz ...
    #
    # Maxima solution
    #   
    ## vol = volume_cc(cryst_const)
    ## cz = vol / (a*b*sin(gamma))
    ## cy = sqrt(a**2 * b**2 * c**2 * sin(beta)**2 * sin(gamma)**2 - \
    ##     vol**2) / (a*b*sin(gamma))
    ## cy2 = sqrt(c**2 - cx**2 - cz**2)
    #
    # PWscf , WIEN2K's sgroup, results are the same as with Maxima but the
    # formulas are shorter :)
    cy = c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma)
    cz = sqrt(c**2 - cy**2 - cx**2)
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
def raw_scell(coords, mask, symbols):
    """Build supercell based on `mask`. This function does only translate the
    atomic coords. it does NOT do anything with the crystal axes. See scell()
    for that.

    args:
    -----
    coords : 2d array, (natoms, 3) with atomic positions in *crystal* (fractional)
        coordinates (i.e. in units of the basis vecs in `cp`, for instance in .cif
        files _atom_site_fract_*), these represent the initial single unit cell
    mask : what scell_mask() returns, (N, 3)
    symbols : list of strings with atom symbols, (natoms,), must match with the
        rows of coords

    returns:
    --------
    {symbols, coords)
    symbols : list of strings with atom symbols, (N*natoms,)
    coords : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t the original cell, 
        i.e. the numbers are not in [0,1], but in [0, max(dims)].

    notes:
    ------
    This is tested for coords in 'crystal' coords, i.e. in units of `cp`.
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
    # coords[i,:] = r_i = position vect of atom i in the unit cell in *crystal*
    #           coords!!
    # 
    # super cell:
    # r*_i = r_i + [n1, n2, n3]
    #   for all permutations (see scell_mask()) of n1, n2, n3.
    #   ni = 0, ..., dim_i - 1, i = 1,2,3
    symbols_sc = []
    coords_sc = np.empty((mask.shape[0]*coords.shape[0], 3), dtype=float)
    k = 0
    for iatom in range(coords.shape[0]):
        for j in range(mask.shape[0]):
            coords_sc[k,:] = coords[iatom,:] + mask[j,:]
            symbols_sc.append(symbols[iatom])
            k += 1
    return {'symbols': symbols_sc, 'coords': coords_sc}


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
        rows of coords
    %(align_doc)s

    returns:
    --------
    {symbols, coords, cell_parameters}
    symbols : list of strings with atom symbols for the supercell, (N*natoms,)
    coords : array (N*natoms, 3)
        Atomic crystal coords in the super cell w.r.t `cell_parameters`, i.e.
        the numbers are in [0,1].
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
    txt += " ".join(["%.16e"]*6) % tuple(cryst_const)
    txt += empty
    txt += "/ number of atoms\n%i" %len(symbols)
    txt += empty
    txt += "/ atom list (crystal cooords)\n"
    fmt = ' '.join(['%.16e']*3)
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
        # in global scope (module level) gets changed!
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


@crys_add_doc
def write_axsf(filename, coords, cp, symbols, time_axis=-1, atoms_axis=0, align='rows'):
    """Write animated XSF file. ATM, only fixed cells, i.e. `cp` cannot be 3d
    array, in pwscf: md, relax, not vc-md, vc-relax. Forces are all set to
    zero.

    Note that `cp` must be in Angstrom, not the usual PWscf style scaled `cp`
    in alat units.
    
    args:
    -----
    filename : target file name
    coords : 2d (one unit cell) or 3d array (e.g. MD trajectory)
        crystal (fractional) coords,
        examples:
        2d: (natoms, 3) : atoms_axis=0, time_axis not used
        3d: (natoms, 3, nstep) : atoms_axis=0, time_axis=2
    %(cp_doc)s 
        In Angstrom units.
    symbols : list of strings (natoms,)
        atom symbols
    time_axis : int, optional
        axis number for time (if coords is 3d)
    atoms_axis : int, optional
        axis along which atomic coords [x,y,z] are aligned
    %(align_doc)s

    refs:
    -----
    [XSF] http://www.xcrysden.org/doc/XSF.html
    """
    # notes:
    # ------
    # The XSF spec [XSF] is a little fuzzy about what PRIMCOORD actually is
    # (fractional or cartesian Angstrom). Only the latter case results in a
    # correctly displayed structure in xcrsyden. So we use that.
    #
    # We could extend this to variable cell by allowing `cp` to be 3d and
    # accept an 3d array for forces, too. Then we had (together with
    # parse.Pw*OutputFile) a replacement for pwo2xsf.sh .
    
    if align == 'cols':
        cp = cp.T
    is3d = coords.ndim == 3
    natoms = coords.shape[atoms_axis]
    if is3d:
        nstep = coords.shape[time_axis]
        sl = [slice(None)]*3
    else:
        nstep = 1
        sl = [slice(None)]*2
    axsf_str = "ANIMSTEPS %i\nCRYSTAL\nPRIMVEC\n%s" %(nstep, common.str_arr(cp))
    for istep in range(nstep):
        if is3d:
            sl[time_axis] = istep
        coords_cart = np.dot(coords[sl], cp)
        arr = np.concatenate((coords_cart, np.zeros_like(coords_cart)), axis=1)
        axsf_str += "\nPRIMCOORD %i\n%i 1\n%s" %(istep+1, natoms, 
                                                 atpos_str(symbols, arr))
    common.file_write(filename, axsf_str)

#-----------------------------------------------------------------------------
# atomic coords processing / evaluation, MD analysis
#-----------------------------------------------------------------------------


def rms(arr, nitems='all'):
    """RMS of all elements in a ndarray.
    
    args:
    -----
    arr : ndarray
    nitems : {'all', float)
        normalization constant, the sum of squares is divided by this number,
        set to unity for no normalization, if 'all' then use nitems = number of
        elements in the array

    returns:
    --------
    rms : scalar
    """
    if nitems == 'all':
        nitems = float(arr.nbytes / arr.itemsize)
    else:
        nitems = float(nitems)
    rms = np.sqrt((arr**2.0).sum() / nitems)
    return rms        


def rms3d(arr, axis=-1, nitems='all'):
    """RMS of 3d array along `axis`. Sum all elements of all axes != axis.
    
    args:
    -----
    arr : 3d array
    axis : int
        The axis along which the RMS of all sub-arrays is to be computed.
    nitems : {'all', float)
        normalization constant, the sum of squares is divided by this number,
        set to unity for no normalization, if 'all' then use nitems = number of
        elements in each sub-array along `axis`
    
    returns:
    --------
    rms : 1d array, (arr.shape[axis],)
    """
    assert -1 <= axis <= 2, "allowed axis values: -1,0,1,2"
    assert arr.ndim == 3, "arr must be 3d array"
    if axis == -1:
        axis = arr.ndim - 1
    if nitems == 'all':
        sl = [slice(None)]*arr.ndim
        sl[axis] = 0 # pick out 1st sub-array along axis
        nitems = float(arr[sl].nbytes / arr.itemsize)
    else:
        nitems = float(nitems)
    if axis == 0:
        rms =  np.sqrt((arr**2.0).sum(1).sum(1) / nitems)
    elif axis == 1:
        rms =  np.sqrt((arr**2.0).sum(0).sum(1) / nitems)
    elif axis == 2:
        rms =  np.sqrt((arr**2.0).sum(0).sum(0) / nitems)
    return rms        
    

def rmsd(coords_cart, ref_idx=0, axis=-1):
    """Root mean square distance of an MD trajectory of a whole atomic
    structure. The normalization constant is the number of atoms
    (coords_cart.shape[0]).
    
    args:
    -----
    coords_cart : 3d array 
        atom coords, time axis `axis`, natoms axis must be 0
    ref_idx : time index of the reference structure (i.e. 0 to compare with the
        start structure).
    axis : int
        time axis in `coords_cart`
    
    returns:
    --------
    rmsd : 1d array (coords_cart.shape[axis],)

    examples:
    ---------
    # The RMSD w.r.t. the start structure.
    >>> corrds=rand(10,3,500)
    >>> rmsd(coords, ref_idx=0, axis=-1)
    # For a relaxation run, the RMSD w.r.t. the final converged structure. The
    # RMSD should converge to zero here.
    >>> rmsd(coords, ref_idx=-1, axis=-1)
    """
    # sl_ref : pull out 2d array of coords of the reference structure
    # sl_newaxis : slice to broadcast (newaxis) this 2d array to 3d for easy
    #     substraction
    assert coords_cart.ndim == 3
    ndim = 3
    R = coords_cart.copy()
    sl_ref = [slice(None)]*ndim
    sl_ref[axis] = ref_idx
    sl_newaxis = [slice(None)]*ndim
    sl_newaxis[axis] = None
    ref = R[sl_ref].copy()
    R -= ref[sl_newaxis]
    N = float(R.shape[0])
    return rms3d(R, axis=axis, nitems=N)


#FIXME implement axis kwarg, get rid of loops
##def max_displacement(coords_cart):
##    R = coords_cart
##    md = np.empty((R.shape[0], R.shape[2]), dtype=float)
##    # iatom
##    for i in range(R.shape[0]):
##        # x,y,z
##        for j in range(R.shape[2]):
##            md[i,j] = R[i,:,j].max() - R[i,:,j].min()
##    return md            


def pbc_wrap(coords, copy=True, mask=[True]*3, xyz_axis=-1):
    """Apply periodic boundary conditions. Wrap atoms with fractional coords >
    1 or < 0 into the cell.
    
    args:
    -----
    coords : array 2d or 3d
        fractional coords, if 3d then one axis is assumed to be a time axis and
        the array is a MD trajectory or such
    copy : bool
        Copy coords before applying pbc.     
    mask : sequence of bools, len = 3 for x,y,z
        Apply pbc only x, y or z. E.g. [True, True, False] would not wrap the z
        coordinate.
    xyz_axis : the axis of `coords` where the indices 0,1,2 pull out the x,y,z
        coords. For a usual 2d array of coords with shape (natoms,3), 
        xyz_axis=1 (= last axis = -1). For a 3d array (natoms, nstep, 3),
        xyz_axis=2 (also -1).
    
    returns:
    --------
    coords with all values in [0,1] except for those where mask[i] = False.

    notes:
    ------
    About the copy arg: If copy=False, then this is an in-place operation and
    the array in the global scope is modified! In fact, then these do the same:
    >>> a = pbc_wrap(a, copy=False)
    >>> pbc_wrap(a, copy=False)
    """
    assert coords.shape[xyz_axis] == 3, "dim of xyz_axis of `coords` must be == 3"
    ndim = coords.ndim
    assert ndim in [2,3], "coords must be 2d or 3d array"
    tmp = coords.copy() if copy else coords
    for i in range(3):
        if mask[i]:
            sl = [slice(None)]*ndim
            sl[xyz_axis] = i
            tmp[sl] %= 1.0
    return tmp        


def coord_trans(coords, old=None, new=None, copy=True, align='cols'):
    """Coordinate transformation.
    
    args:
    -----
    coords : array (d0, d1, ..., M) 
        Array of arbitrary rank with coordinates (length M vectors) in old
        coord sys `old`. The only shape resiriction is that the last dim must
        equal the number of coordinates (coords.shape[-1] == M == 3 for normal
        3-dim x,y,z). 
            1d : trivial, transform that vector (length M)
            2d : The matrix must have shape (N,M), i.e. the vectors to be
                transformed are the *rows*.
            3d : coords must have the shape (..., M)                 
    old, new : 2d arrays
        matrices with the old and new basis vectors as rows or cols
    copy : bool, optional
        True: overwrite `coords`
        False: return new array
    align : string
        {'cols', 'rows'}
        cols : basis vecs are columns of `old` and `new`
        rows : basis vecs are rows    of `old` and `new`

    returns:
    --------
    array of shape = coords.shape, coordinates in system `new`
    
    examples:
    ---------
    # Taken from [1].
    >>> import numpy as np
    >>> import math
    >>> v = np.array([1.0,1.5])
    >>> I = np.identity(2)
    >>> X = math.sqrt(2)/2.0*np.array([[1,-1],[1,1]])
    >>> Y = np.array([[1,1],[0,1]])
    >>> coord_trans(v,I,I)
    array([ 1. ,  1.5])
    >>> v_X = coord_trans(v,I,X)
    >>> v_Y = coord_trans(v,I,Y)
    >>> v_X
    array([ 1.76776695,  0.35355339])
    >>> v_Y
    array([-0.5,  1.5])
    >>> coord_trans(v_Y,Y,I)
    array([ 1. ,  1.5])
    >>> coord_trans(v_X,X,I)
    array([ 1. ,  1.5])
    
    >>> coordsold = np.random.rand(30,200,3)
    >>> old = np.random.rand(3,3)
    >>> new = np.random.rand(3,3)
    >>> coordsnew = coord_trans(coordsold, old=old, new=new)
    >>> coordsold2 = coord_trans(coordsnew, old=new, new=old)
    >>> np.testing.assert_almost_equal(coordsold, coordsold2)
    
    # these do the same: A, B have vecs as rows
    >>> coordsB1=coord_trans(coordsold, old=old, new=new, align='rows') 
    >>> coordsB2=coord_trans(coordsold, old=old.T, new=new.T, align='cols') 
    >>> np.testing.assert_almost_equal(coordsold, coordsold2)
    
    # If you have an array of shape, say (10,3,100), i.e. the last dimension is
    # NOT 3, then use numpy.swapaxes():
    >>> coord_trans(arr.swapaxes(1,2), old=..., new=...).swapaxes(1,2)

    refs:
    [1] http://www.mathe.tu-freiberg.de/~eiermann/Vorlesungen/HM/index_HM2.htm
        Kapitel 6
    """ 
    # Coordinate transformation:
    # --------------------------
    #     
    # From the textbook:
    # X, Y square matrices with basis vecs as *columns*.
    #
    # X ... old, shape: (3,3)
    # Y ... new, shape: (3,3)
    # I ... identity matrix, basis vecs of cartesian system, shape: (3,3)
    # A ... transformation matrix, shape(3,3)
    # v_X ... column vector v in basis X, shape: (3,1)
    # v_Y ... column vector v in basis Y, shape: (3,1)
    # v_I ... column vector v in basis I, shape: (3,1)
    #
    # "." denotes matrix multiplication (i.e. dot() in numpy).
    #     
    #     Y . v_Y = X . v_X = I . v_I = v_I
    #     v_Y = Y^-1 . X . v_X = A . v_X
    #
    #     v_Y^T = (A . v_X)^T = v_Y^T . A^T
    # 
    # Every product X . v_X, Y . v_Y, v_I . I (in general [basis] .
    # v_[basis]) is actually an expansion of v_{X,Y,...} in the basis vectors
    # contained in X,Y,... . If the dot product is computed, we always get v in
    # cartesian coords. 
    # 
    # Now, v_X^T is a row(!) vector (1,M). This form is implemented here (see
    # below for why). With
    #     
    #     A^T = [[--- a0 ---], 
    #            [--- a1 ---], 
    #            [--- a2 ---]] 
    # 
    # we have
    #
    #     v_Y^T = v_X^T . A^T = 
    #
    #       = v_X[0]*a0       + v_X[1]*a1       + v_X[2]*a2
    #       
    #       = v_X[0]*A.T[0,:] + v_X[1]*A.T[1,:] + v_X[2]*A.T[2,:]
    #       
    #       = [v_X[0]*A.T[0,0] + v_X[1]*A.T[1,0] + v_X[2]*A.T[2,0],
    #          v_X[0]*A.T[0,1] + v_X[1]*A.T[1,1] + v_X[2]*A.T[2,1],
    #          v_X[0]*A.T[0,2] + v_X[1]*A.T[1,2] + v_X[2]*A.T[2,2]]
    #       
    #       = dot(A, v_X)         <=> v_Y[i] = sum(j=0..2) A[i,j]*v_X[j]
    #       = dot(v_X, A.T)       <=> v_Y[j] = sum(i=0..2) v_X[i]*A[i,j]
    # 
    # numpy note: In numpy A.T is the transpose. `v` is actually an 1d array
    # for which v.T == v, i.e. the transpose is not defined and so dot(A, v_X)
    # == dot(v_X, A.T).
    #
    # In general, if we don't have one vector `v` but an array R (N,M) of row
    # vectors:
    #     
    #     R = [[--- r0 ---],
    #          [--- r1 ---],
    #          ...
    #          [-- rN-1 --]]
    #
    # it's more practical to use dot(R,A.T) instead of dot(A,R) b/c of numpy
    # array broadcasting.
    #         
    # shape of `R`:
    # -------------
    #     
    # If we want to use fast numpy array broadcasting to transform many `v`
    # vectors at once, we must use the form dot(R,A.T) or, well, transform R to
    # have the vectors as cols: dot(A,R.T)).T .
    # The shape of `R` doesn't matter, as long as the last dimension matches
    # the dimensions of A (e.g. R: shape = (n,m,3), A: (3,3), dot(R,A.T): shape
    # = (n,m,3)).
    #  
    # 1d: R.shape = (3,)
    # R == v = [x,y,z] 
    # -> dot(A, v) == dot(v,A.T) = [x', y', z']
    #
    # 2d: R.shape = (N,3)
    # Array of coords of N atoms, R[i,:] = coord of i-th atom. The dot
    # product is broadcast along the first axis of R (i.e. *each* row of R is
    # dot()'ed with A.T).
    # R = 
    # [[x0,       y0,     z0],
    #  [x1,       y1,     z1],
    #   ...
    #  [x(N-1),   y(N-1), z(N-1)]]
    # -> dot(R,A.T) = 
    # [[x0',     y0',     z0'],
    #  [x1',     y1',     z1'],
    #   ...
    #  [x(N-1)', y(N-1)', z(N-1)']]
    # 
    # 3d: R.shape = (natoms, nstep, 3) 
    # R[i,j,:] is the shape (3,) vec of coords for atom i at time step j.
    # Broadcasting along the first and second axis. 
    # These loops have the same result as newR=dot(R, A.T):
    #     # New coords in each (nstep, 3) matrix R[i,...] containing coords
    #     # of atom i for each time step. Again, each row is dot()'ed.
    #     for i in xrange(R.shape[0]):
    #         newR[i,...] = dot(R[i,...],A.T)
    #     
    #     # same as example with 2d array: R[:,j,:] is a matrix with atom
    #     # coord on each row at time step j
    #     for j in xrange(R.shape[1]):
    #             newR[:,j,:] = dot(R[:,j,:],A.T)
                 
    common.assert_cond(old.ndim == new.ndim == 2, "`old` and `new` must be rank 2 arrays")
    common.assert_cond(old.shape == new.shape, "`old` and `new` must have th same shape")
    msg = ''        
    if align == 'rows':
        old = old.T
        new = new.T
        msg = 'after transpose, '
    common.assert_cond(coords.shape[-1] == old.shape[0], 
                       "%slast dim of `coords` must match first dim"
                       " of `old` and `new`" %msg)
    if copy:
        tmp = coords.copy()
    else:
        tmp = coords
    # must use `tmp[:] = ...`, just `tmp = ...` is a new array
    tmp[:] = np.dot(tmp, np.dot(inv(new), old).T)
    return tmp


def min_image_convention(sij, copy=False):
    """Helper function for rpdf(). Apply minimum image convention to
    differences of fractional coords.
    
    args:
    -----
    sij : ndarray
    copy : bool, optional

    returns:
    --------
    sij in-place modified or copy
    """
    _sij = sij.copy() if copy else sij
    _sij[_sij > 0.5] -= 1.0
    _sij[_sij < -0.5] += 1.0
    return _sij


@crys_add_doc
def rmax_smith(cp, align='cols'):
    """Helper function for rpdf(). Calculate rmax as in [Smith].
    The cell vecs must be the rows of `cp`.

    args:
    -----
    %(cp_doc)s
    %(align_doc)s

    returns:
    --------
    rmax : float

    refs:
    -----
    [Smith] W. Smith, The Minimum Image Convention in Non-Cubic MD Cells,
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696
            1989
    """
    if align == 'cols':
        cp = cp.T
    a = cp[:,0]
    b = cp[:,1]
    c = cp[:,2]
    bxc = np.cross(b,c)
    cxa = np.cross(c,a)
    axb = np.cross(a,b)
    wa = abs(np.dot(a, bxc)) / norm(bxc)
    wb = abs(np.dot(b, cxa)) / norm(cxa)
    wc = abs(np.dot(c, axb)) / norm(axb)
    rmax = 0.5*min(wa,wb,wc)
    return rmax


@crys_add_doc
def rpdf(coords, cp, dr, rmax='auto', align='rows', pbc=True, full_output=False):
    """ Radial pair distribution (pair correlation) function. This is for one
    atomic structure, i.e. one unit cell or time step in MD. Can slao handle
    non-orthorhombic unit cells (simulation boxes).
    
    args:
    -----
    coords : 2d array (natoms, 3)
    %(cp_doc)s
    dr : radius spacing (x-axis spacing for RDF)
    rmax : {'auto', float}
        'auto': the method of [Smith] is used to calculate the max. sphere
            raduis for any cell shape
        float: set value yourself
    %(align_doc)s
    pbc : bool
        apply minimum image convention
    full_output : bool

    returns:
    --------
    (rad, hist, (number_density, number_integral, rmax_auto))
    rad : 1d array, radius (x-axis) with spacing `dr`, each value r[i] is the
        middle of a histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    if full_output:
        number_density : float
            natoms / unit cell volume
        number_integral : 1d array, (len(rad)-1,)
            number_density*hist*4*pi*r**2.0*dr
        rmax_auto : float
            auto-calculated rmax, even if not used (i.e. rmax is set from
            input)
    
    examples:
    ---------
    >>> rad, hist, dens, num_int, rmax_auto = rpdf(coords, cp, dr, full_output=True)
    >>> plot(rad, hist)
    >>> plot(rad[:-1]+0.5*np.diff(rad), num_int)
     
    refs:
    -----
    [AT] M. P. Allen, D. J. Tildesley, Computer Simulation of Liquids,
         Clarendon Press, 1989
    [MD] R. Haberlandt, S. Fritzsche, G. Peinel, K. Heinzinger,
         Molekulardynamik - Grundlagen und Anwendungen,
         Friedrich Vieweg & Sohn Verlagsgesellschaft 1995
    [Smith] W. Smith, The Minimum Image Convention in Non-Cubic MD Cells,
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696
            1989
    """
    # TODO:
    # -----
    # * Averaging over time steps (frames in VMD). ATM, you must use this
    #   function on each frame and average yourself.
    # * RDF between selection of atoms, e.g. between H  and O. For that, accept
    #   2 coords arrays. Then, sij will be (N,M,3) and we need to use np.tri()
    #   etc instead of np.triu_indices().
    #
    # Verification
    # ------------
    # 
    # This function was tested against VMD's "measure gofr" command. VMD can
    # only handle orthorhombic boxes. To test non-orthorhombic boxes:
    #   * take your non-orthorhombic unit cell and create a supercell
    #   * transform the crystal coords from your unit cell to cartesian coords
    #     and construct a supercell for that, too -> you got a orthorhombic box
    #     now
    #   * make sure that both boxes can contain a sphere of `rmax`
    #   * feed the orthorhombic box to VMD, use the SAME rmax for both boxes
    #
    # Theory
    # ------
    #
    # Below, "density" always means number density, i.e. number of particles
    # (N atoms in the unit cell)  / (unit cell) volume V.
    #
    # g(r) is (a) the average number of atoms in a shell [r,r+dr] around an
    # atom or (b) the average density of atoms in that shell -- relative to an
    # "ideal gas" (random distribution) of density N/V. Also sometimes: The
    # number of atom pairs with distance r relative the the number of pairs in
    # a random distribution.
    #
    # For each atom i=1,N, count the number dn(r) of atoms j around it in the
    # shell [r,r+dr] with rij = r_i - r_j.
    #   
    #   dn(r) = sum(i=1,N) sum(j=1,N, j!=i) delta(r - rij)
    # 
    # We sum over N atoms, so we have to divide by N -- that's why g(r) is an
    # average. Also, we normalize to ideal gas values
    #   
    #   g(r) = dn(r) / (N * V(r) * rho)
    #   V(r) = 4*pi*r**2*dr = 4/3*pi*[(r+dr)**3 - r**3]
    #   rho = N/V
    # 
    # where V(r) the volume of the shell. Formulation (a) from above: rho*V(r)
    # is the number of atoms in the shell for an ideal gas or (b):  dn(r) /
    # V(r) is the density of atoms in the shell and dn(r) / (V(r) * rho) is
    # that density relative to the ideal gas density rho. Clear? :)
    # 
    # g(r) -> 1 for r -> inf in liquids, i.e. long distances are not
    # correlated. Their distribution is random. In a crystal, we get an
    # infinite series of delta peaks at the distances of the 1st, 2nd, ...
    # nearest neighbor shell.
    #
    # The number integral is
    #
    #   I(r1,r2) = int(r=r1,r2) rho*g(r)*4*pi*r**2*dr
    # 
    # which can be used to calculate coordination numbers, i.e. it counts the
    # average number of atoms around an atom in a shell [r1,r2]. 
    #   
    # Integrating to infinity
    #
    #   I(0,inf) = N-1
    #
    # gives the average number of *all* atoms around an atom, *excluding* the
    # central one. This integral will converge to N-1 with or without PBC, but
    # w/o PBC, the nearest neigbor numbers I(r1,r2) will be wrong! Always use
    # PBC (minimum image convention). Have a look at the following table:
    #
    #                                nearest neighb.     I(0,rmax) = N-1  
    # pbc=Tue,   rmax <  rmax-auto   +                   -
    # pbc=Tue,   rmax >> rmax-auto   +                   +
    # pbc=False, rmax <  rmax-auto   -                   -
    # pbc=Nalse, rmax >> rmax-auto   -                   +
    # 
    # Interestingly though, in the 2nd case the nearest neighbors are also
    # correct (pbc=True, rmax > rmax-auto). Why? A too big rmax should be
    # inconsistent with the MIC. These results were obtained with a
    # rocksalt-AlN crystal test case. However, [Smith] says: Use only the 1st
    # case: pbc=True, rmax < rmax-auto.
    #
    # For a crystal, integrating over a peak [r-dr/2, r+dr/2] gives *exactly*
    # the number of nearest neighbor atoms for that distance r b/c the
    # normalizatiobn factor -- the number of atoms in an ideal gas for a narrow
    # shell of width dr -- is 1.
    #
    assert coords.shape[1] == 3
    assert cp.shape == (3,3)
    if align == 'cols':
        cp = cp.T
    rmax_auto = rmax_smith(cp)
    if rmax == 'auto':
        rmax = rmax_auto
    natoms = coords.shape[0]
    volume = np.linalg.det(cp) # or crys.volume_cp()
    number_density = float(natoms) / volume
    # sij and rij
    #
    # sij: (N,N,3); sij is a "(N,N)-matrix" is length=3 distance vectors
    #   equivalent 2d:  
    #   >>> a=arange(5)
    #   >>> sij = a[:,None]-a
    #   or
    #   >>> sij = a[:,None]-a[None,:]
    sij = coords[:,None,:] - coords[None,...]
    # np.tri() etc. for (N,M,3) and coords1[xxx]-coords2[yyy] for 2 selections
    idx = np.triu_indices(natoms, 1)
    # ((N**2 - N)/2, 3) 
    sij = sij[idx[0], idx[1],:]
    if pbc:
        sij = min_image_convention(sij)
    rij = np.dot(sij, cp) 
    # ((N**2 - N)/2,) 
    dists = np.sqrt((rij**2.0).sum(axis=1))
    dists = dists[dists < rmax]
    bins = np.arange(0, rmax+dr, dr)
    hist, rad_hist = np.histogram(dists, bins=bins)
    rad = rad_hist[:-1]+0.5*dr
    volume_shells = 4.0/3.0*pi*(rad_hist[1:]**3.0 - rad_hist[:-1]**3.0)
    hist = hist * 2.0 / volume_shells * volume / (natoms)**2.0
    out = (rad, hist)
    if full_output:
        number_integral = cumtrapz(number_density*hist*4*pi*rad**2.0, rad)
        out += (number_density, number_integral, rmax_auto) 
    return out
