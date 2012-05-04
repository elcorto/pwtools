# crys.py
#
# Crystal and unit-cell related tools, MD analysis, container classes.
#

from math import acos, pi, sin, cos, sqrt
import textwrap
import time
import os
import tempfile

import numpy as np
from scipy.linalg import inv
from scipy.integrate import cumtrapz

from common import assert_cond
import common
from decorators import crys_add_doc
import num, periodic_table
from base import FlexibleGetters
from verbose import verbose
from constants import Bohr, Angstrom
import constants

#-----------------------------------------------------------------------------
# misc math
#-----------------------------------------------------------------------------

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
def volume_cell(cell):
    """Volume of the unit cell from cell vectors. Calculates the triple
    product 
        np.dot(np.cross(a,b), c) == det(cell)
    of the basis vectors a,b,c contained in `cell`. Note that (mathematically)
    the vectors can be either the rows or the cols of `cell`.

    args:
    -----
    %(cell_doc)s

    returns:
    --------
    volume, unit: [a]**3

    example:
    --------
    >>> a = [1,0,0]; b = [2,3,0]; c = [1,2,3.];
    >>> m = np.array([a,b,c])
    >>> volume_cell(m)
    9.0
    >>> volume_cell(m.T)
    9.0
    >>> m = rand(3,3)
    >>> volume_cell(m)
    0.11844733769775126
    >>> volume_cell(m.T)
    0.11844733769775123
    >>> np.linalg.det(m)
    0.11844733769775125
    >>> np.linalg.det(m.T)
    0.11844733769775125

    notes:
    ------
    %(notes_cell_crys_const)s
    """    
    assert_cond(cell.shape == (3,3), "input must be (3,3) array")
##    return np.dot(np.cross(cell[0,:], cell[1,:]), cell[2,:])
    return abs(np.linalg.det(cell))

def volume_cell3d(cell, axis=0):
    """Same as volume_cell() for 3d arrays.
    
    args:
    -----
    cell : 3d array
    axis : time axis (e.g. cell.shape = (100,3,3) -> axis=0)
    """
    assert cell.ndim == 3
    sl = [slice(None)]*cell.ndim
    ret = []
    for ii in range(cell.shape[axis]):
        sl[axis] = ii
        ret.append(volume_cell(cell[sl]))
    return np.array(ret)        

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
    %(notes_cell_crys_const)s

    refs:
    -----
    [1] http://en.wikipedia.org/wiki/Parallelepiped
    """
    assert cryst_const.shape == (6,), "shape must be (6,)"
    a = cryst_const[0]
    b = cryst_const[1]
    c = cryst_const[2]
    alpha = cryst_const[3]*pi/180
    beta = cryst_const[4]*pi/180
    gamma = cryst_const[5]*pi/180
    return a*b*c*sqrt(1+ 2*cos(alpha)*cos(beta)*cos(gamma) -\
          cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 )


def volume_cc3d(cryst_const, axis=0):
    """Same as volume_cc() for 2d arrays (the name "*3d" is just to indicate
    that we work w/ trajectories).
    
    args:
    -----
    cryst_const : 2d array
    axis : time axis (e.g. cryst_const.shape = (100,6) -> axis=0)
    """
    assert cryst_const.ndim == 2
    sl = [slice(None)]*cryst_const.ndim
    ret = []
    for ii in range(cryst_const.shape[axis]):
        sl[axis] = ii
        ret.append(volume_cc(cryst_const[sl]))
    return np.array(ret)        

@crys_add_doc
def cell2cc(cell):
    """From ``cell`` to crystallographic constants a, b, c, alpha, beta,
    gamma. 
    This mapping is unique in the sense that multiple `cell`s will have
    the same `cryst_const`, i.e. the representation of the cell in
    `cryst_const` is independent from the spacial orientation of the cell
    w.r.t. a cartesian coord sys.
    
    args:
    -----
    %(cell_doc)s

    returns:
    --------
    %(cryst_const_doc)s, 
        unit: [a]**3

    notes:
    ------
    %(notes_cell_crys_const)s
    """
    cell = np.asarray(cell)
    assert_cond(cell.shape == (3,3), "cell must be (3,3) array")
    cryst_const = np.empty((6,), dtype=float)
    # a = |a|, b = |b|, c = |c|
    cryst_const[:3] = np.sqrt((cell**2.0).sum(axis=1))
    va = cell[0,:]
    vb = cell[1,:]
    vc = cell[2,:]
    # alpha
    cryst_const[3] = angle(vb,vc)
    # beta
    cryst_const[4] = angle(va,vc)
    # gamma
    cryst_const[5] = angle(va,vb)
    return cryst_const

def cell2cc3d(cell, axis=0):
    """Same as cell2cc() for 3d arrays.
    
    args:
    -----
    cell : 3d array
    axis : time axis (e.g. cell.shape = (100,3,3) -> axis=0)
    """
    assert cell.ndim == 3
    sl = [slice(None)]*cell.ndim
    ret = []
    for ii in range(cell.shape[axis]):
        sl[axis] = ii
        ret.append(cell2cc(cell[sl]))
    return np.array(ret)        

@crys_add_doc
def cc2cell(cryst_const):
    """From crystallographic constants a, b, c, alpha, beta,
    gamma to CELL_PARAMETERS.
    This mapping not NOT unique in the sense that one set of `cryst_const` can
    have arbitrarily many representations in terms of `cell`s. We stick to a
    common convention. See notes below.
    
    args:
    -----
    %(cryst_const_doc)s
    
    returns:
    --------
    %(cell_doc)s
        unit: [a]**3
    
    notes:
    ------
    * %(notes_cell_crys_const)s
    
    * Basis vectors fulfilling the crystallographic constants are arbitrary
      w.r.t. their orientation in space. We choose the common convention that
        va : along x axis
        vb : in the x-y plane
      Then, vc is fixed. 
      cell = [[-- va --],
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

def cc2cell3d(cryst_const, axis=0):
    """Same as cc2cell() for 2d arrays (the name "*3d" is just to indicate
    that we work w/ trajectories).
    
    args:
    -----
    cryst_const : 2d array
    axis : time axis (e.g. cryst_const.shape = (100,6) -> axis=0)
    """
    assert cryst_const.ndim == 2
    sl = [slice(None)]*cryst_const.ndim
    ret = []
    for ii in range(cryst_const.shape[axis]):
        sl[axis] = ii
        ret.append(cc2cell(cryst_const[sl]))
    return np.array(ret)        

@crys_add_doc
def recip_cell(cell):
    """Reciprocal lattice vectors.
        {a,b,c}* = 2*pi / V * {b,c,a} x {c, a, b}
    
    The volume is calculated using `cell`, so make sure that all units match.

    args:
    -----
    %(cell_doc)s

    returns:
    --------
    Shape (3,3) numpy array with reciprocal vectors as rows.

    notes:
    ------
    %(notes_cell_crys_const)s

    The unit of the recip. vecs is 1/[cell] and the unit of the volume is
    [cell]**3.
    """
    cell = np.asarray(cell)
    assert_cond(cell.shape == (3,3), "cell must be (3,3) array")
    cell_recip = np.empty_like(cell)
    vol = volume_cell(cell)
    a = cell[0,:]
    b = cell[1,:]
    c = cell[2,:]
    cell_recip[0,:] = 2*pi/vol * np.cross(b,c)
    cell_recip[1,:] = 2*pi/vol * np.cross(c,a)
    cell_recip[2,:] = 2*pi/vol * np.cross(a,b)
    return cell_recip

@crys_add_doc
def cc2celldm(cryst_const, fac=1.0):
    """
    Convert cryst_const to PWscf `celldm`.

    args:
    -----
    %(cryst_const_doc)s
    fac : float, optional
        conversion a[any unit] -> a[Bohr]
    
    returns:
    --------
    %(celldm)s
    """
    assert len(cryst_const) == 6, ("cryst_const has length != 6")
    celldm = np.empty((6,), dtype=np.float)
    a,b,c,alpha,beta,gamma = np.asarray(cryst_const, dtype=np.float)
    celldm[0] = a*fac
    celldm[1] = b/a
    celldm[2] = c/a
    celldm[3] = cos(alpha*pi/180.0)
    celldm[4] = cos(beta*pi/180.0)
    celldm[5] = cos(gamma*pi/180.0)
    return celldm

@crys_add_doc
def celldm2cc(celldm, fac=1.0):
    """Convert PWscf celldm to cryst_const.
    
    args:
    -----
    %(celldm)s
    fac : float, optional
        conversion a[Bohr] -> a[any unit]
    """
    assert len(celldm) == 6, ("celldm has length != 6")
    cryst_const = np.empty((6,), dtype=np.float)
    a,ba,ca,cos_alpha,cos_beta,cos_gamma = np.asarray(celldm, dtype=np.float)
    a = a*fac
    cryst_const[0] = a
    cryst_const[1] = ba * a
    cryst_const[2] = ca * a
    cryst_const[3] = acos(cos_alpha) / pi * 180.0
    cryst_const[4] = acos(cos_beta) / pi * 180.0
    cryst_const[5] = acos(cos_gamma) / pi * 180.0
    return cryst_const

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
    >>> # 2x2x1 slab = "plane" of 4 cells           
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


def scell(struct, dims, method=1):
    """Build supercell based on `dims`. It scales the unit cell to the dims of
    the super cell and returns crystal atomic positions w.r.t. this cell.
    
    args:
    -----
    struct : Structure
    dims : tuple (nx, ny, nz) for a N = nx * ny * nz supercell
    method : int, optional
        Switch between numpy-ish (1) or loop (2) implementation. (2) should
        always produce correct results but is sublty slower.

    returns:
    --------
    Structure
    """
    assert_cond(struct.cell.shape == (3,3), "cell must be (3,3) array")
    mask = scell_mask(*tuple(dims))
    # Place each atom N = dim1*dim2*dim3 times in the
    # supercell, i.e. copy unit cell N times. Actually, N-1, since
    # n1=n2=n3=0 is the unit cell itself.
    #
    # mask[j,:] = [n1, n2, n3], ni = integers (floats actually, but
    #   mod(ni, floor(ni)) == 0.0)
    #
    # original cell:
    # coords[i,:] = position vect of atom i in the unit cell in *crystal*
    #   coords!!
    # 
    # super cell:
    # sc_coords[i,:] = coords[i,:] + [n1, n2, n3]
    #   for all permutations (see scell_mask()) of n1, n2, n3.
    #   ni = 0, ..., dim_i - 1, i = 1,2,3
    #
    # sc_coords : crystal coords w.r.t the *old* cell, i.e. the entries are in
    # [0,(max(dims))], not [0,1], is scaled below
    #
    nmask = mask.shape[0]
    if method == 1:   
        sc_symbols = np.array(struct.symbols).repeat(nmask).tolist() if (struct.symbols \
                     is not None) else None
        # (natoms, 1, 3) + (1, nmask, 3) -> (natoms, nmask, 3)
        sc_coords_frac = (struct.coords_frac[:,None] \
                          + mask[None,:]).reshape(struct.natoms*nmask,3)
    elif method == 2:        
        sc_symbols = []
        sc_coords_frac = np.empty((nmask*struct.natoms, 3), dtype=float)
        k = 0
        for iatom in range(struct.natoms):
            for j in range(nmask):
                if struct.symbols is not None:
                    sc_symbols.append(struct.symbols[iatom])  
                sc_coords_frac[k,:] = struct.coords_frac[iatom,:] + mask[j,:]
                k += 1
    else:
        raise StandardError("unknown method: %s" %repr(method))
    # scale cell acording to super cell dims
    sc_cell = struct.cell * np.asarray(dims)[:,None]
    # Rescale crystal coords_frac to new bigger cell (coord_trans
    # actually) -> all values in [0,1] again
    sc_coords_frac[:,0] /= dims[0]
    sc_coords_frac[:,1] /= dims[1]
    sc_coords_frac[:,2] /= dims[2]
    return Structure(coords_frac=sc_coords_frac,
                     cell=sc_cell,
                     symbols=sc_symbols)

@crys_add_doc
def scell3d(traj, dims):
    """Build supercell of a trajectory (i.e. not just a single structure) based
    on `dims`. It scales the unit cell to the dims of the super cell and
    returns crystal atomic positions w.r.t. this cell.

    This is a special-case version of scell() for trajectories, where at least
    ``traj.coords_frac`` must be a 3d array.
    
    args:
    -----
    traj : Trajectory
    dims : tuple (nx, ny, nz) for a N = nx * ny * nz supercell

    returns:
    --------
    Trajectory
    """
    mask = scell_mask(*tuple(dims))
    nmask = mask.shape[0]
    sc_symbols = np.array(traj.symbols).repeat(nmask).tolist() if (traj.symbols \
                 is not None) else None
    # cool, eh?
    # (nstep, natoms, 1, 3) + (1, 1, nmask, 3) -> (nstep, natoms, nmask, 3)
    sc_coords_frac = (traj.coords_frac[...,None,:] \
                      + mask[None,None,...]).reshape(traj.nstep,traj.natoms*nmask,3)
    # (nstep,3,3) * (1,3,1) -> (nstep, 3,3)                      
    sc_cell = traj.cell * np.asarray(dims)[None,:,None]
    sc_coords_frac[:,:,0] /= dims[0]
    sc_coords_frac[:,:,1] /= dims[1]
    sc_coords_frac[:,:,2] /= dims[2]
    return Trajectory(coords_frac=sc_coords_frac,
                      cell=sc_cell,
                      symbols=sc_symbols)

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


def rms3d(arr, axis=0, nitems='all'):
    """RMS of 3d array along `axis`. Sum all elements of all axes != axis.
    
    args:
    -----
    arr : 3d array
    axis : int
        The axis along which the RMS of all sub-arrays is to be computed
        (usually time axis in MD).
    nitems : {'all', float)
        normalization constant, the sum of squares is divided by this number,
        set to unity for no normalization, if 'all' then use nitems = number of
        elements in each sub-array along `axis`
    
    returns:
    --------
    rms : 1d array, (arr.shape[axis],)
    """
    # We could use num.sum() and would be able to generalize to nd arrays. But
    # not needed now.
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
    

def rmsd(traj, ref_idx=0):
    """Root mean square distance over an MD trajectory. The normalization
    constant is the number of atoms.
    
    args:
    -----
    traj : Trajectory object
    ref_idx : time index of the reference structure (i.e. 0 to compare with the
        start structure, -1 for the last along `axis`).
    
    returns:
    --------
    rmsd : 1d array (traj.nstep,)

    examples:
    ---------
    # We only need traj.{coords,nstep,timeaxis}, no symbols, cell, ...
    >>> traj = crys.Trajectory(coords=rand(500,10,3))
    # The RMSD w.r.t. the start structure. See when the structure starts to
    # "converge" to a stable mean configuration during an MD.
    >>> rmsd(traj, ref_idx=0)
    # For a relaxation run, the RMSD w.r.t. the final converged structure. The
    # RMSD should converge to zero here.
    >>> rmsd(traj, ref_idx=-1)
    """
    # sl_ref : pull out 2d array of coords of the reference structure
    # sl_newaxis : slice to broadcast (newaxis) this 2d array to 3d for easy
    #     substraction
    assert traj.coords.ndim == 3
    ndim = 3
    coords = traj.coords.copy()
    sl_ref = [slice(None)]*ndim
    sl_ref[traj.timeaxis] = ref_idx
    sl_newaxis = [slice(None)]*ndim
    sl_newaxis[traj.timeaxis] = None
    ref = coords[sl_ref].copy()
    coords -= ref[sl_newaxis]
    return rms3d(coords, axis=traj.timeaxis, nitems=float(traj.natoms))


def pbc_wrap(coords_frac, copy=True, mask=[True]*3, xyz_axis=-1):
    """Apply periodic boundary conditions. Wrap atoms with fractional coords >
    1 or < 0 into the cell.
    
    args:
    -----
    coords_frac : array 2d or 3d
        fractional coords, if 3d then one axis is assumed to be a time axis and
        the array is a MD trajectory or such
    copy : bool
        Copy coords_frac before applying pbc.     
    mask : sequence of bools, len = 3 for x,y,z
        Apply pbc only x, y or z. E.g. [True, True, False] would not wrap the z
        coordinate.
    xyz_axis : the axis of `coords_frac` where the indices 0,1,2 pull out the x,y,z
        coords. For a usual 2d array of coords with shape (natoms,3), 
        xyz_axis=1 (= last axis = -1). For a 3d array (natoms, nstep, 3),
        xyz_axis=2 (also -1).
    
    returns:
    --------
    coords_frac with all values in [0,1] except for those where mask[i] = False.

    notes:
    ------
    About the copy arg: If copy=False, then this is an in-place operation and
    the array in the global scope is modified! In fact, then these do the same:
    >>> a = pbc_wrap(a, copy=False)
    >>> pbc_wrap(a, copy=False)
    """
    assert coords_frac.shape[xyz_axis] == 3, "dim of xyz_axis of `coords_frac` must be == 3"
    ndim = coords_frac.ndim
    assert ndim in [2,3], "coords_frac must be 2d or 3d array"
    tmp = coords_frac.copy() if copy else coords_frac
    for i in range(3):
        if mask[i]:
            sl = [slice(None)]*ndim
            sl[xyz_axis] = i
            tmp[sl] %= 1.0
    return tmp        


def coord_trans(coords, old=None, new=None, copy=True, axis=-1):
    """General-purpose n-dimensional coordinate transformation. `coords` can
    have arbitrary dimension, i.e. it can contain many vectors to be
    transformed at once. But `old` and `new` must have ndim=2, i.e. only one
    old and new coord sys for all vectors in `coords`. 
    
    The most general case is that you want to transform an MD trajectory from a
    variable cell run, you have smth like this:
        coords.shape = (natoms, 3, nstep)
        old.shape/new.shape = (3,3,nstep)
    You have a set of old and new coordinate systems at each step. 
    Then, use a loop over all time steps and call this function nstep times.

    args:
    -----
    coords : array (d0, d1, ..., M) 
        Array of arbitrary rank with coordinates (length M vectors) in old
        coord sys `old`. The only shape resiriction is that the last dim must
        equal the number of coordinates (coords.shape[-1] == M == 3 for normal
        3-dim x,y,z). 
            1d : trivial, transform that vector (length M)
            2d : The matrix must have shape (N,M), i.e. N vectors to be
                transformed are the *rows*.
            3d : coords must have shape (..., M)
        If `coords` has a different shape, use `axis` to define the M-axis.
    old, new : 2d arrays (M,M)
        Matrices with the old and new basis vectors as *rows*. Note that in the
        usual math literature, columns are used. In that case, use ``old.T`` and/or
        ``new.T``.
    copy : bool, optional
        True: overwrite `coords`
        False: return new array
    axis : the axis along which the length-M vectors are placed in `coords`,
        default is -1, i.e. coords.shape = (...,M)

    returns:
    --------
    array of shape = coords.shape, coordinates in system `new`
    
    examples:
    ---------
    # Taken from [1].
    >>> import numpy as np
    >>> import math
    >>> v_I = np.array([1.0,1.5])
    >>> I = np.identity(2)
    >>> X = math.sqrt(2)/2.0*np.array([[1,-1],[1,1]]).T
    >>> Y = np.array([[1,1],[0,1]]).T
    >>> coord_trans(v_I,I,I)
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
    
    >>> c_old = np.random.rand(30,200,3)
    >>> old = np.random.rand(3,3)
    >>> new = np.random.rand(3,3)
    >>> c_new = coord_trans(c_old, old=old, new=new)
    >>> c_old2 = coord_trans(c_new, old=new, new=old)
    >>> np.testing.assert_almost_equal(c_old, c_old2)
    
    # If you have an array of shape, say (10,3,100), i.e. the last dimension is
    # NOT 3, then use numpy.swapaxes() or axis:
    >>> coord_trans(arr, old=..., new=..., axis=1)
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
    # (A . B)^T = B^T . A^T, so for *row* vectors v^T, we have
    #     
    #     v_Y^T = (A . v_X)^T = v_X^T . A^T
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
    #                   | [[--- a0 ---], 
    #                   |  [--- a1 ---], 
    #                   |  [--- a2 ---]]
    #     ------------------------
    #     [--- v_X ---] |  [--- v_Y---]
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
    # In general, we don't have one vector v_X but an array R_X (N,M) of row
    # vectors:
    #     
    #     R_X = [[--- v_X0 ---],
    #            [--- v_X1 ---],
    #            ...
    #            [-- v_XN-1 --]]
    #
    # We want to use fast numpy array broadcasting to transform the `v_X`
    # vectors at once and therefore must use the form dot(R_X,A.T) or, well,
    # transform R_X to have the vectors as cols: dot(A,R_X.T)).T . The shape of
    # `R_X` doesn't matter, as long as the last dimension matches the
    # dimensions of A (e.g. R_X: shape = (n,m,3), A: (3,3), dot(R_X,A.T): shape
    # = (n,m,3)).
    # 
    # 1d: R.shape = (3,)
    # R_X == v_X = [x,y,z] 
    # R_Y = dot(A, R_X) 
    #     = dot(R_X,A.T) 
    #     = dot(R_X, dot(inv(Y), X).T) 
    #     = linalg.solve(Y, dot(X, R_X))
    #     = [x', y', z']
    #
    # >>> X=rand(3,3); v_X=rand(3); Y=rand(3,3)
    # >>> v_Y1=dot(v_X, dot(inv(Y), X).T)
    # >>> v_Y2=linalg.solve(Y, dot(X,v_X))
    # >>> v_Y1-v_Y2
    # array([ 0.,  0.,  0.])
    #
    # 2d: R_X.shape = (N,3)
    # Array of coords of N atoms, R_X[i,:] = coord of i-th atom. The dot
    # product is broadcast along the first axis of R_X (i.e. *each* row of R_X is
    # dot()'ed with A.T).
    # R_X = [[x0,       y0,     z0],
    #        [x1,       y1,     z1],
    #         ...
    #        [x(N-1),   y(N-1), z(N-1)]]
    # R_Y = dot(R,A.T) = 
    #       [[x0',     y0',     z0'],
    #        [x1',     y1',     z1'],
    #         ...
    #        [x(N-1)', y(N-1)', z(N-1)']]
    #
    # >>> X=rand(3,3); v_X=rand(5,3); Y=rand(3,3)
    # >>> v_Y1=dot(v_X, dot(inv(Y), X).T) 
    # >>> v_Y2=linalg.solve(Y, dot(v_X,X.T).T).T
    # >>> v_Y1-v_Y2
    # array([[ -3.05311332e-16,   2.22044605e-16,   4.44089210e-16],
    #        [  4.44089210e-16,   1.11022302e-16,  -1.33226763e-15],
    #        [ -4.44089210e-16,   0.00000000e+00,   1.77635684e-15],
    #        [  2.22044605e-16,   2.22044605e-16,   0.00000000e+00],
    #        [ -2.22044605e-16,   0.00000000e+00,   0.00000000e+00]])
    # Here we used the fact that linalg.solve can solve for many rhs's at the
    # same time (Ax=b, A:(M,M), b:(M,N) where the rhs's are the columns of b).
    #
    # 3d: R_X.shape = (natoms, nstep, 3) 
    # R_X[i,j,:] is the shape (3,) vec of coords for atom i at time step j.
    # Broadcasting along the first and second axis. 
    # These loops have the same result as R_Y=dot(R_X, A.T):
    #
    #     # New coords in each (nstep, 3) matrix R_X[i,...] containing coords
    #     # of atom i for each time step. Again, each row is dot()'ed.
    #     for i in xrange(R_X.shape[0]):
    #         R_Y[i,...] = dot(R_X[i,...],A.T)
    #     
    #     # same as example with 2d array: R_X[:,j,:] is a matrix with atom
    #     # coord on each row at time step j
    #     for j in xrange(R_X.shape[1]):
    #             R_Y[:,j,:] = dot(R_X[:,j,:],A.T)
    # Here, linalg.solve cannot be used b/c R_X is 3d. 
    #
    # It is said that calculating the inverse should be avoided where possible.
    # In the 1d and 2d case, there are two ways to calculate the transformation:
    #   (i)  A = dot(inv(Y), X), R_Y=dot(R_X, A.T)
    #   (ii) R_I = dot(R_X, X.T), linalg.solve(Y, R_I.T).T
    # So, one can use linalg.solve() instead. But this is not possible for the
    # nd case (R_X.ndim > 2). Maybe numpy's
    #   (i)  tensordot + (tensor)inv
    #   (ii) tensorsolve
    # But their docstrings are too cryptic for me, sorry.  
     
    common.assert_cond(old.ndim == new.ndim == 2, 
                       "`old` and `new` must be rank 2 arrays")
    common.assert_cond(old.shape == new.shape, 
                       "`old` and `new` must have th same shape")
    common.assert_cond(old.shape[0] == old.shape[1], 
                      "`old` and `new` must be square")
    # arr.T and arr.swapaxes() are no in-place operations, just views, input
    # arrays are not changed, but copy() b/c we can overwrite coords
    _coords = coords.copy() if copy else coords
    mx_axis = _coords.ndim - 1
    axis = mx_axis if (axis == -1) else axis
    # must use `coords[:] = ...`, just `coords = ...` is a new array
    if axis != mx_axis:
        # bring xyz-axis to -1 for broadcasting
        _coords[:] = _trans(_coords.swapaxes(-1, axis), 
                            old,
                            new).swapaxes(-1, axis)
    else:                            
        _coords[:] = _trans(_coords, 
                            old,
                            new)
    return _coords

def _trans(coords, old, new):
    """Helper for coord_trans()."""
    common.assert_cond(coords.shape[-1] == old.shape[0], 
                       "last dim of `coords` must match first dim"
                       " of `old` and `new`")
    # The equation works for ``old.T`` and ``new.T`` = columns.
    return np.dot(coords, np.dot(inv(new.T), old.T).T)


def coord_trans3d(coords, old=None, new=None, copy=True, axis=-1, timeaxis=0):
    """Special case version for debugging mostly. It does the loop for the
    general case where coords+old+new are 3d arrays (e.g. variable cell MD
    trajectory).
    
    This may be be slow for large ``nstep``. All other cases (``coords`` has
    arbitrary many dimensions, i.e. ndarray + old/new are fixed) are covered
    by coord_trans(). Also some special cases may be possible to solve with
    np.dot() alone if the transformation simplifes. Check your math. 

    args:
    -----
    coords : 3d array, one axis (``axis``) must have length-M vectors, another
        (``timeaxis``) must be length ``nstep``
    old,new : 2d arrays, two axes must be of equal length
    copy : see coord_trans()
    axis : axis where length-M vecs are placed if the timeaxis is removed
    timeaxis : time axis along which 2d arrays are aligned

    example:
    --------
    M = 3
    coords :  (nstep,natoms,3)
    old,new : (nstep,3,3)
    timeaxis = 0
    axis = 1 == -1 (remove timeaxis -> 2d slices (natoms,3) and (3,3) -> axis=1)
    """
    a,b,c = coords.ndim, old.ndim, new.ndim
    assert a == b == c, "ndim: coords: %i, old: %i, new: %i" %(a,b,c)
    a,b,c = coords.shape[timeaxis], old.shape[timeaxis], new.shape[timeaxis]
    assert a == b == c, "shape[timeaxis]: coords: %i, old: %i, new: %i" %(a,b,c)
    
    ndim = coords.ndim
    nstep = coords.shape[timeaxis]
    ret = []
    sl = [slice(None)]*ndim
    ret = []
    for ii in range(nstep):
        sl[timeaxis] = ii
        ret.append(coord_trans(coords[sl],
                               old=old[sl],
                               new=new[sl],
                               axis=axis,
                               copy=copy))
    ret = np.array(ret)
    if timeaxis != 0:
        return np.rollaxis(ret, 0, start=timeaxis+1)
    else:
        return ret

def min_image_convention(sij, copy=False):
    """Helper function for rpdf(). Apply minimum image convention to
    differences of fractional coords. 

    Handles also cases where coordinates are separated by an arbitrary number
    of periodic images.
    
    args:
    -----
    sij : ndarray
        Fractional coordinates.
    copy : bool, optional

    returns:
    --------
    sij in-place modified or copy
    """
    sij = sij.copy() if copy else sij
    mask = sij >= 0.5
    while mask.any():
        sij[mask] -= 1.0
        mask = sij >= 0.5
    mask = sij < -0.5
    while mask.any():
        sij[mask] += 1.0
        mask = sij < -0.5
    return sij


@crys_add_doc
def rmax_smith(cell):
    """Helper function for rpdf(). Calculate rmax as in [Smith].
    The cell vecs must be the rows of `cell`.

    args:
    -----
    %(cell_doc)s

    returns:
    --------
    rmax : float

    refs:
    -----
    [Smith] W. Smith, The Minimum Image Convention in Non-Cubic MD Cells,
            http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696
            1989
    """
    a = cell[0,:]
    b = cell[1,:]
    c = cell[2,:]
    bxc = np.cross(b,c)
    cxa = np.cross(c,a)
    axb = np.cross(a,b)
    wa = abs(np.dot(a, bxc)) / norm(bxc)
    wb = abs(np.dot(b, cxa)) / norm(cxa)
    wc = abs(np.dot(c, axb)) / norm(axb)
    rmax = 0.5*min(wa,wb,wc)
    return rmax

def rpdf(trajs, dr=None, rmax='auto', amask=None, tmask=None, 
         dmask=None, pbc=True, norm_vmd=False):
    """Radial pair distribution (pair correlation) function.
    Can also handle non-orthorhombic unit cells (simulation boxes). 
    Only fixed-cell MD.

    rmax
    ----
    The maximal `rmax` for which g(r) is correctly normalized is the
    result of rmax_smith(cell), i.e. the radius if the biggest sphere which
    fits entirely into the cell. This is simply L/2 for cubic boxes, for
    instance. We do explicitely allow rmax > rmax_smith() for testing, but be
    aware that g(r) and the number integral are *wrong* for rmax >
    rmax_smith(). 
    
    Even though the number integral will always converge to the number of all
    neighbors for r -> infinity, the integral value (the number of neigbors) is
    correct only up to rmax_smith().

    See examples/rpdf/ for educational evidence. For notes on how VMD does
    this, see comments in the code below.

    args:
    -----
    trajs : Structure or Trajectory, list of one or two Structure or
        Trajectory objects. 
        The case len(trajs)==1 is the same as providing the object directly
        (most common case). Internally we expand the input to [trajs, trajs],
        i.e. the RPDF of the 2nd coord set w.r.t. to the first is calculated --
        the order matters! This is like selection 1 and 2 in VMD, but nornmally
        you would use `amask` instead. The option to provide a list of two
        Trajectory objects exists for cases where you don't want to use
        `amask`, but create two different Trajectory objects outside.
    dr : float
        Radius spacing. Must have the same unit as `cell`, e.g. Angstrom.
    rmax : {'auto', float}, optional
        Max. radius up to which minimum image nearest neighbors are counted.
        For cubic boxes of side length L, this is L/2 [AT,MD].
        'auto': the method of [Smith] is used to calculate the max. sphere
            raduis for any cell shape
        float: set value yourself
    amask : None, list of one or two bool 1d arrays, list of one or two
        strings, optional
        Atom mask. This is the complementary functionality to `sel` in
        vmd_measure_gofr(). If len(amask)==1, then we expand to [amask, amask]
        internally, which would calculate the RPDF between the same atom
        selection. If two masks are given, then the first is applied to
        trajs[0] and the second to trajs[1]. Use this to select only certain
        atoms in each Trajectory. The default is to provide bool arrays. If you
        provide strings, they are assumed to be atom names and we create a
        bool array np.array(symbols) == amask[i].
    tmask : None or slice object, optional
        Time mask. Slice for the time axis, e.g. to use only every 100th step,
        starting from step 2000 to the end, use tmask=slice(2000,None,100),
        which is the same as np.s_[2000::100].
    dmask : None or string, optional
        Distance mask. Restrict to certain distances using numpy syntax for
        creating bool arrays:
            '>=1.0'
            '{d} >=1.0' # the same
            '({d} > 1.0) & ({d} < 3.0)'
        where '{d}' is a placeholder for the distance array (you really have to
        use '{d}'). The placeholder is optional in some pattern. This is similar
        to VMD's "within" or "pbwithin" syntax. 
    pbc : bool, optional
        apply minimum image convention to distances
    norm_vmd : bool, optional
        Normalize g(r) like in VMD by counting duplicate atoms and normalize to
        (natoms0 * natoms - duplicates) instead of (natoms0*natoms1). Affects
        all-all correlations only. num_int is not affected. Use this only for
        testing.

    returns:
    --------
    array (len(rad), 3), colums 0,1,2:
    rad : 1d array, radius (x-axis) with spacing `dr`, each value r[i] is the
        middle of a histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),), the (averaged) number integral
        number_density*hist*4*pi*r**2.0*dr
    
    notes:
    ------
    The selection mechanism with `amask` is in principle as capable as VMD's,
    but relies completely on the user's ability to create bool arrays to filter
    the atoms. In practice, anything more complicated than array(symbols)=='O'
    ("name O" in VMD) is much more difficult than VMD's powerful selection
    syntax.

    Curently, the atom distances are calculated by using numpy fancy indexing.
    That creates (big) arrays in memory. For data from long MDs, you may run
    into trouble here. For a 20000 step MD, start by using every 200th step or
    so (use ``tmask=slice(None,None,200)``) and look at the histogram, as you
    take more and more points into account (every 100th, 50th step, ...).
    Especially for Car Parrinello, where time steps are small and the structure
    doesn't change much, there is no need to use every step.
    
    examples:
    ---------
    # simple all-all RPDF
    >>> d = rpdf(traj, dr=0.1)

    # 2 selections: RPDF of all H's around all O's, average time step 3000 to
    # end, take every 50th step
    >>> traj = parse.PwMDOutputFile(...).get_traj()
    >>> d = rpdf(traj, dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50])
    >>> plot(d[:,0], d[:,1], label='g(r)')
    >>> plot(d[:,0], d[:,2], label='number integral')
    # the same as rpdf(traj,...)
    >>> d = rpdf([traj], dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50])
    >>> d = rpdf([traj, traj], dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50])
    # use bool arrays for `amask`, may need this for more complicated pattern
    >>> sy = np.array(traj.symbols)
    >>> d = rpdf(traj, dr=0.1, amask=[sy=='O', sy=='H'],tmask=np.s_[3000::50])
    # skip distances >1 Ang
    >>> d = rpdf(traj, dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50]
    ...          dmask='{d}>1.0')
     
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
    # Theory
    # ======
    # 
    # 1) N equal particles (atoms) in a volume V.
    #
    # Below, "density" always means number density, i.e. (N atoms in the unit
    # cell)  / (unit cell volume V).
    #
    # g(r) is (a) the average number of atoms in a shell [r,r+dr] around an
    # atom at r=0 or (b) the average density of atoms in that shell -- relative
    # to an "ideal gas" (random distribution) of density N/V. Also sometimes:
    # The number of atom pairs with distance r relative to the number of pairs
    # in a random distribution.
    #
    # For each atom i=1,N, count the number dn(r) of atoms j around it in the
    # shell [r,r+dr] with r_ij = r_i - r_j, r < r_ij <= r+dr
    #   
    #   dn(r) = sum(i=1,N) sum(j=1,N, j!=i) delta(r - r_ij)
    # 
    # In practice, this is done by calculating all distances r_ij and bin them
    # into a histogram dn(k) with k = r_ij / dr the histogram index.
    # 
    # We sum over N atoms, so we have to divide by N -- that's why g(r) is an
    # average. Also, we normalize to ideal gas values
    #   
    #   g(r) = dn(r) / [N * (N/V) * V(r)]
    #        = dn(r) / [N**2/V * V(r)]
    #   V(r) = 4*pi*r**2*dr = 4/3*pi*[(r+dr)**3 - r**3]
    # 
    # where V(r) the volume of the shell. Normalization to V(r) is necessary
    # b/c the shell [r, r+dr] has on average more atoms for increasing "r".
    #
    # Formulation (a) from above: (N/V) * V(r) is the number of atoms in the
    # shell for an ideal gas (density*volume) or (b):  dn(r) / V(r) is the
    # density of atoms in the shell and dn(r) / [V(r) * (N/V)] is that density
    # relative to the ideal gas density N/V. Clear? :)
    # 
    # g(r) -> 1 for r -> inf in liquids, i.e. long distances are not
    # correlated. Their distribution is random. In a crystal, we get an
    # infinite series of delta peaks at the distances of the 1st, 2nd, ...
    # nearest neighbor shell.
    #
    # The number integral is
    #
    #   I(r1,r2) = int(r=r1,r2) N/V*g(r)*4*pi*r**2*dr
    #            = int(r=r1,r2) N/V*g(r)*V(r)
    #            = int(r=r1,r2) 1/N*dn(r)
    # 
    # This can be used to calculate coordination numbers, i.e. it counts the
    # average (that's why 1/N) number of atoms around an atom in a shell
    # [r1,r2]. 
    #   
    # Integrating to infinity
    #
    #   I(0,inf) = N-1
    #
    # gives the average number of *all* atoms around an atom, *excluding* the
    # central one. This integral will converge to N-1 with or without PBC, but
    # w/o PBC, the nearest neigbor numbers I(r1,r2) will be wrong! Always use
    # PBC (minimum image convention). Have a look at the following table.
    # rmax_auto is the rmax value for the given unit cell by the method of
    # [Smith], which is L/2 for a cubic box of side length L. It is the radius
    # of the biggest sphere which still fits entirely into the cell. In the
    # table: "+" = OK, "-" = wrong.
    #
    #                                    nearest neighb.     I(0,rmax) = N-1  
    # 1.) pbc=Tue,   rmax <  rmax_auto   +                   -
    # 2.) pbc=Tue,   rmax >> rmax_auto   + (< rmax_auto)     +
    # 3.) pbc=False, rmax <  rmax_auto   -                   -
    # 4.) pbc=False, rmax >> rmax_auto   -                   +
    # 
    # (1) is the use case in [Smith]. Always use this. 
    #
    # (2) appears to be also useful. However, it can be shown that nearest
    # neigbors are correct only up to rmax_auto! See examples/rpdf/rpdf_aln.py.
    # This is because if rmax > rmax_auto (say > L/2), then the shell is empty
    # for all r outside of the box, which means that the counted number of
    # surrounding atoms will be to small.
    #
    # For a crystal, integrating over a peak [r-dr/2, r+dr/2] gives *exactly*
    # the number of nearest neighbor atoms for that distance r b/c the
    # normalization factor -- the number of atoms in an ideal gas for a narrow
    # shell of width dr -- is 1.
    #
    # 2) 2 selections
    #
    # Lets say you have 10 waters -> 10 x O (atom type A), 20 x H (type B),
    # then let A = 10, B = 20.
    #
    #   dn(r) = sum(i=1,A) sum(j=1,B) delta(r - r_ij) = 
    #           dn_AB(r) + dn_BA(r)
    # 
    # where dn_AB(r) is the number of B's around A's and vice versa. With the
    # densities A/V and B/V, we get
    #
    #   g(r) = g_AB(r) + g_BA(r) = 
    #          dn_AB(r) / [A * (B/V) * V(r)] + 
    #          dn_BA(r) / [B * (A/V) * V(r)]
    # 
    # Note that the density used is always the density of the *sourrounding*
    # atom type. g_AB(r) or g_BA(r) is the result that you want. Finally, we
    # can also write g(r) for the all-all case, i.e. 1 atom type.
    #
    #  g(r) = [dn_AB(r) +  dn_BA(r)] / [A*B/V * V(r)]
    # 
    # Note the similarity to the case of one atom type:
    #
    #  g(r) = dn(r) / [N**2/V * V(r)]
    # 
    # The integrals are:
    # 
    #  I_AB(r1,r2) = int(r=r1,r2) (B/V)*g_AB(r)*4*pi*r**2*dr
    #                int(r=r1,r2) 1/A*dn_AB(r)
    #  I_BA(r1,r2) = int(r=r1,r2) (A/V)*g_BA(r)*4*pi*r**2*dr
    #                int(r=r1,r2) 1/B*dn_BA(r)
    # 
    # Note the similarity to the one-atom case:
    #   
    #  I(r1,r2)    = int(r=r1,r2) 1/N*dn(r)
    #
    # These integrals converge to the total number of *sourrounding*
    # atoms of the other type:
    #
    #   I_AB(0,inf) = B  (not B-1 !)
    #   I_BA(0,inf) = A  (not A-1 !)
    #
    # Verification
    # ============
    # 
    # This function was tested against VMD's "measure gofr" command. VMD can
    # only handle orthorhombic boxes. To test non-orthorhombic boxes, see 
    # examples/rpdf/.
    #
    # Make sure to convert all length to Angstrom of you compare with VMD.
    #
    # Implementation details
    # ======================
    #
    # Number integral mehod
    # ---------------------
    #
    # To match with VMD results, we use the "rectangle rule", i.e. just y_i*dx.
    # This is even cheaper than the trapezoidal rule, but by far accurate
    # enough for small ``dr``. Try yourself by using a more sophisticated
    # method like 
    # >>> num_int2 = scipy.integrate.cumtrapz(hist, rad)
    # >>> plot(rad[:-1]+0.5*dr, num_int2)
    #   
    # distance calculation
    # --------------------
    # sij : "matrix" of distance vectors in crystal coords
    # rij : in cartesian coords, same unit as `cell`, e.g. Angstrom
    # 
    # sij:        (natoms0, natoms1, 3) # coords 2d
    # sij: (nstep, natoms0, natoms1, 3) # coords 3d
    # 
    # broadcasting 2d:
    #   
    #   coords0:        (natoms0, 1,       3) 
    #   coords1:        (1,       natoms1, 3)
    #   sij:            (natoms0, natoms1, 3)
    #   >>> coords0[:,None,:] - coords1[None,:,:] 
    #
    # broadcasting 3d:
    #
    #   coords0: (nstep, natoms0, 1,       3) 
    #   coords1: (nstep, 1,       natoms1, 3)
    #   sij:     (nstep, natoms0, natoms1, 3)
    #   >>> coords0[:,:,None,:] - coords1[:,None,:,:] 
    # 
    # If we have arbitrary selections, we cannot use np.tri() to select only
    # the upper (or lower) triangle of this "matrix" to skip duplicates (zero
    # distance on the main diagonal). Note that if we used tri(), we'd have to
    # multiply the histogram by two, b/c now, we always double-count ij and ji
    # distances, which seems to be correct (compare w/ VMD).
    #
    # We can easily create a MemoryError b/c of the temp arrays that numpy
    # creates. But even w/ numexpr, which avoids big temp arrays, we store the
    # result sij, which is a 4d array. For natoms=20, nstep=10000, we already
    # have a 12 GB array in RAM! The only solution is to code the section
    # Fortran/Cython/whatever in loops:
    #   * distances
    #   * apply min_image_convention() (optional)
    #   * sij -> rij transform
    #   * redcution to distances
    # 
    # Differences to VMD's measure gofr
    # =================================
    #
    # duplicates
    # ----------
    # In vmd/src/Measure.C, they count the number of identical atoms in both
    # selections (variable ``duplicates``). These atoms lead to an r=0 peak in
    # the histogram, which is bogus and must be corrected. VMD subtracts these
    # number from the first histogram bin, while we simply set it to zero and
    # don't count ``duplicates`` at all.
    #
    # normalization
    # -------------
    # For normalizing g(r) to account for growing shell volumes around the
    # central atom for increasing r, we use the textbook formulas, which lead
    # to
    #     
    #     norm_fac = volume / volume_shells / (natoms0 * natoms1)
    # 
    # while VMD uses smth similar to
    #     
    #     norm_fac = volume / volume_shells / (natoms0 * natoms1 - duplicates)
    # 
    # VMD calculates g(r) using this norm_fac, but the num_int is always
    # calculated using the textbook result
    #   
    #   I_AB(r1,r2) = int(r=r1,r2) 1/A*dn_AB(r)
    # 
    # which is what we do, i.e. just integrate the histogram. That means VMD's
    # results are inconsistent if duplicates != 0. In that case g(r) is
    # slightly wrong, but num_int is still correct. This is only the case for
    # simple all-all correlation (i.e. all atoms are considered the same),
    # duplicates = natoms0 = natoms1 = the number of zeros on the distance
    # matrix' main diagonal. Then we have a small difference in g(r), where
    # VMD's is always a little higher b/c norm_fac is smaller then it should.
    #
    # As a result, VMD's g(r) -> 1.0 for random points (= ideal gas) and
    # num_int -> N (would VMD's g(r) be integrated directly), while our g(r) ->
    # < 1.0 (e.g. 0.97) and num_int -> N-1.  
    #
    # rmax
    # ----
    # VMD has a unique feature that lets you use a higher rmax. VMD extends the
    # range of rmax over rmax_auto, up to rmax_vmd=2*sqrt(0.5)*rmax_auto (~
    # 14.14 for rmax_auto=10) which is just the length of the vector
    # [rmax_auto, rmax_auto], i.e. the radius of a sphere which touches one
    # vertice of the box. Then, we have a spherical cap, which partly covers
    # the smallest box side (remember that VMD can do orthorhombic boxes only).
    # VMD corrects the volume of the shells for normalization in that case for
    # rmax_auto < r < rmax_vmd.
    #
    # For distinct selections, our g(r) and VMD's are exactly the same up to
    # rmax_auto. After that, VMD's are correct up to rmax_vmd. At that value,
    # VMD sets g(r) and num_int to 0.0.
    #
    # The problem is: Even if g(r) is normalized correctly for rmax_auto < r <
    # rmax_vmd, the num_int in that region will be wrong b/c the integral
    # formula must be changed for that region to account for the changed
    # normalization factor, which VMD doesn't do, as far as I read the code. If
    # I'm wrong, send me an email. All in all, VMD's num_int sould be trusted
    # up to rmax_auto, just as in our case. The only advantage is a correctly
    # normalized g(r) for rmax_auto < r < rmax_vmd, which is however of little
    # use, if the num_int doesn't match.
    
    if amask is None:
        amask = [slice(None)]
    if tmask is None:
        tmask = slice(None)
    if type(trajs) != type([]):
        trajs = [trajs]
    if len(trajs) == 1:
        trajs *= 2
    if len(amask) == 1:
        amask *= 2
    trajs = map(struct2traj, trajs)
    assert len(trajs) == 2, "len(trajs) != 2"
    assert len(amask) == 2, "len(amask) != 2"
    assert trajs[0].symbols == trajs[1].symbols, ("symbols differ")
    assert trajs[0].coords_frac.ndim == trajs[1].coords_frac.ndim == 3, \
        ("coords do not both have ndim=3")
    assert trajs[0].nstep == trajs[1].nstep, ("nstep differs")        
    # this maybe slow, we need a better and faster test to ensure fixed
    # cell        
    assert (trajs[0].cell == trajs[1].cell).all(), ("cells are not the same")
    assert np.abs(trajs[0].cell - trajs[0].cell[0,...]).sum() == 0.0
    # special case: amask is string: 'Ca' -> sy=='Ca' bool array
    sy = np.array(trajs[0].symbols)
    for ii in range(len(amask)):
        if type(amask[ii]) == type('x'):
            amask[ii] = sy==amask[ii]
    clst = [trajs[0].coords_frac[tmask,amask[0],:],
            trajs[1].coords_frac[tmask,amask[1],:]]
    # Add time axis back if removed after time slice, e.g. if tmask=np.s_[-1]
    # (only one step). One could also slice ararys and put them thru the
    # Trajectory() machinery again to assert 3d arrays.
    for ii in range(len(clst)):
        if len(clst[ii].shape) == 2:
            clst[ii] = clst[ii][None,...]
            assert len(clst[ii].shape) == 3
            assert clst[ii].shape[2] == 3
    natoms0 = clst[0].shape[1]
    natoms1 = clst[1].shape[1]
    # assume fixed cell, 2d 
    cell = trajs[0].cell[0,...]
    volume = trajs[0].volume[0] 
    nstep = float(clst[0].shape[0])
    rmax_auto = rmax_smith(cell)
    if rmax == 'auto':
        rmax = rmax_auto
    bins = np.arange(0, rmax+dr, dr)
    rad = bins[:-1]+0.5*dr
    volume_shells = 4.0/3.0*pi*(bins[1:]**3.0 - bins[:-1]**3.0)
    norm_fac_pre = volume / volume_shells

    # distances
    # sij: (nstep, natoms0, natoms1, 3)
    sij = clst[0][:,:,None,:] - clst[1][:,None,:,:]
    assert sij.shape == (nstep, natoms0, natoms1, 3)
    if pbc:
        sij = min_image_convention(sij)
    # sij: (nstep, atoms0 * natoms1, 3)
    sij = sij.reshape(nstep, natoms0*natoms1, 3)
    # rij: (nstep, natoms0 * natoms1, 3)
    rij = np.dot(sij, cell)
    # dists_all: (nstep, natoms0 * natoms1)
    dists_all = np.sqrt((rij**2.0).sum(axis=2))
    
    if norm_vmd:
        msk = dists_all < 1e-15
        dups = [len(np.nonzero(entry)[0]) for entry in msk]
    else:
        dups = np.zeros((nstep,))
    # Not needed b/c bins[-1] == rmax, but doesn't hurt. Plus, test_rpdf.py
    # would fail b/c old reference data calculated w/ that setting (difference
    # 1%, only the last point differs).
    dists_all[dists_all >= rmax] = 0.0
    
    if dmask is not None:
        placeholder = '{d}'
        if placeholder in dmask:
            _dmask = dmask.replace(placeholder, 'dists_all')
        else:
            _dmask = 'dists_all ' + dmask
        dists_all[np.invert(eval(_dmask))] = 0.0

    hist_sum = np.zeros(len(bins)-1, dtype=float)
    number_integral_sum = np.zeros(len(bins)-1, dtype=float)
    # Calculate hists for each time step and average them. This Python loop is
    # the bottleneck if we have many timesteps.
    for idx in range(int(nstep)):
        dists = dists_all[idx,...]
        norm_fac = norm_fac_pre / (natoms0 * natoms1 - dups[idx])
        # rad_hist == bins
        hist, rad_hist = np.histogram(dists, bins=bins)
        # works only if we don't set dists_all[dists_all >= rmax] = 0.0
        ##hist[0] -= dups[idx]
        if bins[0] == 0.0:
            hist[0] = 0.0
        hist_sum += hist * norm_fac
        # The result is always the same b/c if norm_vmd=False, then
        # dups[idx]=0.0 and the equation reduces to the exact same.
        ##if norm_vmd:
        ##    number_integral = np.cumsum(hist)*1.0 / natoms0
        ##else:            
        ##    number_integral = np.cumsum(1.0*natoms1/volume*hist*norm_fac*4*pi*rad**2.0 * dr)
        number_integral = np.cumsum(hist)*1.0 / natoms0
        number_integral_sum += number_integral
    out = np.empty((len(rad), 3))
    out[:,0] = rad
    out[:,1] = hist_sum / nstep
    out[:,2] = number_integral_sum / nstep
    return out


def call_vmd_measure_gofr(trajfn, dr=None, rmax=None, sel=['all','all'],
                          fntype='xsf', first=0, last=-1, step=1, usepbc=1,
                          datafn=None, scriptfn=None, logfn=None, tmpdir=None,
                          verbose=False):
    """Call VMD's "measure gofr" command. This is a simple interface which does
    in fact the same thing as the gofr GUI, only scriptable. Accepts a file
    with trajectory data.

    Only orthogonal boxes are allowed (like in VMD).
    
    args:
    -----
    trajfn : filename of trajectory which is fed to VMD (e.g. foo.axsf)
    dr : float
        dr in Angstrom
    rmax : float
        Max. radius up to which minimum image nearest neighbors are counted.
        For cubic boxes of side length L, this is L/2 [AT,MD].
    sel : list of two strings, optional
        string to select atoms, ["name Ca", "name O"], ["all", "all"], ...,
        where sel[0] is selection 1, sel[1] is selection 2 in VMD
    fntype : str, optional
        file type of `fn` for the VMD "mol" command
    first, last, step: int, optional
        Select which MD steps are averaged. Like Python, VMD starts counting at
        0. Last is -1, like in Python. 
    usepbc : int {1,0}, optional
        Whether to use the minimum image convention.
    datafn : str, optional
        temp file where VMD results are written to and loaded
    scriptfn : str, optional
        temp file where VMD tcl input script is written to
    logfn : str, optional
        file where VMD output is logged 
    tmpdir : str, optional
        dir where auto-generated tmp files are written
    verbose : bool, optional
        display VMD output

    returns:
    --------
    array (len(rad), 3), colums 0,1,2:
    rad : 1d array, radius (x-axis) with spacing `dr`, each value r[i] is the
        middle of a histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),), the (averaged) number integral
        number_density*hist*4*pi*r**2.0*dr
    """
    vmd_tcl = textwrap.dedent("""                    
    # VMD interface script. Call "measure gofr" and write RPDF to file.
    # Tested with VMD 1.8.7, 1.9
    #
    # Automatically generated by pwtools, XXXTIME
    #
    # Format of the output file (columns):
    #
    # radius    avg(g(r))    avg(number integral)
    # [Ang]
    
    # Load molecule file with MD trajectory. Typically, foo.axsf with type=xsf
    mol new XXXTRAJFN type XXXFNTYPE  waitfor all
    
    # "top" is the current top molecule (the one labeled with "T" in the GUI). 
    set molid top
    set selstr1 "XXXSELSTR1"
    set selstr2 "XXXSELSTR2"
    set first XXXFIRST 
    set last XXXLAST
    set step XXXSTEP 
    set delta XXXDR
    set rmax XXXRMAX
    set usepbc XXXUSEPBC
    
    set sel1 [atomselect $molid "$selstr1"]
    set sel2 [atomselect $molid "$selstr2"]
    
    # $result is a list of 5 lists, we only need the first 3
    set result [measure gofr $sel1 $sel2 delta $delta rmax $rmax first $first last $last step $step usepbc $usepbc]
    set rad [lindex $result 0]
    set hist [lindex $result 1]
    set num_int [lindex $result 2]
    
    # write to file
    set fp [open "XXXDATAFN" w]
    foreach r $rad h $hist i $num_int {
        puts $fp "$r $h $i"
    }    
    quit
    """)
    # Skip test if cell is orthogonal, VMD will complain anyway if it isn't
    assert None not in [dr, rmax], "`dr` or `rmax` is None"
    assert len(sel) == 2
    assert fntype == 'xsf', ("only XSF files supported")
    if tmpdir is None:
        tmpdir = '/tmp'
    if datafn is None:
        datafn = tempfile.mkstemp(dir=tmpdir, prefix='vmd_data_', text=True)[1]
    if scriptfn is None:
        scriptfn = tempfile.mkstemp(dir=tmpdir, prefix='vmd_script_', text=True)[1]
    if logfn is None:
        logfn = tempfile.mkstemp(dir=tmpdir, prefix='vmd_log_', text=True)[1]
    dct = {}
    dct['trajfn'] = trajfn
    dct['fntype'] = fntype
    dct['selstr1'] = sel[0]
    dct['selstr2'] = sel[1]
    dct['first'] = first
    dct['last'] = last
    dct['step'] = step
    dct['dr'] = dr
    dct['rmax'] = rmax
    dct['usepbc'] = usepbc
    dct['datafn'] = datafn
    dct['time'] = time.asctime()
    for key,val in dct.iteritems():
        vmd_tcl = vmd_tcl.replace('XXX'+key.upper(), str(val))
    common.file_write(scriptfn, vmd_tcl)
    cmd = "vmd -dispdev none -eofexit -e %s " %scriptfn
    if verbose:
        cmd += "2>&1 | tee %s" %logfn
    else:        
        cmd += " > %s 2>&1" %logfn
    out = common.backtick(cmd).strip()
    if out != '': print(out)
    data = np.loadtxt(datafn)
    return data

def vmd_measure_gofr(traj, dr, rmax='auto', sel=['all','all'], first=0,
                     last=-1, step=1, usepbc=1, 
                     slicefirst=True, verbose=False, tmpdir=None):
    """Call call_vmd_measure_gofr(), accept Structure / Trajectory as input.
    This is intended as a complementary function to rpdf() and should, of
    course, produce the "same" results.

    Only orthogonal boxes are allowed (like in VMD).
    
    args:
    -----
    traj : Structure or Trajectory
    dr : float
        dr in Angstrom
    rmax : {'auto', float}, optional
        Max. radius up to which minimum image nearest neighbors are counted.
        For cubic boxes of side length L, this is L/2 [AT,MD].
        'auto': the method of [Smith] is used to calculate the max. sphere
            raduis for any cell shape
        float: set value yourself
    sel : list of two strings, optional
        string to select atoms, ["name Ca", "name O"], ["all", "all"], ...,
        where sel[0] is selection 1, sel[1] is selection 2 in VMD
    first, last, step: int, optional
        Select which MD steps are averaged. Like Python, VMD starts counting at
        0. Last is -1, like in Python. 
    usepbc : int {1,0}, optional
        Whether to use the minimum image convention.
    slicefirst : bool, optional
        Whether to slice coords here in the wrapper based on first,last,step.
        This will write a smaller XSF file, which can save time. In the VMD
        script, we always use first=0,last=-1,step=1 in that case.
    verbose : bool, optional
        display VMD output
    tmpdir : str, optional
        dir where auto-generated tmp files are written

    returns:
    --------
    array (len(rad), 3), colums 0,1,2:
    rad : 1d array, radius (x-axis) with spacing `dr`, each value r[i] is the
        middle of a histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),), the (averaged) number integral
        number_density*hist*4*pi*r**2.0*dr
    """
    from pwtools.io import write_axsf
    traj = struct2traj(traj)
    # Speed: The VMD command "measure gofr" is multithreaded and written in C.
    # That's why it is faster then the pure Python rpdf() above when we have to
    # average many timesteps. But the writing of the .axsf file here is
    # actually the bottleneck and makes this function slower.
    if tmpdir is None:
        tmpdir = '/tmp'
    trajfn = tempfile.mkstemp(dir=tmpdir, prefix='vmd_xsf_', text=True)[1]
    cell = traj.cell[0,...]
    cc = traj.cryst_const[0,...]
    if np.abs(cc[3:] - 90.0).max() > 0.1:
        print cell
        raise StandardError("`cell` is not orthogonal, check angles")
    rmax_auto = rmax_smith(cell)
    if rmax == 'auto':
        rmax = rmax_auto
    # Slice here and write less to xsf file (speed!). Always use first=0,
    # last=-1, step=1 in vmd script.
    if slicefirst:
        sl = slice(first, None if last == -1 else last+1, step)
        traj2 = Trajectory(coords_frac=traj.coords_frac[sl,...],        
                           cell=cell,
                           symbols=traj.symbols)
        first = 0
        last = -1
        step = 1
    else:
        traj2 = traj
    write_axsf(trajfn, traj2)
    ret = call_vmd_measure_gofr(trajfn, dr=dr, rmax=rmax, sel=sel, 
                                fntype='xsf', first=first,
                                last=last, step=step, usepbc=usepbc,
                                verbose=verbose,tmpdir=tmpdir)
    return ret

#-----------------------------------------------------------------------------
# Container classes for crystal structures and trajectories.
#-----------------------------------------------------------------------------

class UnitsHandler(FlexibleGetters):
    def __init__(self, units=None):
        # XXX cryst_const is not in 'length' and needs to be treated specially,
        # see _apply_units_raw()
        self.units_map = \
            {'length':      ['cell', 'coords', 'abc'],
             'energy':      ['etot', 'ekin'],
             'stress':      ['stress'],
             'forces':      ['forces'],
             'temperature': ['temperature'],
             'velocity':    ['velocity'],
             'time':        ['timestep'],
             }
        self._default_units = dict([(key, 1.0) for key in self.units_map.keys()])
        self.units_applied = False
        self.init_units()
        self.update_units(units)

    def _apply_units_raw(self):
        """Only used by derived classes. Apply unit factors to all attrs in
        self.units_map."""
        assert not self.units_applied, ("_apply_units_raw() already called")
        # XXX special-case cryst_const for trajectory case here (ndim = 2), it
        # would be better to split cryst_const into self.abc and self.angles or
        # so, but that would break too much code, BUT we could just add
        # backward compat get_cryst_const, which concatenates these ...
        if self.is_set_attr('cryst_const'):
            cc = self.cryst_const.copy()
            if cc.ndim == 1:
                cc[:3] *= self.units['length']
            elif cc.ndim == 2:
                cc[:,:3] *= self.units['length']
            else:
                raise StandardError("self.cryst_const has ndim != [1,2]")
            self.cryst_const = cc
        for unit, lst in self.units_map.iteritems():
            if self.units[unit] != 1.0:
                for attr_name in lst:
                    if self.is_set_attr(attr_name):
                        attr = getattr(self, attr_name)
                        setattr(self, attr_name,  attr * self.units[unit])
        self.units_applied = True
    
    def apply_units(self):
        """Like _apply_units_raw(), make sure that units are only applied once."""
        if not self.units_applied:
            self._apply_units_raw()

    def init_units(self):
        """Init all unit factors in self.units to 1.0."""
        self.units = self._default_units.copy()
    
    def update_units(self, units):
        """Update self.units dict from `units`. All units not contained in
        `units` remain at the default (1.0), see self._default_units.
        
        args:
        -----
        units : dict, {'length': 5, 'energy': 30, ...}
        """
        if units is not None:
            all_units = self.units_map.keys()
            for key in units.keys():
                if key not in all_units:
                    raise StandardError("unknown unit: %s" %str(key))
            self.units.update(units)

class Structure(UnitsHandler):
    """Base class for containers which hold a single crystal structure (unit
    cell + atoms).
    
    This is a defined minimal interface for how to store a crystal structure in
    pwtools.

    Derived classes may add attributes and getters but the idea is that this
    class is the minimal API for how to pass an atomic structure around.
    
    Units are supposed to be similar to ASE:
        length:     Angstrom        (1e-10 m)
        energy:     eV              (1.602176487e-19 J)
        forces:     eV / Angstrom
        stress:     GPa             (not eV/Angstrom**3)
        time:       fs              (1e-15 s)
        velocity:   Angstrom / fs
        mass:       amu             (1.6605387820000001e-27 kg)

    Note that we cannot verify the unit of input args to the constructor, but
    all functions in this package, which use Structure / Trajectory as
    container classes, assume these units.

    This class is very much like ase.Atoms, but without the "calculators".
    You can use get_ase_atoms() to get an Atoms object.
    
    Derived classes should use set_all_auto=False if they call
    Structure.__init__() explicitely in their __init__(). set_all_auto=True is
    default for container use.

    class Derived(Structure):
        def __init__(self, ...):
            Structure.__init__(set_all_auto=False, ...)

    example:
    --------
    >>> symbols=['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> coords_frac=rand(len(symbols),3)
    >>> cryst_const=np.array([5,5,5,90,90,90.0])
    >>> st=structure.Structure(coords_frac=coords_frac, 
    ...                        cryst_const=cryst_const, 
    ...                        symbols=symbols)
    >>> st.symbols
    ['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> st.symbols_unique
    ['Al', 'N']
    >>> st.order
    2}
    >>> st.typat
    [2, 1, 1, 1, 2, 2, 1]
    >>> st.znucl
    [13, 7]
    >>> st.ntypat
    2
    >>> st.nspecies
    3}
    >>> st.coords
    array([[ 1.1016541 ,  4.52833103,  0.57668453],
           [ 0.18088339,  3.41219704,  4.93127985],
           [ 2.98639824,  2.87207221,  2.36208784],
           [ 2.89717342,  4.21088541,  3.13154023],
           [ 2.28147351,  2.39398397,  1.49245281],
           [ 3.16196033,  3.72534409,  3.24555934],
           [ 4.90318748,  2.02974457,  2.49846847]])
    >>> st.coords_frac
    array([[ 0.22033082,  0.90566621,  0.11533691],
           [ 0.03617668,  0.68243941,  0.98625597],
           [ 0.59727965,  0.57441444,  0.47241757],
           [ 0.57943468,  0.84217708,  0.62630805],
           [ 0.4562947 ,  0.47879679,  0.29849056],
           [ 0.63239207,  0.74506882,  0.64911187],
           [ 0.9806375 ,  0.40594891,  0.49969369]])
    >>> st.cryst_const
    array([  5.,   5.,   5.,  90.,  90.,  90.])
    >>> st.cell
    array([[  5.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  3.06161700e-16,   5.00000000e+00,   0.00000000e+00],
           [  3.06161700e-16,   3.06161700e-16,   5.00000000e+00]])
    >>> st.get_ase_atoms()
    Atoms(symbols='NAl3N2Al', positions=..., cell=[[2.64588604295, 0.0, 0.0],
    [1.6201379367036871e-16, 2.64588604295, 0.0], [1.6201379367036871e-16,
    1.6201379367036871e-16, 2.64588604295]], pbc=[True, True, True])
    """
    # from input **kwds, get_<attr>() by default
    input_attr_lst = [\
        'coords',
        'coords_frac',
        'symbols',
        'cell',
        'cryst_const',
        'forces',
        'stress',
        'etot',
        ]
    # derived, get_<attr>() by default
    derived_attr_lst = [\
        'natoms',
        'symbols_unique',
        'order',
        'typat',
        'znucl',
        'ntypat',
        'nspecies',
        'mass',
        'volume',
        'pressure',
        ]
    # extra convenience attrs, derived, have a getter, but don't
    # get_<attr>() them by default
    extra_attr_lst = [\
        'ase_atoms',
        ]
    
    @crys_add_doc
    def __init__(self, 
                 set_all_auto=True,
                 units=None,
                 **kwds):
        """
        args:
        -----
        coords : 2d array (natoms, 3) [Ang]
            Cartesian coords.
            Optional if `coords_frac` given.
        coords_frac : 2d array (natoms, 3)
            Fractional coords w.r.t. `cell`.
            Optional if `coords` given.
        symbols : sequence of strings (natoms,)
            atom symbols
        %(cell_doc)s 
            Vectors are rows. [Ang]
            Optional if `cryst_const` given.
        %(cryst_const_doc)s
            cryst_const[:3] = [a,b,c]. [Ang]
            Optional if `cell` given.
        forces : optional, 2d array (natoms, 3) with forces [eV/Ang]
        stress : optional, 2d array (3,3), stress tensor [GPa]
        etot : optional, scalar, total energy [eV]
        units : optional, dict, see UnitsHandler
        set_all_auto : optional, bool
            Call self.set_all() in __init__(). Set to False in derived
            classes.

        notes:
        ------
        cell, cryst_const : Provide either `cell` or `cryst_const`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See {cell2cc,cc2cell}.
        coords, coords_frac : Provide either `coords` or `coords_frac`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See coord_trans().
        """
        UnitsHandler.__init__(self, units=units)
        self._init(kwds, set_all_auto)
        
    def _init(self, kwds, set_all_auto):
        """Join attr lists, assign input args from **kwds, call set_all() to calculate
        properties if set_all_auto==True. Shall be called in __init__, also in
        derived classes."""
        self.set_all_auto = set_all_auto
        
        # for get_traj()
        self.timeaxis = 0
        
        # init all in self.attr_lst to None
        self.attr_lst = self.input_attr_lst + self.derived_attr_lst
        self.init_attr_lst()
        self.init_attr_lst(self.extra_attr_lst)
        
        # input args may overwrite default None
        #   self.foo = foo
        #   self.bar = bar
        #   ...
        for attr in kwds.keys():
            if attr not in self.input_attr_lst:
                raise StandardError("illegal input arg: '%s', allowed: %s" \
                    %(attr, str(self.input_attr_lst)))
            else:                    
                setattr(self, attr, kwds[attr])
        
        if self.set_all_auto:
            self.set_all()
    
    def set_all(self):
        """Populate object. Apply units, call all getters."""
        self.apply_units()
        # Call UnitsHandler.set_all(), which is FlexibleGetters.set_all().
        super(Structure, self).set_all()
    
    def _extend_if_possible(self, arr, nstep):
        if arr is not None:
            return num.extend_array(arr, nstep, axis=self.timeaxis)
        else:
            return None

    def get_traj(self, nstep):
        """Return a Trajectory object, where this Structure is copied `nstep`
        times. Only structure-related attrs are passed."""
        return Trajectory(coords=self._extend_if_possible(self.coords, nstep),
                          coords_frac=self._extend_if_possible(self.coords_frac,
                            nstep),
                          cell=self._extend_if_possible(self.cell, nstep),
                          cryst_const=self._extend_if_possible(self.cryst_const,
                            nstep),
                          symbols=self.symbols,
                          forces=self._extend_if_possible(self.forces, nstep),
                          stress=self._extend_if_possible(self.stress, nstep))

    def get_coords(self):
        if not self.is_set_attr('coords'):
            if self.is_set_attr('coords_frac') and \
               self.check_set_attr('cell'):
                # short-cut to bypass coord_trans() 
                return np.dot(self.coords_frac, self.cell)
            else:
                return None
        else:
            return self.coords
    
    def get_coords_frac(self):
        if not self.is_set_attr('coords_frac'):
            if self.is_set_attr('coords') and self.check_set_attr('cell'):
                return coord_trans(coords=self.coords,
                                   old=np.identity(3),
                                   new=self.cell)
            else:
                return None
        else:
            return self.coords_frac
    
    def get_symbols(self):
        return self.symbols
    
    def get_forces(self):
        return self.forces

    def get_stress(self):
        return self.stress

    def get_etot(self):
        return self.etot

    def get_cell(self):
        if not self.is_set_attr('cell'):
            if self.is_set_attr('cryst_const'):
                return cc2cell(self.cryst_const)
            else:
                return None
        else:
            return self.cell
    
    def get_cryst_const(self):
        if not self.is_set_attr('cryst_const'):
            if self.is_set_attr('cell'):
                return cell2cc(self.cell)
            else:
                return None
        else:
            return self.cryst_const
    
    def get_natoms(self):
        if self.is_set_attr('symbols'):
            return len(self.symbols)
        elif self.is_set_attr('coords'):
            return self.coords.shape[0]
        elif self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[0]
        else:
            return None
    
    def get_ase_atoms(self):
        """Return ASE Atoms object. Obviously, you must have ASE installed. We
        use scaled_positions=self.coords_frac, so only self.cell must be in
        [Ang].
        """
        req = ['coords_frac', 'cell', 'symbols']
        if self.check_set_attr_lst(req):
            # We don't wanna make ase a dependency. Import only when needed.
            from ase import Atoms
            return Atoms(symbols=self.symbols,
                         scaled_positions=self.coords_frac,
                         cell=self.cell,
                         pbc=[1,1,1])
        else:
            return None

    def get_symbols_unique(self):
        return np.unique(self.symbols).tolist() if \
            self.check_set_attr('symbols') else None

    def get_order(self):
        if self.check_set_attr('symbols_unique'):
            return dict([(sym, num+1) for num, sym in
                         enumerate(self.symbols_unique)])
        else:
            return None

    def get_typat(self):
        if self.check_set_attr_lst(['symbols', 'order']):
            return [self.order[ss] for ss in self.symbols]
        else:
            return None
    
    def get_znucl(self):
        if self.check_set_attr('symbols_unique'):
            return [periodic_table.pt[sym]['number'] for sym in self.symbols_unique]
        else:
            return None

    def get_ntypat(self):
        if self.check_set_attr('order'):
            return len(self.order.keys())
        else:
            return None
    
    def get_nspecies(self):
        if self.check_set_attr_lst(['order', 'typat']):
            return dict([(sym, self.typat.count(idx)) for sym, idx in 
                         self.order.iteritems()])
        else:
            return None
    
    def get_mass(self):
        """1D array of atomic masses in amu (atomic mass unit 1.660538782e-27
        kg as in periodic table). The order is the one from self.symbols."""
        if self.check_set_attr('symbols'):
            return np.array([periodic_table.pt[sym]['mass'] for sym in
                             self.symbols])
        else:
            return None
    
    def get_volume(self):
        if self.check_set_attr('cell'):
            return volume_cell(self.cell)
        else:
            return None
    
    def get_pressure(self):
        """As in PWscf, pressure = 1/3*trace(stress), ignoring
        off-diagonal elements."""
        if self.check_set_attr('stress'):
            return np.trace(self.stress)/3.0
        else:
            return None

class Trajectory(Structure):
    """Here all arrays (input and attrs) have a time axis, i.e. all arrays
    have one dim more along the time axis (self.timeaxis) compared to
    Structure, e.g. 
        coords      (natoms,3)  -> (nstep, natoms, 3)
        cryst_const (6,)        -> (nstep, 6)
        ...
    
    An exception for fixed-cell MD-data are the inputs ``cell`` (
    ``cryst_const``), which can be 2d (1d)  and will be "broadcast"
    automatically along the time axis (see num.extend_array()).
    
    args:
    -----
    See Structure, plus:

    ekin : optional, 1d array [eV]
        Kinetic energy of the ions. Calculated from `velocity` if not given.
    temperature : optional, 1d array [K]        
        Ionic temperature. Calculated from `ekin` if not given.
    velocity: optional, 3d array (nstep, natoms, 3) [Ang / fs]
        Ionic velocity. Calculated from `coords` if not given.
    timestep : scalar [fs]
        Ionic (and cell) time step.
    
    notes:
    ------
    We calculate coords -> velocity -> ekin -> temperature for the ions if
    these quantities are not provided as input args. One could do the same
    thing for cell, if one treats `cell` as coords of 3 atoms. This is not done
    currently. We would also need a new input arg mass_cell. The CPMD parsers
    have something like ekin_cell, temperature_cell etc, parsed from CPMD's
    output, though.
    """
    # additional input args, some are derived if not given in the input
    input_attr_lst = Structure.input_attr_lst + [\
        'ekin',
        'temperature',
        'velocity',
        'timestep',
        ]
    derived_attr_lst = Structure.derived_attr_lst + [\
        'nstep',
        ]
    extra_attr_lst = Structure.extra_attr_lst + [\
        ]
    timeaxis = 0        
    
    def set_all(self):
        """Populate object. Apply units, extend arrays, call all getters."""
        # If these are given as input args, then they must be 3d.        
        self.attrs_3d = ['coords', 
                         'coords_frac', 
                         'stress', 
                         'forces',
                         'velocity']
        for attr in self.attrs_3d:
            if self.is_set_attr(attr):
                assert getattr(self, attr).ndim == 3, ("not 3d array: %s" %attr)
        self.apply_units()                
        self._extend() 
        # Don't call super(Trajectory, self).set_all(), as this will call
        # Structure.set_all(), which in turn may do something we don't want,
        # like applying units 2 times. ATM, it would work b/c
        # UnitsHandler.apply_units() won't do that.
        super(Structure, self).set_all()

    def _extend(self):
        if self.is_set_attr('cell') and self.check_set_attr('nstep'):
            self.cell = self._extend_cell(self.cell)
        if self.is_set_attr('cryst_const') and self.check_set_attr('nstep'):
            self.cryst_const = self._extend_cc(self.cryst_const)

    def _extend_array(self, arr, nstep=None):
        if nstep is None:
            self.assert_set_attr('nstep')
            nstep = self.nstep
        return num.extend_array(arr, nstep, axis=self.timeaxis)
    
    def _extend_cell(self, cell):
        if cell is None:
            return cell
        if cell.shape == (3,3):
            return self._extend_array(cell)
        elif cell.shape == (1,3,3):            
            return self._extend_array(cell[0,...])
        else:
            return cell

    def _extend_cc(self, cc):
        if cc is None:
            return cc
        if cc.shape == (6,):
            return self._extend_array(cc)
        elif cc.shape == (1,6):            
            return self._extend_array(cc[0,...])
        else:
            return cc
    
    def get_velocity(self):
        # Simple finite differences are used. This is OK if self.timestep is
        # small, e.g. the trajectory is smooth. But forces calculated from that
        # (force = mass * dv / dt) are probably not very accurate,
        # which is OK b/c we don't do that :)
        # One *could* create 3*natoms Spline objects thru coords (splines along
        # time axis) and calc 1st and 2nd deriv from that. Just an idea ...
        if not self.is_set_attr('velocity'):
            if self.check_set_attr_lst(['coords', 'timestep']):
                return np.diff(self.coords, n=1, axis=self.timeaxis) / self.timestep
            else:
                return None
        else:
            return self.velocity
    
    def get_ekin(self):
        if not self.is_set_attr('ekin'):
            if self.check_set_attr_lst(['mass', 'velocity']):
                # velocity [Ang/fs], mass [amu]
                vv = self.velocity
                mm = self.mass
                amu = constants.amu # kg
                fs = constants.fs
                eV = constants.eV
                assert self.timeaxis == 0
                return ((vv**2.0).sum(axis=2)*mm[None,:]/2.0).sum(axis=1) * (Angstrom/fs)**2 * amu / eV
            else:
                return None
        else:
            return self.ekin
    
    def get_temperature(self):
        if not self.is_set_attr('temperature'):
            if self.check_set_attr_lst(['ekin', 'natoms']):
                return self.ekin * constants.eV / self.natoms / constants.kb * (2.0/3.0)
            else:
                return None
        else:
            return self.temperature

    def get_ase_atoms(self):
        raise NotImplementedError("makes no sense for trajectories")

    def get_natoms(self):
        if self.is_set_attr('symbols'):
            return len(self.symbols)
        elif self.is_set_attr('coords'):
            return self.coords.shape[1]
        elif self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[1]
        else:
            return None
    
    def get_coords(self):
        if not self.is_set_attr('coords'):
            if self.is_set_attr('coords_frac') and \
               self.check_set_attr_lst(['cell', 'natoms']):
                nstep = self.coords_frac.shape[self.timeaxis]
                req_shape_coords_frac = (nstep,self.natoms,3)
                assert self.coords_frac.shape == req_shape_coords_frac, ("shape "
                    "mismatch: coords_frac: %s, need: %s" %(str(self.coords_frac.shape),
                    str(req_shape_coords_frac)))
                assert self.cell.shape == (nstep,3,3), ("shape mismatch: "
                    "cell: %s, coords_frac: %s" %(self.cell.shape, self.coords_frac.shape))
                # Can use dot() directly if we special-case fixed-cell and use
                # cell = 2d array                    
                arr = coord_trans3d(self.coords_frac,
                                    old=self.cell,
                                    new=self._extend_array(np.identity(3), 
                                                           nstep=nstep),
                                    axis=1,
                                    timeaxis=self.timeaxis)
                return arr
            else:
                return None
        else:
            return self.coords

    def get_coords_frac(self):
        if not self.is_set_attr('coords_frac'):
            if self.is_set_attr('coords') and \
               self.check_set_attr_lst(['cell', 'natoms']):
                nstep = self.coords.shape[self.timeaxis]
                req_shape_coords = (nstep,self.natoms,3)
                assert self.coords.shape == req_shape_coords, ("shape "
                    "mismatch: coords: %s, need: %s" %(str(self.coords.shape),
                    str(req_shape_coords)))
                assert self.cell.shape == (nstep,3,3), ("shape mismatch: "
                    "cell: %s, coords: %s" %(self.cell.shape, self.coords.shape))
                arr = coord_trans3d(self.coords,
                                    old=self._extend_array(np.identity(3),
                                                           nstep=nstep),
                                    new=self.cell,
                                    axis=1,
                                    timeaxis=self.timeaxis)
                return arr
            else:
                return None
        else:
            return self.coords_frac
    
    def get_volume(self):
        if self.check_set_attr('cell'):
            return volume_cell3d(self.cell, axis=self.timeaxis)
        else:
            return None
    
    def get_cell(self):
        if not self.is_set_attr('cell'):
            if self.is_set_attr('cryst_const'):
                cc = self._extend_cc(self.cryst_const)
                return cc2cell3d(cc, axis=self.timeaxis)
            else:
                return None
        else:
            return self._extend_cell(self.cell)
    
    def get_cryst_const(self):
        if not self.is_set_attr('cryst_const'):
            if self.is_set_attr('cell'):
                cell = self._extend_cell(self.cell)
                return cell2cc3d(cell, axis=self.timeaxis)
            else:
                return None
        else:
            return self._extend_cc(self.cryst_const)
    
    def get_nstep(self):
        if self.is_set_attr('coords'):
            return self.coords.shape[self.timeaxis]
        elif self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[self.timeaxis]
        else:
            return None
    
    def get_pressure(self):
        if self.check_set_attr('stress'):
            assert self.timeaxis == 0
            return np.trace(self.stress,axis1=1, axis2=2)/3.0
        else:
            return None
    
    def get_timestep(self):
        return self.timestep

    def get_traj(self):
        return None


def struct2traj(obj):
    """Transform Structure to Trajectory with nstep=1."""
    # XXX not very safe test, need to play w/ isinstance(). May be a problem
    # b/c Structure -> Trajectory inherit.
    if hasattr(obj, 'get_nstep'):
        return obj
    else:
        return obj.get_traj(nstep=1)
    
