# Crystal and unit-cell related tools, MD analysis, container classes

from math import acos, pi, sin, cos, sqrt
import textwrap, time, os, tempfile, types, copy, itertools

import numpy as np
from scipy.linalg import inv

from pwtools import common, signal, num, atomic_data, constants, _flib
from pwtools.common import assert_cond
from pwtools.decorators import crys_add_doc
from pwtools.base import FlexibleGetters
from pwtools.constants import Bohr, Angstrom
from pwtools.num import fempty, rms, rms3d, match_mask, norm
import warnings
warnings.simplefilter('always')


#-----------------------------------------------------------------------------
# misc math
#-----------------------------------------------------------------------------

def angle(x,y):
    """Angle between vectors `x` and `y` in degrees.
    
    Parameters
    ----------
    x,y : 1d numpy arrays
    """
    # Numpy's `acos' is "acrcos", but we take the one from math for scalar
    # args.
    return acos(np.dot(x,y)/norm(x)/norm(y))*180.0/pi


#-----------------------------------------------------------------------------
# crystallographic constants and basis vectors
#-----------------------------------------------------------------------------

@crys_add_doc
def volume_cell(cell):
    """Volume of the unit cell from cell vectors. Calculates the triple
    product::
        
        np.dot(np.cross(a,b), c) == det(cell)
    
    of the basis vectors a,b,c contained in `cell`. Note that (mathematically)
    the vectors can be either the rows or the cols of `cell`.

    Parameters
    ----------
    %(cell_doc)s

    Returns
    -------
    volume, unit: [a]**3

    Examples
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
    """    
    assert_cond(cell.shape == (3,3), "input must be (3,3) array")
##    return np.dot(np.cross(cell[0,:], cell[1,:]), cell[2,:])
    return abs(np.linalg.det(cell))

def volume_cell3d(cell, axis=0):
    """Same as volume_cell() for 3d arrays.
    
    Parameters
    ----------
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
    """Volume of the unit cell from crystallographic constants [1]_.
    
    Parameters
    ----------
    %(cryst_const_doc)s
    
    Returns
    -------
    volume, unit: [a]**3
    
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Parallelepiped
    """
    assert len(cryst_const) == 6, "shape must be (6,)"
    a = cryst_const[0]
    b = cryst_const[1]
    c = cryst_const[2]
    alpha = cryst_const[3]*pi/180
    beta = cryst_const[4]*pi/180
    gamma = cryst_const[5]*pi/180
    return a*b*c*sqrt(1+ 2*cos(alpha)*cos(beta)*cos(gamma) -\
          cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 )


def volume_cc3d(cryst_const, axis=0):
    """Same as volume_cc() for 2d arrays (the name "3d" is just to indicate
    that we work w/ trajectories).
    
    Parameters
    ----------
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
    """From `cell` to crystallographic constants a, b, c, alpha, beta,
    gamma. 
    This mapping is unique in the sense that multiple cells will have
    the same `cryst_const`, i.e. the representation of the cell in
    `cryst_const` is independent from the spacial orientation of the cell
    w.r.t. a cartesian coord sys.
    
    Parameters
    ----------
    %(cell_doc)s

    Returns
    -------
    %(cryst_const_doc)s, 
        unit: [a]**3
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
    
    Parameters
    ----------
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
    gamma to `cell`.
    This mapping not NOT unique in the sense that one set of `cryst_const` can
    have arbitrarily many representations in terms of cells. We stick to a
    common convention. See notes below.
    
    Parameters
    ----------
    %(cryst_const_doc)s
    
    Returns
    -------
    %(cell_doc)s
        unit: [a]**3
    
    Notes
    -----
    Basis vectors fulfilling the crystallographic constants are arbitrary
    w.r.t. their orientation in space. We choose the common convention that

    | va : along x axis
    | vb : in the x-y plane
    
    Then, vc is fixed.
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
    """Same as cc2cell() for 2d arrays (the name "3d" is just to indicate
    that we work w/ trajectories).
    
    Parameters
    ----------
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
    """Reciprocal lattice vectors ``{a,b,c}* = 2*pi / V * {b,c,a} x {c,a,b}``.
    
    The reciprocal volume is ``(2*pi)**3/V``. The length unit of the reciprocal
    vectors is 1/(length unit of `cell`), e.g. 1/Angstrom.

    Parameters
    ----------
    %(cell_doc)s

    Returns
    -------
    rcell : array (3,3) 
        Reciprocal vectors as rows.
    
    Examples
    --------
    >>> # the length of recip. cell vectors for a cubic cell of 1 Ang side
    >>> # length is 2*pi -> reciprocal length unit is 1/Ang
    >>> crys.recip_cell(identity(3))/2/pi
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> crys.recip_cell(identity(3)*2)/2/pi
    array([[ 0.5,  0. ,  0. ],
           [ 0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0.5]])
    """
    cell = np.asarray(cell, dtype=float)
    assert_cond(cell.shape == (3,3), "cell must be (3,3) array")
    rcell = np.empty_like(cell)
    vol = volume_cell(cell)
    a = cell[0,:]
    b = cell[1,:]
    c = cell[2,:]
    rcell[0,:] = 2*pi/vol * np.cross(b,c)
    rcell[1,:] = 2*pi/vol * np.cross(c,a)
    rcell[2,:] = 2*pi/vol * np.cross(a,b)
    return rcell


def grid_in_cell(cell, h=None, size=None, minpoints=1, even=False, fullout=False):
    """For a given cell, generate grid `size` from grid spacing `h` or vice
    versa. 
    
    Usually used to calculate k-grids for reciprocal cells. See also `kgrid()`.

    Parameters
    ----------
    cell : array (3,3)
        Cell with vectors as rows.
    h : float
        1d target grid spacing along a cell axis. Unit is that of the cell
        sides.
    size : sequence (3,)
        Use either `h` or `size`.
    minpoints : int
        Minimal number of grid points in each direction. May result in smaller
        effective `h`. `minpoints=1` (default) asserts that at least the
        Gamma point [1,1,1] is returned.  Otherwise, big cells or big `h`
        values will create zero grid points.
    even : bool
        Force even grid point numbers. Here, we add 1 to odd points, thus
        creating a grid more dense than requested by `h`.
    fullout : bool
        See below.

    Returns
    -------
    size : if `h != None` + `fullout=False` 
    size, spacing : if `h != None` + `fullout=True`
    spacing : if `size` != None and `h=None`
    size : array (3,) [nx, ny, nz]
        Integer numbers of grid points along each reciprocal axis.
    spacing : 1d array (3,) 
        Result spacing along each reciprocal axis if `size` would be used.
   
    Notes
    -----
    * B/c an integer array is created by rounding, the effective grid spacing
      will mostly be slightly bigger/smaller then `h` (see `fullout`).

    Examples
    --------
    >>> crys.grid_in_cell(diag([1,2,3]), h=1)
    array([1, 2, 3])
    >>> crys.grid_in_cell(diag([1,2,3]), h=0.5)
    array([2, 4, 6])
    >>> crys.grid_in_cell(diag([1,2,3]), h=0.5, fullout=True)
    (array([2, 4, 6]), array([ 0.5,  0.5,  0.5]))
    >>> crys.grid_in_cell(diag([1,2,3]), size=[2,2,2])
    array([ 0.5,  1. ,  1.5])
    """
    spacing = h
    assert None in [spacing, size], "use either `h` or `size`"
    assert minpoints >= 0
    cell = np.asarray(cell, dtype=float)
    norms = np.sqrt((cell**2.0).sum(axis=1))
    if size is None:
        size = np.round(norms / spacing)
        if even:
            size += (size % 2.0)
        size = size.astype(int)
        mask = size < minpoints
        if mask.any():
            size[mask] = minpoints
        # only possible if minpoints=0        
        if (size == 0).any():
            raise StandardError("at least one point count is zero, decrease `spacing`, "
                                 "size=%s" %str(size))
        if fullout:
            return size, norms * 1.0 / size
        else:
            return size.astype(int)
    else:
        size = np.array(size)
        return norms * 1.0 / size


def kgrid(cell, **kwds):
    """Calculate k-point grid for given real-space cell or grid spacing from
    grid size.

    This is a convenience and backward compat function which does
    ``grid_in_cell(recip_cell(cell), **kwds)``.
    
    Parameters
    ----------
    cell : array (3,3)
        Real space unit cell.
    **kwds : See grid_in_cell()        

    Returns
    -------
    See grid_in_cell().

    Notes
    -----
    * Since the reciprocal cell is calculated with `recip_cell`, ``h=0.5``
      1/Ang seems to produce a sufficiently dense grid for insulators. Metals
      need a finer k-grid for electrons.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pwtools.crys import kgrid
    >>> cell = np.diag([5,5,8])
    >>> kgrid(cell, h=0.5)
    array([3, 3, 2])
    >>> # see effective grid spacing
    >>> kgrid(cell, h=0.5, fullout=True)
    (array([3, 3, 2]), array([ 0.41887902,  0.41887902,  0.39269908]))
    >>> # reverse: effective grid spacing for given size
    >>> kgrid(cell, size=[3,3,2])
    array([ 0.41887902,  0.41887902,  0.39269908])
    >>> # even grid
    >>> kgrid(cell, h=0.5, even=True)
    array([4, 4, 2])
    >>> # big cell, at least Gamma with minpoints=1
    >>> kgrid(cell*10, h=0.5)
    array([1, 1, 1])
    >>> # Create MP mesh
    >>> ase.dft.monkhorst_pack(kgrid(cell, h=0.5))
    >>> # cell: 1 Ang side length, recip cell 2*pi/Ang side length, 
    >>> # unit of h: 1/Ang
    >>> crys.recip_cell(np.identity(3))
    array([[ 6.28318531,  0.        ,  0.        ],
           [ 0.        ,  6.28318531,  0.        ],
           [ 0.        ,  0.        ,  6.28318531]])
    >>> kgrid(np.identity(3), h=pi, fullout=True)
    (array([2, 2, 2]), array([ 3.14159265,  3.14159265,  3.14159265]))
    """
    if kwds.has_key('dk'):
        warnings.warn("`dk` is deprecated, use `h` instead",
                      DeprecationWarning)
        kwds['h'] = kwds['dk']
        kwds.pop('dk')
    return grid_in_cell(recip_cell(cell), **kwds)


@crys_add_doc
def cc2celldm(cryst_const, fac=1.0):
    """
    Convert cryst_const to PWscf `celldm`.

    Parameters
    ----------
    %(cryst_const_doc)s
    fac : float, optional
        conversion a[any unit] -> a[Bohr]
    
    Returns
    -------
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
    
    Parameters
    ----------
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

def scell_mask(nx, ny, nz, direc=1):
    """Build a mask for the creation of a nx x ny x nz supercell (for 3d
    coordinates).  
    
    Return all possible permutations with repitition of the integers ix, iy, iz
    = 0, ..., nx-1, ny-1, nz-1 . Dimensions can also be negative, in
    which case i = 0,-1,...,-n+1 . Parameter `direc` reverses the ordering.

    Parameters
    ----------
    nx, ny, nz : int
    direc : int 
        1 or -1, order mask 0,...,n-1 (cells placed "center to edge") or
        reverse n-1,...,0 ("egde to center")

    Returns
    -------
    mask : 2d array, shape (nx*ny*nz, 3)

    Examples
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
    >>> # direction reversed
    >>> scell_mask(2,2,1,direc=-1)
    array([[ 1.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])
    """
    if direc == 1:
        mkrange = lambda x: range(0,x) if x >= 0 else range(0,x,-1)
    elif direc == -1:
        mkrange = lambda x: range(x-1,-1,-1) if x >= 0 else range(x+1,1)
    return np.array([k for k in itertools.product(mkrange(nx), mkrange(ny),
                                                  mkrange(nz))], 
                     dtype=float)


def scell(obj, dims, method=1, **kwds):
    """Build supercell based on `dims`. 

    Uses coords_frac and cell.

    Parameters
    ----------
    obj : Structure or Trajectory
    dims : tuple (nx, ny, nz) for a N = nx * ny * nz supercell
    method : int, optional
        Switch between numpy-ish (1) or loop (2) implementation. (2) should
        always produce correct results but is sublty slower. Only for
        Structure.
    **kwds : see :func:`scell_mask`
    
    Notes
    -----
    The mask for the supercell is created by :func:`scell_mask` and applied to
    each atom in `obj` one after another, i.e. each atom is repeated nx*ny*nz
    times according to the mask pattern, independently of how the pattern looks
    like (e.g. the `direc` parameter in :func:`scell_mask`). So, just as rows
    in np.repeat(), we have:
        | original:     symbols=[A,B,C,D]
        | 2 x 1 x 1:    symbols=[A,A,B,B,C,C,D,D]
        | nx x ny x nz: symbols=[(nx*ny*nz) x A, (nx*ny*nz) x B, ...]
    

    Returns
    -------
    scell : Structure
    """
    # Place each atom N = nx*ny*nz times in the supercell, i.e. copy unit cell
    # N times. Actually, N-1, since ix=iy=iz=0 is the unit cell itself.
    #
    # Let k = {x,y,z}.
    #
    # mask[j,:] = [ix, iy, iz], ik = integers (floats actually, but
    #   mod(ik, floor(ik)) == 0.0)
    #
    # original cell:
    # coords_frac[i,:] = position vect of atom i in the unit cell in *crystal*
    #   coords!!
    # 
    # super cell:
    # sc_coords_frac[i,:] = coords_frac[i,:] + [ix, iy, iz]
    #   for all permutations (see scell_mask()) of ix, iy, iz.
    #   ik = 0, ..., nk - 1
    #
    # sc_coords_frac : crystal coords w.r.t the *old* cell, i.e. the entries are in
    # [0,(max(dims))], not [0,1], is scaled below
    #
    if not kwds.has_key('direc'):
        kwds['direc'] = 1
    mask = scell_mask(*tuple(dims), **kwds)
    nmask = mask.shape[0]
    if obj.is_struct:        
        sc_cell = obj.cell * np.asarray(dims)[:,None]
        container = Structure
    elif obj.is_traj:
        # (nstep,3,3) * (1,3,1) -> (nstep, 3,3)                      
        sc_cell = obj.cell * np.asarray(dims)[None,:,None]
        container = Trajectory
    else:
        raise StandardError("unknown input type")
    if method == 1:
        sc_symbols = np.array(obj.symbols).repeat(nmask).tolist() if (obj.symbols \
                     is not None) else None
        if obj.is_struct:
            # (natoms, 1, 3) + (1, nmask, 3) -> (natoms, nmask, 3)
            sc_coords_frac = (obj.coords_frac[:,None,:] \
                              + mask[None,...]).reshape(obj.natoms*nmask,3)
        elif obj.is_traj:
            # cool, eh?
            # (nstep, natoms, 1, 3) + (1, 1, nmask, 3) -> (nstep, natoms, nmask, 3)
            sc_coords_frac = (obj.coords_frac[...,None,:] \
                              + mask[None,None,...]).reshape(obj.nstep,obj.natoms*nmask,3)
        else:
            raise StandardError("huh!?")
    # explicit loop version for testing, this is the reference implementation,
    # only for Structure
    elif method == 2:
        if obj.is_struct:
            sc_symbols = []
            sc_coords_frac = np.empty((nmask*obj.natoms, 3), dtype=float)
            k = 0
            for iatom in range(obj.natoms):
                for j in range(nmask):
                    if obj.symbols is not None:
                        sc_symbols.append(obj.symbols[iatom])  
                    sc_coords_frac[k,:] = obj.coords_frac[iatom,:] + mask[j,:]
                    k += 1
        else:
            raise StandardError("method=2 only implemented for Structure")
    else:
        raise StandardError("unknown method: %s" %repr(method))
    sc_coords_frac[...,0] /= dims[0]
    sc_coords_frac[...,1] /= dims[1]
    sc_coords_frac[...,2] /= dims[2]
    return container(coords_frac=sc_coords_frac,
                     cell=sc_cell,
                     symbols=sc_symbols)


def scell3d(traj, dims, **kwds):
    """Supercell for Trajectory. Deprecated. Use :func:`scell` instead."""
    warnings.warn("scell3d() is deprecated, use scell() for Trajectory as well",
                  DeprecationWarning)
    return scell(traj, dims, **kwds)

#-----------------------------------------------------------------------------
# atomic coords processing / evaluation, MD analysis
#-----------------------------------------------------------------------------

def velocity_traj(arr, dt=1.0, axis=0, endpoints=True):
    """Calculate velocity from `arr` (usually coordinates) along time`axis`
    using timestep `dt`.
    
    Central differences are used (example x-coord of atom 0:
    ``x=coords[:,0,0]``):: 
        
        v[i] = [ x[i+1] - x[i-1] ] / (2*dt)

    which returns nstep-2 points belonging to the the middle of the
    trajectory x[1:-1]. To get an array which is `nstep` long, the fist and
    last velocity are set to the first and last calculated value (if
    ``endpoints=True``)::

        v[0,...] == v[1,...]
        v[-1,...] == v[-2,...]
    """
    # Central diffs are more accurate than simple finite diffs
    #
    #   v[i] = [ x[i+1] - x[i] ] / dt
    # 
    # These return nstep-1 points (one more then central diffs) but we
    # have the problem of assigning the velocity array to the time axis:
    # t[1:] or t[:-1] are both shifted w.r.t. to the real time axis
    # position -- the correct way would be to assign it to t[:-1] + 0.5*dt.
    # In contrast, central diffs belong to t[1:-1] by definition.
    # 
    # If self.timestep is small (i.e. the trajectory is smooth), all this is
    # not really a problem, but central diffs are just better and more
    # consistent. Even forces calculated from these velocities (force =
    # mass * dv / dt) are reasonably accurate compared to the forces from
    # the MD trajectory input. One could implement get_forces() like that
    # if needed, but so far all MD codes provide us their forces, of
    # course. Also, one *could* create 3*natoms Spline objects thru coords
    # (splines along time axis) and calc 1st and 2nd deriv from that. But
    # that's probably very slow.
    if endpoints:
        vv = np.empty_like(arr)
    # To support general axis stuff, use slice magic ala slicetake/sliceput        
    assert axis == 0, ("only axis==0 implemented ATM")
    tmp = (arr[2:,...] - arr[:-2,...]) / 2.0 / dt
    if endpoints:
        vv[1:-1,...] = tmp
        vv[0,...] = tmp[0,...]
        vv[-1,...] = tmp[-1,...]
    else:
        vv = tmp
    return vv    


def rmsd(traj, ref_idx=0):
    """Root mean square distance over an MD trajectory. 
    
    The normalization constant is the number of atoms. Takes the RMS of the
    difference of *cartesian* coords at each time step. Only meaningful if
    ``tr.coords`` are *not* pbc-wrapped.
    
    Parameters
    ----------
    traj : Trajectory object
    ref_idx : int, optional
        time index of the reference structure (i.e. 0 to compare with the
        start structure, -1 for the last along `axis`).
    
    Returns
    -------
    rmsd : 1d array (traj.nstep,)

    Examples
    --------
    >>> # We only need traj.{coords,nstep,timeaxis}, no symbols, cell, ...
    >>> traj = crys.Trajectory(coords=rand(500,10,3))
    >>> # The RMSD w.r.t. the start structure. See when the structure starts to
    >>> # "converge" to a stable mean configuration during an MD.
    >>> rmsd(traj, ref_idx=0)
    >>> # For a relaxation run, the RMSD w.r.t. the final converged structure. The
    >>> # RMSD should converge to zero here.
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


def pbc_wrap_coords(coords_frac, copy=True, mask=[True]*3, xyz_axis=-1):
    """Apply periodic boundary conditions to array of fractional coords. 
    
    Wrap atoms with fractional coords > 1 or < 0 into the cell.
    
    Parameters
    ----------
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
    
    Returns
    -------
    coords_frac : array_like(coords_frac)
        Array with all values in [0,1] except for those where ``mask[i]=False``.

    Notes
    -----
    About the copy arg: If ``copy=False``, then this is an in-place operation
    and the array in the global scope is modified! In fact, then these do the
    same::
    
        >>> a = pbc_wrap_coords(a, copy=False)
        >>> pbc_wrap_coords(a, copy=False)
    """
    assert coords_frac.shape[xyz_axis] == 3, "dim of xyz_axis of `coords_frac` must be == 3"
    ndim = coords_frac.ndim
    assert ndim in [2,3], "coords_frac must be 2d or 3d array"
    tmp = coords_frac.copy() if copy else coords_frac
    for i in range(3):
        if mask[i]:
            sl = [slice(None)]*ndim
            sl[xyz_axis] = i
            tmp[sl] = np.remainder(tmp[sl], 1.0)
    return tmp        


def pbc_wrap(obj, copy=True, **kwds):
    """Apply periodic boundary conditions to fractional coords. 

    Same as ``pbc_wrap_coords`` but accepts a Structure or Trajectory instead
    of the array ``coords_frac``. Returns an object with atoms
    (coords_frac and coords) wrapped into the cell.

    Parameters
    ----------
    obj : Structure or Trajectory
    copy : bool
        Return copy or in-place modified object.
    **kwds : keywords
        passed to pbc_wrap_coords()
    """
    out = obj.copy() if copy else obj
    # set to None so that it will be re-calculated by set_all()
    out.coords = None
    # copy=False: in-place modify b/c we copied the whole object before if
    # requested by user        
    pbc_wrap_coords(out.coords_frac, copy=False, **kwds)
    out.set_all()
    return out


def coord_trans(coords, old=None, new=None, copy=True, axis=-1):
    """General-purpose n-dimensional coordinate transformation. `coords` can
    have arbitrary dimension, i.e. it can contain many vectors to be
    transformed at once. But `old` and `new` must have ndim=2, i.e. only one
    old and new coord sys for all vectors in `coords`. 
    
    The most general case is that you want to transform an MD trajectory from a
    variable cell run, you have smth like this:

        | coords.shape = (nstep,natoms,3)
        | old.shape/new.shape = (nstep,3,3)
    
    You have a set of old and new coordinate systems at each step. 
    Then, use a loop over all time steps and call this function nstep times.
    See also coord_trans3d().

    Parameters
    ----------
    coords : array (d0, d1, ..., M) 
        Array of arbitrary rank with coordinates (length M vectors) in old
        coord sys `old`. The only shape resiriction is that the last dim must
        equal the number of coordinates (coords.shape[-1] == M == 3 for normal
        3-dim x,y,z).

        | 1d : trivial, transform that vector (length M)
        | 2d : The matrix must have shape (N,M), i.e. N vectors to be 
        |      transformed are the *rows*.
        | 3d : coords must have shape (..., M)
        
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

    Returns
    -------
    array of shape = coords.shape, coordinates in system `new`
    
    Examples
    --------
    >>> # Taken from [1]_.
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
    >>> # 3d example
    >>> c_old = np.random.rand(30,200,3)
    >>> old = np.random.rand(3,3)
    >>> new = np.random.rand(3,3)
    >>> c_new = coord_trans(c_old, old=old, new=new)
    >>> c_old2 = coord_trans(c_new, old=new, new=old)
    >>> np.testing.assert_almost_equal(c_old, c_old2)
    >>> # If you have an array of shape, say (10,3,100), i.e. the last
    >>> # dimension is NOT 3, then use numpy.swapaxes() or axis:
    >>> coord_trans(arr, old=..., new=..., axis=1)
    >>> coord_trans(arr.swapaxes(1,2), old=..., new=...).swapaxes(1,2)

    References
    ----------
    .. [1] http://www.mathe.tu-freiberg.de/~eiermann/Vorlesungen/HM/index_HM2.htm, Ch.6
    
    See Also
    --------
    coord_trans3d
    """ 
     
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

    Parameters
    ----------
    coords : 3d array 
        one axis (`axis`) must have length-M vectors, another (`timeaxis`) must
        be length `nstep`
    old,new : 2d arrays, two axes must be of equal length
    copy : see coord_trans()
    axis : axis where length-M vecs are placed if the timeaxis is removed
    timeaxis : time axis along which 2d arrays are aligned

    Examples
    --------
    | M = 3
    | coords :  (nstep,natoms,3)
    | old,new : (nstep,3,3)
    | timeaxis = 0
    | axis = 1 == -1 (remove timeaxis -> 2d slices (natoms,3) and (3,3) -> axis=1)
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
    """Apply minimum image convention to differences of fractional coords. 

    Handles also cases where coordinates are separated by an arbitrary number
    of periodic images.
    
    Parameters
    ----------
    sij : ndarray
        Differences of fractional coordinates, usually (natoms, natoms, 3),
        i.e. a, "matrix" of distance vectors, obtained by smth like
        ``sij = coords_frac[:,None,:] - coords_frac[None,:,:]`` where
        ``coords_frac.shape = (natoms,3)``.
    copy : bool, optional

    Returns
    -------
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
    """Calculate rmax as in [Smith]_, where rmax = the maximal distance up to
    which minimum image nearest neighbor distances are correct.
    
    The cell vecs must be the rows of `cell`.

    Parameters
    ----------
    %(cell_doc)s

    Returns
    -------
    rmax : float

    References
    ----------
    .. [Smith] W. Smith, The Minimum Image Convention in Non-Cubic MD Cells,
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

def rpdf(trajs, dr=0.05, rmax='auto', amask=None, tmask=None, 
         dmask=None, pbc=True, norm_vmd=False, maxmem=2.0):
    """Radial pair distribution (pair correlation) function.
    Can also handle non-orthorhombic unit cells (simulation boxes). 
    Only fixed-cell MD at the moment.

    Notes
    -----
    rmax : The maximal `rmax` for which g(r) is correctly normalized is the
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

    Parameters
    ----------
    trajs : Structure or Trajectory ot list of one or two such objects
        The case len(trajs)==1 is the same as providing the
        object directly (most common case). Internally we expand the input to
        [trajs, trajs], i.e. the RPDF of the 2nd coord set w.r.t. to the first
        is calculated -- the order matters! This is like selection 1 and 2 in
        VMD, but nornmally you would use `amask` instead. The option to provide
        a list of two Trajectory objects exists for cases where you don't want
        to use `amask`, but create two different Trajectory objects outside.
    dr : float, optional
        Radius spacing. Must have the same unit as `cell`, e.g. Angstrom.
    rmax : {'auto', float}, optional
        Max. radius up to which minimum image nearest neighbors are counted.
        For cubic boxes of side length L, this is L/2 [AT,MD].

        | 'auto' : the method of [Smith] is used to calculate the max. sphere
        |     raduis for any cell shape
        | float : set value yourself
    amask : None, list of one or two bool 1d arrays, list of one or two
        strings, optional
        Atom mask. This is the complementary functionality to `sel` in
        vmd_measure_gofr(). If len(amask)==1, then we expand to [amask, amask]
        internally, which would calculate the RPDF between the same atom
        selection. If two masks are given, then the first is applied to
        trajs[0] and the second to trajs[1]. Use this to select only certain
        atoms in each Trajectory. The default is to provide bool arrays. If you
        provide strings, they are assumed to be atom names and we create a
        bool array ``np.array(symbols) == amask[i]``.
    tmask : None or slice object, optional
        Time mask. Slice for the time axis, e.g. to use only every 100th step,
        starting from step 2000 to the end, use ``tmask=slice(2000,None,100)``,
        which is the same as ``np.s_[2000::100]``.
    dmask : None or string, optional
        Distance mask. Restrict to certain distances using numpy syntax for
        creating bool arrays::

            '>=1.0'
            '{d} >=1.0' # the same
            '({d} > 1.0) & ({d} < 3.0)'
        
        where ``{d}`` is a placeholder for the distance array (you really have to
        use ``{d}``). The placeholder is optional in some pattern. This is similar
        to VMD's "within" (pbc=False) or "pbwithin" (pbc=True) syntax. 
    pbc : bool, optional
        apply minimum image convention to distances
    norm_vmd : bool, optional
        Normalize g(r) like in VMD by counting duplicate atoms and normalize to
        (natoms0 * natoms1 - duplicates) instead of (natoms0*natoms1). Affects
        all-all correlations only. num_int is not affected. Use this only for
        testing.
    maxmem : float, optional
        Maximal allowed memory to use, in GB.

    Returns
    -------
    array (len(rad), 3), the columns are
    rad : 1d array 
        radius (x-axis) with spacing `dr`, each value r[i] is the middle of a
        histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),) 
        the (averaged) number integral ``number_density*hist*4*pi*r**2.0*dr``
    
    Notes
    -----
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
    doesn't change much, there is no need to use every step. See also `maxmem`.

    Examples
    --------
    >>> # simple all-all RPDF
    >>> d = rpdf(traj, dr=0.1)

    >>> # 2 selections: RPDF of all H's around all O's, average time step 3000 to
    >>> # end, take every 50th step
    >>> traj = parse.CpmdMDOutputFile(...).get_traj() # or io.read_cpmd_md(...)
    >>> d = rpdf(traj, dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50])
    >>> plot(d[:,0], d[:,1], label='g(r)')
    >>> twinx()
    >>> plot(d[:,0], d[:,2], label='number integral')
    
    >>> # the same as rpdf(traj,...)
    >>> d = rpdf([traj], ...)
    >>> d = rpdf([traj, traj], ...)
    
    >>> # use bool arrays for `amask`, need this for more complicated pattern
    >>> sy = np.array(traj.symbols)
    >>> # VMD: sel1='name O', sel2='name H'
    >>> d = rpdf(traj, dr=0.1, amask=[sy=='O', sy=='H'],tmask=np.s_[3000::50])
    >>> # VMD: sel1='name O', sel2='name H Cl', note that the bool arrays must
    >>> # be logically OR'ed (| operator) to get the ffect of "H and Cl"
    >>> d = rpdf(traj, dr=0.1, amask=[sy=='O', (sy=='H') | (sy=='Cl')],tmask=np.s_[3000::50])
    
    >>> # skip distances >1 Ang
    >>> d = rpdf(traj, dr=0.1, amask=['O', 'H'],tmask=np.s_[3000::50]
    ...          dmask='{d}>1.0')
     
    References
    ----------
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
    # result sij, which is a 4d array. For natoms=100, nstep=1e5, we already
    # have a 24 GB array in RAM! The only solution is to code this section
    # using Fortran/Cython/whatever in loops:
    #   * distances
    #   * apply min_image_convention() (optional)
    #   * sij -> rij transform
    #   * redcution to distances
    # 
    # Variable cell
    # -------------
    # Currently, we allow only fixed cell data b/c then we can use numpy
    # broadcasting to convert fractional to cartesian coords. But if we
    # implement the distance calculation in Fortran, we can easily allow
    # variable cell b/c then, we explicitely loop over time steps and can
    # perform the conversion at every step.
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
    nstep = clst[0].shape[0]
    rmax_auto = rmax_smith(cell)
    if rmax == 'auto':
        rmax = rmax_auto
    bins = np.arange(0, rmax+dr, dr)
    rad = bins[:-1]+0.5*dr
    volume_shells = 4.0/3.0*pi*(bins[1:]**3.0 - bins[:-1]**3.0)
    norm_fac_pre = volume / volume_shells
    
    if nstep * natoms0 * natoms1 * 24.0 / 1e9 > maxmem:
        raise StandardError("would use more than maxmem=%f GB of memory, "
                            "try `tmask` to reduce time steps" %maxmem)

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
    out[:,1] = hist_sum / float(nstep)
    out[:,2] = number_integral_sum / float(nstep)
    return out


def call_vmd_measure_gofr(trajfn, dr=None, rmax=None, sel=['all','all'],
                          fntype='xsf', first=0, last=-1, step=1, usepbc=1,
                          datafn=None, scriptfn=None, logfn=None, tmpdir=None,
                          verbose=False):
    """Call VMD's "measure gofr" command. This is a simple interface which does
    in fact the same thing as the gofr GUI, only scriptable. Accepts a file
    with trajectory data.

    Only orthogonal boxes are allowed (like in VMD).
    
    Parameters
    ----------
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
    first, last, step : int, optional
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

    Returns
    -------
    array (len(rad), 3), colums 0,1,2:

    rad : 1d array 
        radius (x-axis) with spacing `dr`, each value r[i] is the middle of a
        histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),) 
        the (averaged) number integral ``number_density*hist*4*pi*r**2.0*dr``
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
    
    Parameters
    ----------
    traj : Structure or Trajectory
    dr : float
        dr in Angstrom
    rmax : {'auto', float}, optional
        Max. radius up to which minimum image nearest neighbors are counted.
        For cubic boxes of side length L, this is L/2 [AT,MD].

        | 'auto' : the method of [Smith] is used to calculate the max. sphere
        |          raduis for any cell shape
        | float : set value yourself
    sel : list of two strings, optional
        string to select atoms, ["name Ca", "name O"], ["all", "all"], ...,
        where sel[0] is selection 1, sel[1] is selection 2 in VMD
    first,last,step : int, optional
        Select which MD steps are averaged. Like Python, VMD starts counting at
        zero. Last is -1, like in Python. 
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

    
    Returns
    -------
    array (len(rad), 3), colums 0,1,2:
    rad : 1d array 
        radius (x-axis) with spacing `dr`, each value r[i] is the middle of a
        histogram bin 
    hist : 1d array, (len(rad),)
        the function values g(r)
    num_int : 1d array, (len(rad),) 
        the (averaged) number integral ``number_density*hist*4*pi*r**2.0*dr``
    """
    # Need to import here b/c of cyclic dependency crys -> io -> crys ...
    from pwtools import io
    traj = struct2traj(traj)
    # Speed: The VMD command "measure gofr" is multithreaded and written in C.
    # That's why it is faster than the pure Python rpdf() above when we have to
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
    io.write_axsf(trajfn, traj2)
    ret = call_vmd_measure_gofr(trajfn, dr=dr, rmax=rmax, sel=sel, 
                                fntype='xsf', first=first,
                                last=last, step=step, usepbc=usepbc,
                                verbose=verbose,tmpdir=tmpdir)
    return ret


def distances(struct, pbc=False, squared=False, fullout=False):
    """
    Wrapper for _flib.distsq_frac(). Calculate distances of all atoms in
    `struct`.

    Parameters
    ----------
    struct : Structure instance
    pbc : bool, optional
        Apply PBC wrapping to distances (minimum image distances)
    squared : bool, optional
        Return squared distances
    fullout : bool
        See below

    Returns
    -------
    dists : if fullout=False
    dists, distvecs, distvecs_frac : if fullout=True
    dists : 2d array (natoms, natoms)
        (Squared, see `squared` arg) distances. Note that ``dists[i,j] ==
        dists[j,i]``.
    distvecs : (natoms,natoms,3)
        Cartesian distance vectors.
    distvecs_frac : (natoms,natoms,3)
        Fractional distance vectors.
    """
    # numpy version (10x slower):
    #
    # cf = struct.coords_frac
    # cell = struct.cell
    # distvecs_frac = cf[:,None,:] - cf[None,:,:]
    # if pbc:
    #     distvecs_frac = min_image_convention(distvecs_frac)
    # distvecs = np.dot(distvecs_frac, cell)
    # distsq = (distvecs**2.0).sum(axis=2)
    # dists = np.sqrt(distsq)
    nn = struct.natoms
    distsq = fempty((nn,nn))
    distvecs = fempty((nn,nn,3))
    distvecs_frac = fempty((nn,nn,3))
    _flib.distsq_frac(coords_frac=struct.coords_frac, 
                      cell=struct.cell, 
                      pbc=int(pbc),
                      distsq=distsq,
                      distvecs=distvecs,
                      distvecs_frac=distvecs_frac)
    dists = distsq if squared else np.sqrt(distsq)
    if fullout:
        return dists, distvecs, distvecs_frac
    else:        
        del distvecs
        del distvecs_frac
        return dists


def distances_traj(traj, pbc=False):
    """Cartesian distances along a trajectory.

    Wrapper for _flib.distances_traj().
    
    Parameters
    ----------
    traj : Trajectory
    pbc : bool
        Use minimum image distances.
    
    Returns
    -------
    dists : (nstep, natoms, natoms)
    """
    nn = traj.natoms
    dists = fempty((traj.nstep,nn,nn))
    _flib.distances_traj(coords_frac=np.asarray(traj.coords_frac, order='F'), 
                         cell=np.asarray(traj.cell, order='F'), 
                         pbc=int(pbc),
                         dists=dists)
    return dists


def angles(struct, pbc=False, mask_val=999.0, deg=True):
    """
    Wrapper for _flib.angles(), which accepts a Structure. 
    Calculate all angles between atom triples in `struct`.

    Parameters
    ----------
    struct : Structure instance
    pbc : bool, optional
        Apply PBC wrapping to distances (minimum image distances)
    mask_val : float     
        Fill value for ``anglesijk[ii,jj,kk]`` where ``ii==jj`` or ``ii==kk``
        or ``jj==kk``, i.e. no angle defined. Can be used to create bool mask
        arrays in numpy. Should be outside of [-1,1] (``deg=False``) or [0,180]
        (``deg=True``).
    deg : bool
        Return angles in degree (True) or cosine values (False).

    Returns
    -------
    anglesijk : 3d array (natoms,natoms,natoms)
        All angles. See also `mask_val`.

    Examples
    --------
    >>> natoms = struct.natoms
    >>> mask_val = 999
    >>> anglesijk = crys.angles(struct, mask_val=mask_val)
    >>> # angleidx holds all ii,jj,kk triples which we would get from:
    >>> angleidx = []
    ... for ii in range(natoms):
    ...     for jj in range(natoms):
    ...         for kk in range(natoms):
    ...             if (ii != jj) and (ii != kk) and (jj != kk):
    ...                 angleidx.append([ii,jj,kk])
    >>> # which is the same as
    >>> angleidx2 = [x for x in itertools.permutations(range(natoms),3)]
    >>> # or 
    >>> angleidx3 = np.array(zip(*(anglesijk != mask_val).nonzero()))
    >>> # the number of valid angles
    >>> len(angleidx) == natoms * (natoms - 1) * (natoms - 2)
    >>> len(angleidx) == factorial(natoms) / factorial(natoms - 3)
    >>> # angles in 1d array for histogram or whatever
    >>> angles1d = anglesijk[anglesijk != mask_val]
    >>> y,x = np.histogram(angles1d, bins=100)
    >>> plot(x[:-1]+0.5*(x[1]-x[0]), y)
    """
    if deg:
        assert not (0 <= mask_val <= 180), "mask_val must be outside [0,180]"
    else:        
        assert not (-1 <= mask_val <= 1), "mask_val must be outside [-1,1]"
    nn = struct.natoms
    dists, distvecs, distvecs_frac = distances(struct, pbc=pbc, squared=False,
                                               fullout=True) 
    del distvecs_frac
    anglesijk = fempty((nn,nn,nn))
    _flib.angles(distvecs=distvecs,
                 dists=dists,
                 mask_val=mask_val,
                 deg=int(deg),
                 anglesijk=anglesijk)
    return anglesijk


def nearest_neighbors_from_dists(dists, symbols, idx=None, skip=None,
                                 cutoff=None, num=None, pbc=True, 
                                 sort=True, fullout=False):
    """Core part of nearest_neighbors(), which accepts pre-calculated
    distances. 
    
    Can be more efficient in loops where many different
    nearest neighbors should be calculated from the same distances.

    Parameters
    ----------
    dists : 2d array (natoms,natoms)
        Cartesian distances (see distances()).
    symbols : sequence of strings (natoms,)  
        Atom symbols, i.e. struct.symbols

    Rest see nearest_neighbors().    
    """
    assert idx != None, "idx is None"
    assert None in [num,cutoff], "use either num or cutoff"
    # dists: distance matrix (natoms, natoms), each row or col is sorted like
    # struct.symbols
    # 
    # dist from atom `idx` to all atoms, same as dists[idx,:] b/c `dist` is
    # symmetric
    dist1d = dists[:,idx]
    # order by distance, `idx` first with dist=0
    idx_lst_sort = np.argsort(dist1d)
    dist1d_sort = dist1d[idx_lst_sort]
    symbols_sort = np.array(symbols)[idx_lst_sort]
    skip = common.asseq(skip)
    if skip != [None]:
        msk = symbols_sort == skip[0]
        for item in skip[1:]:
            msk = msk | (symbols_sort == item)
        only_msk = np.invert(msk)
    else:            
        only_msk = np.ones((len(symbols_sort),), dtype=bool)
    if cutoff is None:
        # ``1:`` : central atom excluded
        cut_msk = np.s_[1:num+1]
        ret_idx = idx_lst_sort[only_msk][cut_msk]
    else:
        cut_msk = (dist1d_sort > 0) & (dist1d_sort < cutoff)
        ret_idx = idx_lst_sort[cut_msk & only_msk]
    if not sort:
        orig_idx = np.arange(len(dist1d))
        ret_idx = orig_idx[match_mask(orig_idx,ret_idx)]
    if fullout:
        return ret_idx, dist1d[ret_idx]
    else:
        return ret_idx


def nearest_neighbors(struct, idx=None, skip=None, cutoff=None, num=None, pbc=True,
                      sort=True, fullout=False):
    """Indices of the nearest neighbor atoms to atom `idx`, skipping atoms
    whose symbols are `skip`.

    Parameters
    ----------
    struct : Structure
    idx : int
        Atom index of the central atom.
    skip : str or sequence of strings
        Symbol(s) of the atoms to skip.
    num : int
        number of requested nearest neighbors
    cutoff : float
        Cutoff radius in unit defined in `struct`, e.g. Angstrom. Return all
        neighbors within that radius. Use either `num` of `cutoff`.
    pbc : bool
        Apply PBC to distances.
    sort : bool
        Sort `nn_idx` and `nn_dist` by distance.     
    fullout : bool
        See below.

    Returns
    -------
    nn_idx : fullout=False
    nn_idx,nn_dist : fullout=True
    nn_idx : 1d array
        Indices into struct.symbols / coords.
    nn_dist : 1d array
        Distances ordered as in `nn_idx`.
    
    See Also
    --------
    num.match_mask

    Notes
    -----
    `num` : Depending on `struct`, there may not be `num` nearest neighbors,
    especially if you use `skip` to leave certain species out. Then the
    number of returned indices may be less then `num`.
    
    Ordering : If ``sort=True``, then returnd indices `nn_idx` and distances
    `nn_dist` are sorted small -> high. If ``sort=False``, then they are in the
    same order as the symbols in ``struct.symbols``.
    For structs with high symmetry (i.e. bulk crystals) where many
    nearest neighbors have the same distance from the central atom, the
    ordering of depends on how ``numpy.argsort`` sorts equal values in an
    array.

    Examples
    --------
    >>> ni=nearest_neighbors(struct, idx=struct.symbols.index('Ca'), num=6, skip='H')
    >>> ni=nearest_neighbors(struct, idx=23, cutoff=5.3, skip=['H','Cl'])
    >>> # simple rock salt example (used ASE to build dummy struct)
    >>> from ase import lattice
    >>> at=lattice.bulk('AlN', a=4, crystalstructure='rocksalt')
    >>> st=crys.atoms2struct(at); st=crys.scell(st,(2,2,2))
    >>> ni,nd=crys.nearest_neighbors(st, idx=0, num=8, fullout=True)  
    >>> ni
    array([ 9, 10, 11, 12, 13, 14,  1,  2])
    >>> nd
    [ 2. 2. 2. 2. 2. 2. 2.82842712 2.82842712]
    >>> # Use `ni` or bool array created from that for indexing
    >>> array(st.symbols)[ni]
    array(['Al', 'Al', 'N', 'N', 'N', 'N', 'N', 'N'], dtype='|S2')
    >>> msk=num.match_mask(arange(st.natoms), ni)
    >>> array(st.symbols)[msk]
    array(['Al', 'Al', 'N', 'N', 'N', 'N', 'N', 'N'], dtype='|S2')
    >>> # If you have many different symbols to skip and you don't want to type
    >>> # a longish `skip` list, then use smth like this to include only 'O'
    >>> # for example
    >>> symbols=['Ca', 'Cl', 'Cl'] + ['O']*10 + ['H']*20
    >>> skip=filter(lambda x: x!='O', set(symbols))
    >>> ['H', 'Ca', 'Cl']
    """
    # Distance matrix (natoms, natoms). Each row or col is sorted like
    # struct.symbols. If used in loops over trajs, the distances() call is the
    # most costly part, even though coded in Fortran.
    dists = distances(struct, pbc=pbc)
    return nearest_neighbors_from_dists(dists=dists, symbols=struct.symbols, idx=idx,
                                        skip=skip, cutoff=cutoff, num=num, 
                                        sort=sort, fullout=fullout)


def nearest_neighbors_struct(struct, **kwds):
    """Return Structure with only nearest neighbors. 

    Calls ``nearest_neighbors()`` and takes the same arguments. The returned
    Structure contains the central atom set by the `idx` keyword to
    nearest_neighbors().

    Examples
    --------
    >>> from pwtools import crys, visualize
    >>> st = crys.nearest_neighbors_struct(struct, cutoff=3.3, skip='H')
    >>> visualize.view_avogadro(st)
    """
    ni = nearest_neighbors(struct, **kwds)
    # include `idx` atom
    ni = np.concatenate((ni, [kwds['idx']]))
    msk = num.match_mask(np.arange(struct.natoms), ni)
    new_struct = Structure(coords_frac=struct.coords_frac[msk,:],
                           cell=struct.cell,
                           symbols=np.array(struct.symbols)[msk].tolist())
    return new_struct                           


#-----------------------------------------------------------------------------
# Container classes for crystal structures and trajectories.
#-----------------------------------------------------------------------------

class UnitsHandler(FlexibleGetters):
    def __init__(self):
        # XXX cryst_const is not in 'length' and needs to be treated specially,
        # see _apply_units_raw()
        
        # map physical quantity to variable names in Structure/Trajectory
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
        # Init all unit factors in self.units to 1.0
        self.units = self._default_units.copy()

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

    def update_units(self, units):
        """Update self.units dict from `units`. All units not contained in
        `units` remain at the default (1.0), see self._default_units.
        
        Parameters
        ----------
        units : dict, {'length': 5, 'energy': 30, ...}
        """
        if units is not None:
            all_units = self.units_map.keys()
            for key in units.keys():
                if key not in all_units:
                    raise StandardError("unknown unit: %s" %str(key))
            self.units.update(units)


class Structure(UnitsHandler):
    """Container class for representing a single crystal structure (unit
    cell + atoms).
    
    Derived classes may add attributes and getters but the idea is that this
    class is the minimal API for how to pass an atomic structure around.
    
    Units are supposed to be similar to ASE:

    =========== ==============  ===============================
    what        unit            SI
    =========== ==============  ===============================
    length      Angstrom        (1e-10 m)
    energy      eV              (1.602176487e-19 J)
    forces      eV / Angstrom
    stress      GPa             (not eV/Angstrom**3)
    temperature K             
    velocity    Angstrom / fs
    time        fs              (1e-15 s)
    mass        amu             (1.6605387820000001e-27 kg)
    =========== ==============  ===============================
    
    Unit conversion factors, which are applied to input arguments for
    conversion to the above units can be given by the `units` input keyword.

    Note that we cannot verify the unit of input args to the constructor, but
    all functions in this package, which use Structure / Trajectory as
    container classes, assume these units.

    This class is very much like ase.Atoms, but without the "calculators".
    You can use :meth:`get_ase_atoms` to get an Atoms object or
    :meth:`get_fake_ase_atoms` for a minimal Atoms-like object.
    
    Examples
    --------
    >>> symbols=['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> coords_frac=rand(len(symbols),3)
    >>> cryst_const=np.array([5,5,5,90,90,90.0])
    >>> st=Structure(coords_frac=coords_frac, 
    ...              cryst_const=cryst_const, 
    ...              symbols=symbols)
    >>> st.symbols
    ['N', 'Al', 'Al', 'Al', 'N', 'N', 'Al']
    >>> st.symbols_unique
    ['Al', 'N']
    >>> st.order
    {'Al': 1, 'N': 2}
    >>> st.typat
    [2, 1, 1, 1, 2, 2, 1]
    >>> st.znucl_unique
    [13, 7]
    >>> st.nspecies
    {'Al': 4, 'N': 3}
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
    >>> st.get_ase_atoms(pbc=True)
    Atoms(symbols='NAl3N2Al', positions=..., cell=[[2.64588604295, 0.0, 0.0],
    [1.6201379367036871e-16, 2.64588604295, 0.0], [1.6201379367036871e-16,
    1.6201379367036871e-16, 2.64588604295]], pbc=[True, True, True])
    """
    
    # attrs_nstep arrays have shape (nstep,...), i.e time along `timeaxis`
    timeaxis = 0
    
    is_traj = False
    is_struct = True
                
    def __init__(self, set_all_auto=True, units=None, **kwds):
        """
        Parameters
        ----------
        coords : (natoms, 3) [Ang]
            Cartesian coords.
            Optional if `coords_frac` given.
        coords_frac : (natoms, 3)
            Fractional coords w.r.t. `cell`.
            Optional if `coords` given.
        symbols : sequence of strings (natoms,)
            atom symbols
        cell : (3,3)
            Unit cell vectors as rows. [Ang]
            Optional if `cryst_const` given.
        cryst_const : (6,)
            [a,b,c,alpha,beta,gamma]; a,b,c in [Ang]
            Optional if `cell` given.
        forces : (natoms, 3), optional
            [eV/Ang]
        stress : (3,3), optional
            stress tensor [GPa]
        etot : float, optional 
            total energy [eV]
        units : optional, dict, 
            see :class:`UnitsHandler`
        set_all_auto : optional, bool
            Call :meth:`set_all` in :meth:`__init__`.
        
        Only Trajectory

        ekin : (nstep,)
            [eV]
        forces : (nstep,natoms,3)
            [eV/Ang]
        pressure : (nstep,)
            [GPa]
        stress : (nstep,3,3)
            [GPa]
        temperature : (nstep,)
            [K]
        timestep : float
            [fs]
        velocity : (nstep, natoms, 3)
            [Ang/fs]
        volume : (nstep,)
            [Ang^3]

        Notes
        -----
        cell, cryst_const : Provide either `cell` or `cryst_const`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See {cell2cc,cc2cell}.
        coords, coords_frac : Provide either `coords` or `coords_frac`, or both
            (which is redundant). If only one is given, the other is calculated
            from it. See coord_trans().
        """
        # accepted by input, some derived if not given
        self.input_attr_lst = [\
            'cell',
            'coords',
            'coords_frac',
            'cryst_const',
            'ekin',
            'etot',
            'forces',
            'pressure',
            'stress',
            'symbols',
            'temperature',
            'timestep',
            'velocity',
            'volume',
            ]
        # not as input, only derived from input attrs            
        self.derived_attr_lst = [\
            'mass',
            'mass_unique',
            'natoms',
            'nspecies',
            'nstep',
            'ntypat',
            'order',
            'symbols_unique',
            'typat',
            'time',
            'znucl',
            'znucl_unique',
            ] 
        # If these are given as input args, then they must be 3d.
        self.attrs_nstep_3d = [\
            'coords', 
            'coords_frac', 
            'stress', 
            'forces',
            'velocity',
            'cell',         # can be 2d, see _extend()
            ]
        self.attrs_nstep_2d = [\
            'cryst_const',  # can be 1d, see _extend()
            ]
        self.attrs_nstep_1d = [\
            'pressure',
            'volume',   
            'etot',
            'ekin',
            'temperature',
            'time',
            ]
        self.attrs_only_traj = [\
            'nstep',
            'timestep',
            'time',
            'ekin',
            'velocity',
            'temperature',
            ]

        self.attrs_nstep = self.attrs_nstep_3d + self.attrs_nstep_2d + \
                           self.attrs_nstep_1d
        self.attrs_nstep_2d_3d = self.attrs_nstep_3d + self.attrs_nstep_2d
         
        # init all in self.attr_lst to None
        self.attr_lst = self.input_attr_lst + self.derived_attr_lst
        self.init_attr_lst()
        
        # hackish but virtually no overhead here: create Structure by deleting
        # stuff in attributes lists
        if self.is_struct:
            del self.attrs_nstep
            for name in self.attrs_only_traj:
                # while: for some reason list.remove() doesn't always work
                while name in self.attr_lst:
                    self.attr_lst.pop(self.attr_lst.index(name))
        
        super(Structure, self).__init__()
        self.np_array_t = type(np.array([1]))

        # for iteration
        self._index = -1

        # initialize the self.units dictionary with unit conversion factors,
        # used in self.apply_units()
        self.update_units(units)
        
        self.set_all_auto = set_all_auto
        
        # assign input args, overwrite default None
        #   self.foo = foo
        #   self.bar = bar
        #   ...
        for name in kwds.keys():
            assert name in self.input_attr_lst, \
                "illegal input arg: '%s', allowed: %s" %(name, str(self.input_attr_lst))
            # cell can be 2d and will be treated by _extend() later
            if self.is_traj and (name in self.attrs_nstep_3d) and \
               name != 'cell':
                assert kwds[name].ndim == 3, "input '%s' is not 3d" %name 
            setattr(self, name, kwds[name])

        # calculate all missing attrs if requested, their units are based on
        # the ones set above
        if self.set_all_auto:
            self.set_all()
    
    def set_all(self):
        """Extend arrays, apply units, call all getters."""
        self._extend_arrays_apply_units()
        super(Structure, self).set_all()
   
    def _extend_arrays_apply_units(self):
        self.apply_units()
        if self.is_traj:
            self._extend()

    def _extend(self):
        if self.check_set_attr('nstep'):
            if self.is_set_attr('cell'):
                self.cell = self._extend_cell(self.cell)
            if self.is_set_attr('cryst_const'):
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
    
    def compress(self, forget=['forces', 'stress', 
                               'coords','cryst_const'], dtype=np.float32):
        """Compress Trajectory by deleting unused or redundant attrs (see
        `forget`). Cast float arrays to `dtype`. float32 is usually quite OK
        for MD data.
        
        Parameters
        ----------
        forget : list
            Names of attributes to delete. They will be set to None.
        dtype : numpy dtype
        """
        for name in self.attr_lst:
            if name in forget:
                setattr(self, name, None)
            else:            
                attr = getattr(self, name)
                if (type(attr) == self.np_array_t) and (attr.dtype.kind == 'f') and \
                    attr.dtype != dtype:
                    setattr(self, name, attr.astype(dtype))

    def copy(self):
        """Return a copy of the inctance."""
        if self.is_struct:
            obj = Structure(set_all_auto=False)
        elif self.is_traj:
            obj = Trajectory(set_all_auto=False)
        # Copy attrs over
        for name in self.attr_lst:
            val = getattr(self, name)
            if val is None:
                setattr(obj, name, None)
            # dict.copy() is shallow, use deepcopy instead    
            elif hasattr(val, 'copy') and not isinstance(val, types.DictType):
                setattr(obj, name, val.copy())
            else:
                setattr(obj, name, copy.deepcopy(val))
        return obj           

    def get_velocity(self):
        """Calculate `velocity` from `coords` and `timestep` if
        `velocity=None`. 
        """ 
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
        if not self.is_set_attr('velocity'):
            if self.check_set_attr_lst(['coords', 'timestep']):
                return velocity_traj(self.coords, dt=self.timestep, axis=0,
                                     endpoints=True)
            else:
                return None
        else:
            return self.velocity
    
    def get_ekin(self):
        """ ekin [eV] """
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
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
        """ [K] """
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
        if not self.is_set_attr('temperature'):
            if self.check_set_attr_lst(['ekin', 'natoms']):
                return self.ekin * constants.eV / self.natoms / constants.kb * (2.0/3.0)
            else:
                return None
        else:
            return self.temperature

    def get_natoms(self):
        if self.is_traj:
            axis = 1
        else: 
            axis = 0
        if self.is_set_attr('symbols'):
            return len(self.symbols)
        elif self.is_set_attr('coords'):
            return self.coords.shape[axis]
        elif self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[axis]
        else:
            return None
    
    def get_coords(self):
        if self.is_struct:
            if not self.is_set_attr('coords'):
                if self.is_set_attr('coords_frac') and self.check_set_attr('cell'):
                    return np.dot(self.coords_frac, self.cell)
                else:
                    return None
            else:
                return self.coords
        else: 
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
                    return _flib.frac2cart_traj(self.coords_frac, self.cell)
                else:
                    return None
            else:
                return self.coords

    def get_coords_frac(self):
        if self.is_struct:
            if not self.is_set_attr('coords_frac'):
                if self.is_set_attr('coords') and self.check_set_attr('cell'):
                    return _flib.cart2frac(self.coords, self.cell)
                else:
                    return None
            else:
                return self.coords_frac
        else: 
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
                    return _flib.cart2frac_traj(self.coords, self.cell)                                    
                else:
                    return None
            else:
                return self.coords_frac
    
    def get_volume(self):
        if not self.is_set_attr('volume'):
            if self.check_set_attr('cell'):
                if self.is_traj:
                    return volume_cell3d(self.cell, axis=self.timeaxis)
                else:    
                    return volume_cell(self.cell)
            else:
                return None
        else:
            return self.volume
    
    def get_cell(self):
        if not self.is_set_attr('cell'):
            if self.is_set_attr('cryst_const'):
                if self.is_traj:
                    cc = self._extend_cc(self.cryst_const)
                    return cc2cell3d(cc, axis=self.timeaxis)
                else:
                    return cc2cell(self.cryst_const)
            else:
                return None
        else:
            return self.cell
    
    def get_cryst_const(self):
        if not self.is_set_attr('cryst_const'):
            if self.is_set_attr('cell'):
                if self.is_traj:
                    cell = self._extend_cell(self.cell)
                    return cell2cc3d(cell, axis=self.timeaxis)
                else:    
                    return cell2cc(self.cell)
            else:
                return None
        else:
            return self.cryst_const
    
    def get_pressure(self):
        if not self.is_set_attr('pressure'):
            if self.check_set_attr('stress'):
                if self.is_traj:
                    assert self.timeaxis == 0
                    return np.trace(self.stress,axis1=1, axis2=2)/3.0
                else:
                    return np.trace(self.stress)/3.0
            else:
                return None
        else:
            return self.pressure
    
    def get_time(self):
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
        else:            
            if self.check_set_attr_lst(['timestep', 'nstep']):
                return np.linspace(0, (self.nstep-1)*self.timestep, self.nstep)
            else:
                return None
                        
    def get_timestep(self):
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
        else:            
            return self.timestep

    def get_nstep(self):
        if self.is_struct:
            raise NotImplementedError("only in Trajectory")
        if self.is_set_attr('coords'):
            return self.coords.shape[self.timeaxis]
        elif self.is_set_attr('coords_frac'):
            return self.coords_frac.shape[self.timeaxis]
        else:
            return None
    
    def get_symbols(self):
        """List of atomic symbols."""
        return self.symbols
    
    def get_forces(self):
        """Forces."""
        return self.forces

    def get_stress(self):
        """Stress tensor"""
        return self.stress

    def get_etot(self):
        """Total anergy."""
        return self.etot

    def get_symbols_unique(self):
        """List of unique atom symbols. 
        
        ``[Al,N]`` if ``symbols=['Al']*10 + ['N']*10``.
        ``len(self.symbols_unique)`` = number of atomic species"""
        return np.unique(self.symbols).tolist() if \
            self.check_set_attr('symbols') else None

    def get_order(self):
        """Dict which maps ``symbols_unique`` to numbers, starting at 1.
        
        ``{'Al': 1, 'N':2, 'O': 3, 'Si': 4}`` for ``symbols=['Al']*5 + ['N']*5
        + ['O']*10 + ['Si']*20``.
        Can be used in mapping a atom "type" number to a symbol (e.g. in
        LAMMPS)."""
        if self.check_set_attr('symbols_unique'):
            return dict([(sym, num+1) for num, sym in
                         enumerate(self.symbols_unique)])
        else:
            return None

    def get_typat(self):
        """List of atom type integers in ``self.order``, same length as
        `symbols`.

        ``[1]*10 + [2]*10`` for ````symbols=['Al']*10 + ['N']*10``.
        """
        if self.check_set_attr_lst(['symbols', 'order']):
            return [self.order[ss] for ss in self.symbols]
        else:
            return None
    
    def get_znucl_unique(self):
        """Unique atomic numbers. 
        
        ``[13,7]`` for ``symbols = ['Al','Al','N',N']``.
        """
        if self.check_set_attr('symbols_unique'):
            return [atomic_data.numbers[sym] for sym in self.symbols_unique]
        else:
            return None

    def get_znucl(self):
        """All atomic numbers. 
        
        ``[13,13,7,7]`` for ``symbols = ['Al','Al','N',N']``.
        """
        if self.check_set_attr('symbols'):
            return [atomic_data.numbers[sym] for sym in self.symbols]
        else:
            return None

    def get_ntypat(self):
        """Number of atomic species.
        
        2 for ``symbols=['Al','Al','N',N']``.
        """
        if self.check_set_attr('symbols_unique'):
            return len(self.symbols_unique)
        else:
            return None
    
    def get_nspecies(self):
        """Dict with number of atoms per species."""
        if self.check_set_attr_lst(['order', 'typat']):
            return dict([(sym, self.typat.count(idx)) for sym, idx in 
                         self.order.iteritems()])
        else:
            return None
    
    def get_mass(self):
        """1D array of atomic masses in amu (atomic mass unit 1.660538782e-27
        kg as in periodic table). The order is the one from self.symbols."""
        if self.check_set_attr('symbols'):
            return np.array([atomic_data.pt[sym]['mass'] for sym in
                             self.symbols])
        else:
            return None
    
    def get_mass_unique(self):
        if self.check_set_attr('znucl_unique'):
            return np.array([atomic_data.masses[z] for z in self.znucl_unique])
        else:
            return None

    def get_ase_atoms(self, **kwds):
        """Return ASE Atoms object. 
        
        Obviously, you must have ASE installed. We use
        ``scaled_positions=self.coords_frac``, so only ``self.cell`` must be in
        [Ang].

        Parameters
        ----------
        **kwds : 
            additional keywords passed to the Atoms() constructor.
        
        See Also
        --------
        :meth:`get_fake_ase_atoms`

        Notes
        -----
        By default, we use ``Atoms(...,pbc=False)`` to avoid pbc-wrapping
        ``atoms.scaled_positions`` (we don't want that for MD structures, for
        instance). If you need the pbc flag in your Atoms object, then use::
        
        >>> # Note that the `pbc` flag is passed to ase.Atoms, so you can use
        >>> # whatever that accepts, like pbc=[1,1,1] etc. 
        >>> atoms=struct.get_ase_atoms(pbc=True)
        >>> # or 
        >>> atoms=struct.get_ase_atoms()
        >>> atoms.set_pbc(True) 

        but then, ``scaled_positions`` will be wrapped by ASE and I'm not sure
        if ``atoms.positions`` is updated in that case. Please test that -- I
        don't use ASE much.
        """
        req = ['coords_frac', 'cell', 'symbols']
        if self.check_set_attr_lst(req):
            # We don't wanna make ase a dependency. Import only when needed.
            from ase import Atoms
            _kwds = {'pbc': False}
            _kwds.update(kwds)
            at = Atoms(symbols=self.symbols,
                       scaled_positions=self.coords_frac,
                       cell=self.cell,
                       **_kwds)
            return at                
        else:
            return None
    
    def get_fake_ase_atoms(self):
        """:class:`FakeASEAtoms` instance representing this Structure."""
        return FakeASEAtoms(scaled_positions=self.get_coords_frac(),
                            cell=self.get_cell(),
                            symbols=self.get_symbols())

    def get_traj(self, nstep):
        """Return a Trajectory object, where this Structure is copied `nstep`
        times."""
        tr = Trajectory(set_all_auto=False)
        for attr_name in self.attr_lst:
            attr = getattr(self, attr_name)
            if attr is None:
                new_attr = None
            elif attr_name in tr.attrs_nstep:
                if type(attr) == self.np_array_t:
                    new_attr = num.extend_array(attr, nstep, axis=self.timeaxis)
                else:
                    new_attr = np.array([attr]*nstep)
            else:
                new_attr = copy.deepcopy(attr)
            setattr(tr, attr_name, new_attr)
        # re-calculate nstep    
        tr.nstep = None
        tr.set_all()    
        return tr

    
class Trajectory(Structure):
    """Like :class:`Structure`, but all attrs in `attrs_nstep` have a timeaxis
    along axis=0 and length `nstep`:
       
    ===========     ============    ================
    attribute       Structure       Trajectory
    ===========     ============    ================
    coords          (nstoms,3)      (nstep,natoms,3)
    coords_frac     (nstoms,3)      (nstep,natoms,3) 
    forces          (nstoms,3)      (nstep,natoms,3)     
    velocity        --              (nstep,natoms,3) 
    cryst_const     (6,)            (nstep,6)
    cell            (3,3)           (nstep,3,3)
    stress          (3,3)           (nstep,3,3)
    etot            scalar          (nstep,)
    volume          scalar          (nstep,)                 
    pressure        scalar          (nstep,)                 
    ekin            --              (nstep,)
    temperature     --              (nstep,)
    time            --              (nstep,)
    ===========     ============    ================
    
    Also, we have additional attrs which are only defined for
    :class:`Trajectory`, see `attrs_only_traj`: 
    
    | nstep
    | timestep
    | time
    | ekin
    | velocity
    | temperature

    """
    is_traj = True
    is_struct = False

    def __init__(self, *args, **kwds):
        super(Trajectory, self).__init__(*args, **kwds)
    
    def __iter__(self):
        return self
    
    def __getitem__(self, idx):
        want_traj = False
        if isinstance(idx, slice):
            obj = Trajectory(set_all_auto=False)
            timestep_fac = idx.step if idx.step is not None else 1.0
            want_traj = True
        else:            
            obj = Structure(set_all_auto=False)
            timestep_fac = None
        for name in self.attr_lst:
            if not want_traj and name in self.attrs_only_traj:
                continue
            attr = getattr(self, name)
            if attr is not None:    
                if name in self.attrs_nstep:
                    # the timeaxis check may be a problem for parsed MD data
                    # where some arrays are 1 or two steps longer/shorter than
                    # coords (from which we get nstep), for example lammps:
                    # temperature, volume, etc can be longer if multiple runs
                    # are done from the same input file and the parser
                    # currently doesn't handle that
                    if name in self.attrs_nstep_2d_3d \
                        and attr.shape[self.timeaxis] == self.nstep:
                        setattr(obj, name, attr[idx,...])
                    elif name in self.attrs_nstep_1d \
                        and attr.shape[self.timeaxis] == self.nstep:
                        setattr(obj, name, attr[idx])
                else:                        
                    setattr(obj, name, attr)
            else:                    
                setattr(obj, name, None)
        # After possible slicing, calculate new nstep
        if want_traj:
            obj.nstep = obj.get_nstep()
            if obj.is_set_attr('timestep'):
                obj.timestep *= timestep_fac
        return obj
    
    def next(self):
        self._index += 1
        if self._index == self.nstep:
            self._index = -1
            raise StopIteration
        else:
            return self[self._index]
    
    def get_ase_atoms(self):
        raise NotImplementedError("only in Structure")

    def get_fake_ase_atoms(self):
        raise NotImplementedError("only in Structure")

    def get_traj(self):
        raise NotImplementedError("only in Structure")


def compress(traj, copy=True, **kwds):
    """Wrapper for :meth:`Trajectory.compress`. 

    Parameters
    ----------
    copy : bool
        Return compressed copy or in-place modified object.
    **kwds : keywords
        keywords to :meth:`Trajectory.compress`

    Examples
    --------
    >>> trc = compress(tr, copy=True, forget=['coords'])
    >>> trc.dump('very_small_file.pk')
    """
    if copy:
        out = traj.copy()
    else:
        out = traj
    out.compress(**kwds)
    return out


def atoms2struct(at):
    """Transform ASE Atoms object to Structure."""
    return Structure(symbols=at.get_chemical_symbols(),
                     cell=at.get_cell(),
                     coords_frac=at.get_scaled_positions())

def struct2atoms(st, **kwds):
    """Transform Structure to ASE Atoms object."""
    return st.get_ase_atoms(**kwds)

def struct2traj(obj):
    """Transform Structure to Trajectory with nstep=1."""
    if obj.is_traj:
        return obj
    else:
        return obj.get_traj(nstep=1)


class FakeASEAtoms(Structure):
    """Mimic the basic behavior of ``ase.Atoms``. 

    Used for ``spglib``.
    """
    def __init__(self, scaled_positions=None, cell=None, symbols=None):
        super(FakeASEAtoms, self).__init__(coords_frac=scaled_positions,
                                           cell=cell,
                                           symbols=symbols)
        self.get_scaled_positions = self.get_coords_frac
        self.get_positions = self.get_coords
        self.get_number_of_atoms = self.get_natoms

    def get_magnetic_moments(self):
        return None

    def get_atomic_numbers(self):
        return np.array(self.get_znucl())


def populated_attrs(lst):
    """Set with attr names which are not None in all objects in `lst`."""
    attr_lists = [[name for name in obj.attr_lst \
        if getattr(obj,name) is not None] for obj in lst]
    return set.intersection(*(set(x) for x in attr_lists))


def concatenate(lst):
    """Concatenate Structure or Trajectory objects into one Trajectory.
    
    For non-nstep attrs (symbols,...), the first item is used and no check is
    made whether they are the same in the others.

    Parameters
    ----------
    lst : sequence of Structure or Trajectory instances or both

    Returns
    -------
    tr : Trajectory
    """
    trlst = [struct2traj(obj) for obj in lst]
    traj = Trajectory(set_all_auto=False)
    com_attrs = populated_attrs(trlst)
    attr_lst = set.intersection(com_attrs, set(traj.attrs_nstep))
    for name in attr_lst:
        attr = np.concatenate(tuple(getattr(x,name) for x in trlst), 
                              axis=0)
        setattr(traj, name, attr)
    attrs_traj = traj.attrs_nstep + traj.attrs_only_traj                
    for name in set.symmetric_difference(com_attrs, set(attrs_traj)):
        setattr(traj, name, getattr(trlst[0], name))
    traj.timestep = None
    traj.time = None
    traj.nstep = traj.get_nstep()
    return traj                


def mean(traj):
    """Mean of Trajectory along `timeaxis`, like numpy.mean(array,axis=0).

    Parameters
    ----------
    traj : Trajectory

    Returns
    -------
    Structure : 
        instance with extra velocity, temperature, ekin attrs which can hold
        the mean of the input `traj`
    
    Examples
    --------
    >>> #  a slice of the Trajectory
    >>> st = mean(tr[200:500])
    >>> # Say we know that coords_frac is pbc-wrpapped for some reason but
    >>> # coords is not. Make sure that we average only coords and force a
    >>> # recalculation of coords_frac by setting it to None and calling
    >>> # set_all() at the end.
    >>> tr.coords_frac = None
    >>> st = mean(tr)
    >>> st.set_all()
    """
    assert traj.is_traj
    struct = Structure(set_all_auto=False)
    # add some non-Structure attrs like velocity,ekin,temperature 
    attrs_only_traj = ['time', 'timestep', 'nstep']
    extra = list(set.difference(set(traj.attrs_only_traj),
                                set(attrs_only_traj)))
    struct.attr_lst += extra                           
    for attr_name in set.difference(set(traj.attrs_nstep), 
                                    set(attrs_only_traj)):
        attr = getattr(traj, attr_name)
        if attr is not None:
            setattr(struct, attr_name, attr.mean(axis=traj.timeaxis))
    attrs_traj = traj.attrs_nstep + attrs_only_traj
    for attr_name in set.difference(set(traj.attr_lst),
                                    set(attrs_traj)):
        attr = getattr(traj, attr_name)
        if attr is not None:
            setattr(struct, attr_name, attr)
    return struct


def smooth(traj, kern, method=1):
    """Smooth Trajectory along `timeaxis`.

    Each array in `traj.attrs_nstep` is smoothed by convolution with `kern`
    along `timeaxis`, i.e. coords, coords_frac, etot, ... The kernel is only
    required to be a 1d array and is automatically broadcast to the shape of
    each array. A similar feature can be found in VMD -> Representations ->
    Trajectory.

    Parameters
    ----------
    traj : Trajectory
    kern : 1d array
        Convolution kernel (smoothing window, see :func:`~pwtools.signal.smooth`).
    method : int
        Choose how to do the convolution:

            | 1 : loops over 1d convolutions, easy on memory, sometimes faster
                  than method=2 (default)
            | 2 : up to 3d kernel by broadcasting, can be very memory hungry for
                  big `traj` (i.e. 1e5 timesteps, 128 atoms)
    
    Returns
    -------
    tr : Trajectory
        Has the same `nstep` and `timestep` as the input Trajectory.
    
    Examples
    --------
    >>> kern = scipy.signal.hann(101)
    >>> trs = smooth(tr, kern)
    """
    assert traj.is_traj
    assert kern.ndim == 1, "need 1d kernel"
    if traj.timeaxis == 0:
        kern1d = kern
        kern2d = kern[:,None]
        kern3d = kern[:,None,None]
    else:
        # ... but is trivial to add
        raise StandardError("timeaxis != 0 not implemented")
    out = Trajectory(set_all_auto=False)
    for attr_name in traj.attrs_nstep:
        attr = getattr(traj, attr_name)
        if attr is not None:
            if method == 1:
                # Remove that if we want to generalize to timeaxis != 0 and
                # adapt code below.
                if attr.ndim > 1:
                    assert traj.timeaxis == 0
                if attr.ndim == 1:
                    tmp = signal.smooth(attr, kern, axis=traj.timeaxis)
                elif attr.ndim == 2:
                    tmp = np.empty_like(attr)
                    for jj in range(attr.shape[1]):
                        tmp[:,jj] = signal.smooth(attr[:,jj], kern)
                elif attr.ndim == 3:
                    tmp = np.empty_like(attr)
                    for jj in range(attr.shape[1]):
                        for kk in range(attr.shape[2]):
                            tmp[:,jj,kk] = signal.smooth(attr[:,jj,kk], kern)
                else:
                    raise StandardError("ndim != 1,2,3 not allowed")
                setattr(out, attr_name, tmp)
            elif method == 2:
                if attr.ndim == 1:
                    krn = kern1d
                elif attr.ndim == 2:
                    krn = kern2d
                elif attr.ndim == 3:
                    krn = kern3d
                else:
                    raise StandardError("ndim != 1,2,3 not allowed")
                setattr(out, attr_name, signal.smooth(attr, krn,
                                                      axis=traj.timeaxis))
            else:
                raise StandardError("unknown method")
    # nstep and timestep are the same for the smoothed traj, so we can copy all
    # non-nstep attrs over
    for attr_name in set.difference(set(traj.attr_lst), 
                                    set(traj.attrs_nstep)):
        setattr(out, attr_name, getattr(traj, attr_name))
    return out


def mix(st1, st2, alpha):
    """Linear interpolation between two Structures based on the numbers in
    `alpha`. Returns a :class:`Trajectory`.
    
    Mix two structures as (1-alpha)*st1 + alpha*st2. `coords` and `cell` are
    used, as well as `forces` if present.

    Parameters
    ----------
    st1, st2 : Structures
    alpha : 1d sequence
        parameter values for mixing

    Returns
    -------
    tr : Trajectory
        tr.nstep == len(alpha)

    Examples
    --------
    >>> mix(st1, st2, linspace(0,1,50))
    """
    assert st1.coords.ndim == 2
    assert st1.cell.ndim == 2
    assert st1.coords.shape == st2.coords.shape
    assert st1.symbols == st2.symbols
    coords = np.empty_like(st1.coords)
    cell = np.empty_like(st1.cell)
    rr = alpha[:,None,None]
    coords = rr * st2.coords[None,:,:] + (1.0 - rr) * st1.coords[None,:,:]
    cell = rr * st2.cell[None,:,:] + (1.0 - rr) * st1.cell[None,:,:]
    if (st1.forces is not None) and (st2.forces is not None):
        forces = rr * st2.forces[None,:,:] + (1.0 - rr) * st1.forces[None,:,:]
        return Trajectory(coords=coords, cell=cell, symbols=st1.symbols,
                          forces=forces)
    else:
        # cannot use forces=None here, Structure.__init__ complains that it
        # is None ... this is by design but seems stupid -> change input
        # checking logic there
        return Trajectory(coords=coords, cell=cell, symbols=st1.symbols)


def align_cart(obj, x=None, y=None, vecs=None, indices=None, cart=None,
               eps=1e-5):
    """Align obj w.r.t. a new cartesian coord sys defined by x,y and
    z=cross(x,y).
    
    The new coord sys can be defined either by `x` + `y` or `vecs` or
    `indices` or `cart`. Vectors need not be normalized.

    Parameters
    ----------
    obj : Structure or Trajectory
    x, y : (3,)
        The two vectors spanning the x-y plane.
    vecs : (3,3)
        Array with 3 vectors as rows `[v0, v1, v2]` and ``x = v1 - v0``, 
        ``y = v2 - v0``
    indices : sequence (4,) or (3,)
        Indices of atoms in `obj` with positions `v0,v1,v2`. Length 4 for
        obj=Trajectory: ``indices=[time_step, idx0, idx1, idx2]`` and length 3
        for obj=Structure: ``[idx0, idx1, idx2]`` with
           | ``v0 = obj.coords[time_step, idx0, ...]`` (Trajectory)
           | ``v1 = obj.coords[time_step, idx1, ...]`` 
           | ``v2 = obj.coords[time_step, idx2, ...]`` 
        or
           | ``v0 = obj.coords[idx0, ...]`` (Structure)
           | ``v1 = obj.coords[idx1, ...]``
           | ``v2 = obj.coords[idx2, ...]``
    cart : (3,3)
        new cartesian coord sys ``[x,y,z]``, matrix must be orthogonal
    eps : float
        Threshold for orthogonality check. Use `eps <= 0` to disable the check.

    Returns
    -------
    out : Structure or Trajectory

    Notes
    -----
    In case of a :class:`Trajectory`, the same rotation is applied to all
    structs, so the *relative* orientation within the Trajectory is not
    changed. That is OK if each struct shall be rotated in the same way.
    If however each struct has a different orientation, then you need
    to loop over the Trajectory like::
    
    >>> from pwtools.crys import align_cart, concatenate
    >>> trnew = concatenate([align_cart(st, cart=...) for st in tr])
    """
    if cart is None:
        if [x,y] == [None,None]:
            if indices is None:
                v0 = vecs[0,:]    
                v1 = vecs[1,:]
                v2 = vecs[2,:]
            else:
                if len(indices) == 4:
                    v0 = obj.coords[indices[0], indices[1], ...]
                    v1 = obj.coords[indices[0], indices[2], ...]
                    v2 = obj.coords[indices[0], indices[3], ...]
                else:                
                    v0 = obj.coords[indices[0], ...]
                    v1 = obj.coords[indices[1], ...]
                    v2 = obj.coords[indices[2], ...]
            x = v1 - v0
            y = v2 - v0
        xx = x.copy() / norm(x)
        yy = y.copy() / norm(y)
        cart = np.array([xx, yy, np.cross(xx, yy)])
    if eps > 0:
        assert np.allclose(inv(cart), cart.T, atol=eps)
    if obj.is_traj:
        container = Trajectory
    else:
        container = Structure
    obj_new = container(coords_frac=obj.coords_frac.copy(),
                        symbols=obj.symbols,
                        cell=np.dot(obj.cell, cart.T),
                        )
    return obj_new                        


def tensor2voigt(tensor):
    """
    Parameters
    ----------
    tensor : (3,3)

    Returns
    -------
    voigt: [xx,yy,zz,yz,xz,xy]
    """
    assert tensor.shape == (3,3), "tensor must be (3,3)"
    voigt = np.empty(6)
    voigt[0] = tensor[0,0]
    voigt[1] = tensor[1,1]
    voigt[2] = tensor[2,2]
    voigt[3] = tensor[1,2]
    voigt[4] = tensor[0,2]
    voigt[5] = tensor[0,1]
    return voigt


def voigt2tensor(voigt):
    """
    Parameters
    ----------
    voigt: [xx,yy,zz,yz,xz,xy]

    Returns
    -------
    tensor : (3,3)
    """
    assert len(voigt) == 6, "voigt must be length 6 vector"
    tensor = np.empty((3,3))
    tensor[0,0]  = voigt[0] 
    tensor[1,1]  = voigt[1] 
    tensor[2,2]  = voigt[2] 
    tensor[1,2]  = voigt[3] 
    tensor[0,2]  = voigt[4] 
    tensor[0,1]  = voigt[5]
    tensor[2,1] = tensor[1,2]
    tensor[2,0] = tensor[0,2]
    tensor[1,0] = tensor[0,1]
    return tensor

def voigt2tensor3d(voigt):
    """
    Parameters
    ----------
    voigt: (nstep,6)

    Returns
    -------
    tensor : (nstep,3,3)
    """
    nstep = voigt.shape[0]
    assert voigt.ndim == 2, "voigt must be (nstep,6)"
    assert voigt.shape[1] == 6, "voigt must be (nstep,6)"
    tensor = np.empty((nstep,3,3))
    tensor[:,0,0]  = voigt[:,0] 
    tensor[:,1,1]  = voigt[:,1] 
    tensor[:,2,2]  = voigt[:,2] 
    tensor[:,1,2]  = voigt[:,3] 
    tensor[:,0,2]  = voigt[:,4] 
    tensor[:,0,1]  = voigt[:,5]
    tensor[:,2,1] = tensor[:,1,2]
    tensor[:,2,0] = tensor[:,0,2]
    tensor[:,1,0] = tensor[:,0,1]
    return tensor


def tensor2voigt3d(tensor):
    """
    Parameters
    ----------
    tensor : (nstep,3,3)

    Returns
    -------
    voigt: (nstep,6)
    """
    assert tensor.ndim == 3, "tensor must be (nstep,3,3)"
    assert tensor.shape[1:] == (3,3), "tensor must be (nstep,3,3)"
    nstep = tensor.shape[0]
    voigt = np.empty((nstep,6))
    voigt[:,0] = tensor[:,0,0]
    voigt[:,1] = tensor[:,1,1]
    voigt[:,2] = tensor[:,2,2]
    voigt[:,3] = tensor[:,1,2]
    voigt[:,4] = tensor[:,0,2]
    voigt[:,5] = tensor[:,0,1]
    return voigt

