# crys.py
#
# Crystal and unit-cell related tools.
#

import numpy as np
from math import acos, pi, sin, cos, sqrt
from common import assert_cond

#-----------------------------------------------------------------------------

# np.linalg.norm handles also complex arguments, but we don't need that here. 
##norm = np.linalg.norm
def norm(a):
    """2-norm for real vectors."""
    assert_cond(len(a.shape) == 1, "input must be 1d array")
    # math.sqrt is faster then np.sqrt for scalar args
    return sqrt(np.dot(a,a))

#-----------------------------------------------------------------------------

def _add_doc(func):
    """Decorator to add common docstrings to functions."""
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
    $(notes_cp_crys_const)s
    """    
    assert_cond(cp.shape == (3,3), "input must be (3,3) array")
    return np.dot(np.cross(cp[0,:], cp[1,:]), cp[2,:])        

#-----------------------------------------------------------------------------

def volume_cc(cryst_const):
    """Volume of the unit cell from crystallographic constants [1].
    
    args:
    -----
    %(cryst_const_doc)s
    
    notes:
    ------
    $(notes_cp_crys_const)s

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

def angle(x,y):
    """Angle between vectors `x' and `y' in degrees.
    
    args:
    -----
    x,y : 1d numpy arrays
    """
    # Numpy's `acos' is "acrcos", but we take the one from math for scalar
    # args.
    return acos(np.dot(x,y)/norm(x)/norm(y))*180.0/pi

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
    %(cryst_const_doc)s

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
    
    notes:
    ------
    %(notes_cp_crys_const)s
    """
    a = cryst_const[0]
    b = cryst_const[1]
    c = cryst_const[2]
    alpha = cryst_const[3]*pi/180
    beta = cryst_const[4]*pi/180
    gamma = cryst_const[5]*pi/180
    
    # Basis vectors fulfilling the crystallographic constants are arbitrary
    # w.r.t. their orientation in space. We choose (as others also do):
    #
    # va along x axis
    va = np.array([a,0,0])
    # vb in x-y plane
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


