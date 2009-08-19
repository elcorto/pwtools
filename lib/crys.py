# crys.py
#
# Crystal and unit-cell related tools tools.
#

import numpy as np
from math import acos, pi, sin, cos, sqrt

#-----------------------------------------------------------------------------

# np.linalg.norm handles also complex arguments, but we don't need that here. 
##norm = np.linalg.norm
def norm(a):
    """2-norm for real vectors."""
    _assert(len(a.shape) == 1, "input must be 1d array")
    # math.sqrt is faster then np.sqrt for scalar args
    return math.sqrt(np.dot(a,a))

#-----------------------------------------------------------------------------

def _add_doc(func):
    """Decorator to add common docstrings to functions."""
    dct = {}
    dct['align_doc'] = \
    """align: str
        'rows' : basis vecs are the rows of `arr`
        'cols' : basis vecs are the columns of `arr`"""
    dct['arr_doc'] = \
    """arr: array_like, shape (3,3)
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
def volume(arr, align='cols'):
    """Volume of unit cell. Calculates the triple product 
    np.dot(np.cross(a,b), c) of the basis vectors a,b,c contained 
    in `arr`.
    
    args:
    -----
    %(arr_doc)s
    %(align_doc)s
    """    
    if align == 'cols':
        arr = arr.T
    return np.dot(np.cross(arr[0,:], arr[1,:]), arr[2,:])        

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
def cp2crys_const(arr, align='rows'):
    """From CELL_PARAMETERS to crystallographic constants a, b, c, alpha, beta,
    gamma.
    
    args:
    -----
    %(arr_doc)s
    %(align_doc)s

    returns:
    --------
    %(cryst_const_doc)s

    notes:
    ------
    %(notes_cp_crys_const)s
    """
    arr = np.asarray(arr)
    assert arr.shape == (3,3), "arr must be (3,3) array"
    if align == 'cols':
        arr = arr.T
##    print "spat volume:", volume(arr)
    cryst_const = np.empty((6,), dtype=float)
    # a = |a|, b = |b|, c = |c|
    cryst_const[:3] = np.sqrt((arr**2.0).sum(axis=1))
    va = arr[0,:]
    vb = arr[1,:]
    vc = arr[2,:]
    # alpha
    cryst_const[3] = angle(vb,vc)
    # beta
    cryst_const[4] = angle(va,vc)
    # gamma
    cryst_const[5] = angle(va,vb)
    return cryst_const

#-----------------------------------------------------------------------------

@_add_doc
def crys_const2cp(cryst_const):
    """From crystallographic constants a, b, c, alpha, beta,
    gamma to CELL_PARAMETERS.
    
    args:
    -----
    %(cryst_const_doc)s
    
    returns:
    --------
    %(arr_doc)s
        Basis vecs are the rows.
    
    notes:
    ------
    %(notes_cp_crys_const)s
    """
    a = crys_const[0]
    b = crys_const[1]
    c = crys_const[2]
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
    #
    # Maxima solution
    #   
    # # volume of the unit cell, see http://en.wikipedia.org/wiki/Parallelepiped
    # vol = a*b*c*sqrt(1+ 2*cos(alpha)*cos(beta)*cos(gamma) -\
    #       cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 )
    # print "Maxima: vol", vol
    # cz = vol / (a*b*sin(gamma))
    # print "Maxima: cz", cz
    # cy = sqrt(a**2 * b**2 * c**2 * sin(beta)**2 * sin(gamma)**2 - \
    #     vol**2) / (a*b*sin(gamma))
    # print "Maxima: cy", cy
    # cy = sqrt(c**2 - cx**2 - cz**2)
    # print "Pythagoras: cy", cy
    
    # PWscf , WIEN2K's sgroup, results are the same as with Maxima but the
    # formulas are shorter.
    cy = c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma)
##    print "sgroup: cy", cy
    cz = sqrt(c**2 - cy**2 - cx**2)
##    print "sgroup: cz", cz

    vc = np.array([cx, cy, cz])
    return np.array([va, vb, vc])
