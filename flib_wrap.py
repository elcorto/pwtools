import numpy as np
from pwtools import _flib

def cdistsq(arrx, arry):
    """Squared distances between all points in `arrx` and `arry`:
        
        r_ij**2 = sum_k (arrx[i,k] - arry[j,k])**2.0
        i = 1..Mx
        j = 1..My
        k = 1..N

    This is like 
        scipy.spatial.distance.cdist(arrx, arry)**2.0
    
    This is a wrapper for _flib.cdistsq().

    args:
    -----
    arrx, arry : ndarray (Mx,N), (My,N)
        Mx (My) points in N-dim space
    
    returns:
    --------
    2d array (Mx,My)
    """        
    nx, ny = arrx.shape[0], arry.shape[0]
    ndim = arrx.shape[1]
    assert arrx.shape[1] == arry.shape[1]
    # Allocating in F-order is essential for speed! For many points, this step
    # is actually the bottleneck, NOT the Fortran code! This is b/c if `dist`
    # is order='C' (numpy default), then the f2py wrapper makes a copy of the
    # array before starting to crunch numbers.
    dist = np.empty((nx, ny), dtype=arrx.dtype, order='F')
    return _flib.cdistsq(arrx, arry, dist, nx, ny, ndim)
