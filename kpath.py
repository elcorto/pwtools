import numpy as np
import sys
import math

from pwtools.common import str_arr


def vlinspace(a, b, num, endpoint=True):
    """Like numpy.linspace, but for 1d arrays. Generate uniformly spaced points
    (vectors) along the distance vector connecting a and b.
    
    args:
    -----
    a, b : 1d arrays
    num : int

    returns:
    --------
    array (num, len(a)), each row is a "point" between `a` and `b`
    """
    assert a.ndim == b.ndim == 1, "expect 1d arrays"
    assert len(a) == len(b), "`a` and `b` must have equal length"
    # distance vec connecting a and b
    dv = b-a
    if endpoint:
        ddv = dv/float(num-1)
    else:        
        ddv = dv/float(num)
    ret = np.empty((num, len(dv)), dtype=float)
    ret[...] = ddv
    ret[0,:] = a
    return np.cumsum(ret, axis=0)
    

def kpath(vecs, N=10):    
    """Simple k-path. Given a set of K vectors (special points in the BZ),
    generate a "fine path" of N*(K-1)+1 vectors along the path defined by the
    vectors in `vecs`. The K vectors are the "vertices" of the k-path and we
    construct the fine path by connecting the vertices by their distance
    vectors and placing N points on each connection edge.

    args:
    -----
    vecs: array (K,M) with K vectors of the Brillouin zone (so M = 3 usually :)
    N : int

    returns:
    --------
    new_vecs : array (N*(K-1)+1,M) with a fine grid of vectors along the path 
        defined by `vecs`
    
    notes:
    ------
    This is the simplest method one can think of. Points on the "fine path" are
    not equally distributed. The distance between 2 vertices (k-points) doesn't
    matter, you will always get N points between them. For a smooth dispersion
    plot, you need N=20 or more.
    """
    nvecs = vecs.shape[0]
    nnew = (nvecs-1)*N+1
    new_vecs = np.empty((nnew, vecs.shape[1]), dtype=float)
    for i in range(1, nvecs):
        new_vecs[(i-1)*N:i*N, :] = vlinspace(vecs[i-1,:], vecs[i,:], N,
                                             endpoint=False)

    new_vecs[-1,:] = vecs[-1,:]            
    return new_vecs


if __name__ == '__main__':
    vecs = np.loadtxt(sys.argv[1])
    N = int(sys.argv[2])
    print str_arr(kpath(vecs, N), fmt="%f")
