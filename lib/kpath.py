import numpy as np
import sys
from pwtools.pydos import str_arr

def vlinspace(a, b, num, endpoint=True):
    """Like numpy.linspace, but for 1d arrays. Generate uniformly spaces points
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
    
def kpath(vecs, N):    
    """Simple k-path. Generate N uniformly spaced nd-points along the path
    defined by the vectors in `vecs`. These vectors are the "vertices" of the
    k-path and we construct the fine path by connecting the vertices by their
    distance vectors.

    args:
    -----
    vecs: array (K,M) with K vectors of the Brillouin zone (so M = 3 usually :).
    N : int

    returns:
    --------
    new_vecs : array (N,M) with a fine grid of N vectors along the path defined
        by `vecs`.
    
    todo:
    -----
    Don't generate uniform spacing over the whole path. Instead, calculate an
    equal number of points for each distance vector between 2 vertices.
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
    N = 20
    print str_arr(fine_path(vecs, N), fmt="%f")
