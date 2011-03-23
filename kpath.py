import numpy as np
from pwtools import num

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
        new_vecs[(i-1)*N:i*N, :] = num.vlinspace(vecs[i-1,:], vecs[i,:], N,
                                                 endpoint=False)

    new_vecs[-1,:] = vecs[-1,:]            
    return new_vecs


if __name__ == '__main__':
    import sys
    from pwtools.common import str_arr
    vecs = np.loadtxt(sys.argv[1])
    N = int(sys.argv[2])
    print str_arr(kpath(vecs, N), fmt="%f")
