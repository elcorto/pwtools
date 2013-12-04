import numpy as np
from pwtools import num, pwscf, common

def kpath(vecs, N=10):    
    """Simple k-path. Given a set of K vectors (special points in the BZ),
    generate a "fine path" of N*(K-1)+1 vectors along the path defined by the
    vectors in `vecs`. The K vectors are the "vertices" of the k-path and we
    construct the fine path by connecting the vertices by their distance
    vectors and placing N points on each connection edge.

    Parameters
    ----------
    vecs : array (K,M) 
        Array with K vectors of the Brillouin zone (so M = 3 usually :)
    N : int

    Returns
    -------
    new_vecs : array (N*(K-1)+1,M) 
        Array with a fine grid of vectors along the path 
        defined by `vecs`.
    
    Notes
    -----
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


def get_path_norm(ks):
    """Like in QE's plotband.f90, path_norm = kx there. Return a sequence of
    cumulative norms of the difference vectors which connect each two adjacent
    k-points.

    Parameters
    ----------
    ks : array (nks, 3), array with `nks` k-points on the path
    """
    dnorms = np.empty(ks.shape[0], dtype=float)
    dnorms[0] = np.linalg.norm(ks[0,:])
    # diff(...): array with difference vecs, norm of each of them
    dnorms[1:] = np.sqrt((np.diff(ks, axis=0)**2.0).sum(axis=1))
    # cumulative sum
    path_norm = dnorms.cumsum(axis=0)
    return path_norm


class SpecialPoint(object):
    def __init__(self, arr, symbol, path_norm=None):
        # (3,) : the k-point
        self.arr = arr
        # symbol, e.g. 'Gamma'
        self.symbol = symbol
        # not used ...
        ##self.path_norm = path_norm


class SpecialPointsPath(object):
    def __init__(self, sp_lst=None, ks=None, symbols=None):
        # List of SpecialPoint instances.
        self.sp_lst = sp_lst
        
        if self.sp_lst is not None:
            # Array of k-points. (nks, 3)
            self.ks = np.array([sp.arr for sp in self.sp_lst])
            # list (nks,) with a symbol for each special point
            self.symbols = [sp.symbol for sp in self.sp_lst]
        else:
            assert None not in [ks,symbols], "ks and/or symbols is None"
            assert len(ks) == len(symbols), ("ks and symbols have different \
                                            lengths")
            self.ks = ks
            self.symbols = symbols
        # 1d array (nks,) of cumulative norms
        self.path_norm = get_path_norm(self.ks)


def plot_dis(path_norm, freqs, special_points_path=None, **kwargs):
    """Plot dispersion. See bin/plot_dispersion.py for a usage example. This
    lives here (and not in pwscf.py) b/c it is not PWscf-specific. It can be
    used for any dispersion data (band structure).
    
    See pwscf.read_matdyn_freq() for how to get `freqs` in the case of phonon
    dispersions.

    Parameters
    ----------
    path_norm : array (nks,)
        x-axis with cumulative norms of points along the k-path
    freqs : array (nks, nbnd)
        `nbnd` frequencies for each band at each k-point
    special_points_path : optional, a SpecialPointsPath instance,
        used for pretty-printing the x-axis (set special point labels)
    """
    import matplotlib.pyplot as plt
    # Plot columns of `freq` against q points (path_norm)
    plt.plot(path_norm, freqs, **kwargs)
    if special_points_path is not None:
        yl = plt.ylim()
        ks, nrm, symbols = special_points_path.ks, \
                         special_points_path.path_norm, \
                         special_points_path.symbols
        plt.vlines(nrm, yl[0], yl[1])
        fmtfunc = lambda x: "%.2g" %x
        labels = ['%s\n[%s]' %(sym, common.seq2str(kk, func=fmtfunc,sep=','))\
                  for sym,kk in zip(symbols, ks)]
        print nrm
        print labels
        plt.xticks(nrm, labels)
        plt.show()
