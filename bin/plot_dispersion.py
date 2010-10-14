#!/usr/bin/env python

# Plot a dispersion. This is a simplified version of QE's plotband.x . Reads a
# k-point - frequency file produced by QE's matdyn.x See parse_dis().
#
# This is a q'n'd hack. Patches welcome.

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os
nrm = np.linalg.norm


def get_path_norm(ks):
    """Like in QE's plotband.f90, path_norm = kx there. Return a sequence of
    cumulative norms of the difference vectors which connect each two adjacent
    k-points.

    args:
    -----
    ks : array (nks, 3), array with `nks` k-points on the path
    """
    dnorms = np.empty(ks.shape[0], dtype=float)
    dnorms[0] = nrm(ks[0,:])
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
        # don't know what we did with that, apparently only used in
        # get_special()
        self.path_norm = path_norm


class SpecialPointsPath(object):
    def __init__(self, sp_lst):
        # List of SpecialPoint instances.
        self.sp_lst = sp_lst
        
        # Array of k-points. (nks, 3)
        self.ks = np.array([sp.arr for sp in self.sp_lst])
        # 1d array (nks,) of cumulative norms
        self.path_norm = get_path_norm(self.ks)
        # list (nks,) with a symbol for each special point
        self.symbols = [sp.symbol for sp in self.sp_lst]


def get_special(special_points):
    """Utility function. Used by post-processing scripts."""
    x = [sp.path_norm for sp in special_points]
    labels = [sp.symbol for sp in special_points]
    return x, labels


def parse_dis(fn_freq, fn_kpath_def=None):
    """Parse frequency file produced by matdyn.x .
    
    args:
    ----
    fn_freq : name of the frequency file
    fn_kpath_def : optional (only for plotting later), special points definition
        file, see notes below
    
    returns:
    --------
    path_norm, freqs, special_points_path
    path_norm : array (nks,), sequence of cumulative norms of the difference
        vectors which connect each two adjacent k-points
    freqs : array (nks, nbnd), array with `nbnd` frequencies for each k-point,
        nbnd should be = 3*natom (natom = atoms in the unit cell)
    special_points_path : SpecialPointsPath instance or None

    notes:
    ------

    matdyn.x must have been instructed to calculate a phonon dispersion along a
    predefined path in the BZ. e.g. natom=2, nbnd=6, 101 k-points on path
        
        example:        
        -------------------------------------------------------------
        &input
            asr='simple',  
            amass(1)=26.981538,
            amass(2)=14.00674,
            flfrc='fc',
            flfrq='matdyn.freq.disp'
        /
        101                                | nks
        0.000000    0.000000    0.000000   |
        0.037500    0.037500    0.000000   | List of nks = 101 k-points
        ....                               |
        -------------------------------------------------------------


    `fn_freq` has the form
        <header>
        <k-point, (3,)>
        <frequencies,(nbnd,)
        <k-point, (3,)>
        <frequencies,(nbnd,)
        ...

        example:        
        -------------------------------------------------------------
        &plot nbnd=   6, nks= 101 /
                  0.000000  0.000000  0.000000
          0.0000    0.0000    0.0000  456.2385  456.2385  871.5931
                  0.037500  0.037500  0.000000
         23.8811   37.3033   54.3776  455.7569  457.2338  869.8832
        ... 
        -------------------------------------------------------------
    
    `fn_kpath_def` : 
        <coordinate of special point> #<name>
        ...

        example:
        -------------------------------------------------------------
        0    0    0     # $\Gamma$
        0.75 0.75 0     # K
        1 0.5 0         # W
        1 0 0           # X
        0 0 0           # $\Gamma$
        .5 .5 .5        # L
        -------------------------------------------------------------
    Note that you can put special matplotlib math text in this file. Everything
    after `#' is treated as a Python raw string.

    For correct plotting, the k-points defined in `fn_kpath_def` MUST of course
    be on the exact same k-path as the k-points listed in `fn_freq`.
    """

    fh = open(fn_freq)

    # Read number of bands (nbnd) and qpoints (nks).
    # OK, Fortran's namelists win here :)
    line0 = fh.next()
    pat = r'.*\s+nbnd\s*=\s*([0-9]+)\s*,\s*nks\s*=\s*([0-9]+)\s*/'
    match = re.match(pat, line0)
    # number of bands = number of frequencies per k-point
    nbnd = int(match.group(1))
    # number of k-points
    nks = int(match.group(2))
    print "nbnd = %i" %nbnd
    print "nks = %i" %nks
    
    # list of k-points [[k0_x, k0_y, k0_z], [k1_x, k1_y, k1_z], ...]
    ks = []
    # list of frequencies for that k-point
    freqs = []
    for line in fh:
        spl = line.split()
        if len(spl) == 3:
            ks.append(spl)
        else:
            freqs.append(spl)
    fh.close()
    # (nks, 3), each row is a q vector
    ks = np.array(ks, dtype=float)
    # (nks, nbnd), each *row* contains the freqs for ONE q point, so each *column*
    # has len nks and is a band
    freqs = np.array(freqs, dtype=float).reshape(nks, nbnd)
    
    # parse k-path definition file
    if fn_kpath_def is not None:
        special_points = []
        fhk = open(fn_kpath_def)
        for line in fhk:    
            spl = line.strip().split()
            special_points.append(
                SpecialPoint(np.array(spl[:3], dtype=float), 
                    r'%s' %spl[-1].replace('#', '')))
        fhk.close()
        special_points_path = SpecialPointsPath(special_points)
    else:
        special_points_path = None

    # calculate path norms (= x-axis for plotting)
    path_norm = get_path_norm(ks)
    return path_norm, freqs, special_points_path


def plot_dis(path_norm, freqs, special_points_path):
    # Plot columns of `freq` against q points (path_norm)
    plt.plot(path_norm, freqs, '.-')
    if special_points_path is not None:
        yl = plt.ylim()
        x, labels = special_points_path.path_norm, special_points_path.symbols
        print x
        print labels
        plt.vlines(x, yl[0], yl[1])
        plt.xticks(x, labels)
    

if __name__ == '__main__':
    
    # usage:
    #   plot_dispersion.py foo.freq [kpath_def.txt]

    fn_freq = sys.argv[1]
    if len(sys.argv) == 3:
        fn_kpath_def = sys.argv[2]
    else:
        fn_kpath_def = None
    plot_dis(*parse_dis(fn_freq, fn_kpath_def))
    plt.show()

