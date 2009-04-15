#!/usr/bin/env python

# Plot a dispersion. This is a simplified (and non-ugly) version of QE's
# plotband.x . Reads a k-point - frequency file produced by matdyn.x .

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os
from pwtools.lib import mpl
nrm = np.linalg.norm

class SpecialPoint(object):
    def __init__(self, arr, symbol, path_norm=None):
        self.arr = arr
        self.symbol = symbol
        self.path_norm =path_norm

#-----------------------------------------------------------------------------

def parse_dis(filename):
    fh = open(filename)

    # Read number of bands (nbnd) and qpoints (nks).
    # OK, Fortran's namelists win here :)
    line0 = fh.next()
    pat=r'.*[ ]+nbnd[ ]*=[ ]*([0-9]+)[ ]*,[ ]*nks[ ]*=[ ]*([0-9]+)[ ]*/'
    match = re.match(pat, line0)
    nbnd = int(match.group(1))
    nks = int(match.group(2))
    print "nbnd = %i" %nbnd
    print "nks = %i" %nks

    ks = []
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
    ##print freqs
    
    kpath_fn = os.path.join(os.path.dirname(filename), 'kpath_def.txt')
    if os.path.exists(kpath_fn):
        special_points = []
        fhk = open(kpath_fn)
        for line in fhk:    
            spl = line.strip().split()
            special_points.append(
                SpecialPoint(np.array(spl[:3], dtype=float), 
                    spl[-1].replace('#', '')))
        fhk.close()
    else:
        special_points = None
##    special_points = None
    
    # like in plotband.f90, path_norm = kx there.
    # This is a sequence of cummulative norms of the diference vectors which
    # conect each two adjacent special points.
    path_norm = np.empty(nks, dtype=float)
    for i in range(ks.shape[0]):
        if i == 0:
            path_norm[i] = nrm(ks[i,:])
        else:
            path_norm[i] = path_norm[i-1] + nrm(ks[i,:]-ks[i-1,:])
        if special_points is not None:
            for j, sp in enumerate(special_points):
                if j == 0:
                    tmp = nrm(sp.arr)
                else:
                    tmp += nrm(special_points[j].arr - special_points[j-1].arr)
                if abs(tmp - path_norm[i]) < 1e-9:                
                    sp.path_norm = path_norm[i]
    ##print path_norm
    ##print path_norm.shape
    fh.close()
    return path_norm, freqs, special_points

#------------------------------------------------------------------------------

def plot_dis(path_norm, freqs, special_points):
    # Plot columns of `freq` against q points (path_norm)
    plt.plot(path_norm, freqs, '.-')
    if special_points is not None:
        yl = plt.ylim()
        x, labels = get_special(special_points)
        print x
        print labels
        plt.vlines(x, yl[0], yl[1])
        plt.xticks(x, labels)
    

def get_special(special_points):
    x = [sp.path_norm for sp in special_points]
    labels = [sp.symbol for sp in special_points]
    return x, labels


if __name__ == '__main__':
    
    import os
    fn = sys.argv[1]
    plot_dis(*parse_dis(os.path.abspath(fn)))
    plt.show()

