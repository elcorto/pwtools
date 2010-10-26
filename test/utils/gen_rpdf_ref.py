#!/usr/bin/python
#
# gen_rpdf_ref.py
#
# For the two AlN structures pwtools/test/files/rpdf/*.cif, generate
# pwtools/test/files/rpdf/result.*.txt files with reference results. 
# Also, generate results for a random set of atoms.
#
# The AlN structures were generated with pwtools/examples/rpdf/rpdf.py .
#   $ python pwtools/examples/rpdf/rpdf.py
#   $ cp /tmp/rpdf_test/* pwtools/test/files/rpdf/
#   $ python pwtools/test/utils/gen_rpdf_ref.py 
#
# notes:
# ------
# Make sure that you use the same seed (np.random.seed()) for all random
# structs! Otherwise, you will end up with a new structure and you will have
# to "hg commit" that.

import os
import numpy as np
from pwtools import crys, parse, io
pj = os.path.join

if __name__ == '__main__':
    
    for name in ['rand_3d', 'aln_ibrav0_sc', 'aln_ibrav2_sc']:
        dd = '../files/rpdf'
        if name == 'rand_3d':
            # important!
            np.random.seed(3)
            natoms_O = 10
            natoms_H = 3
            symbols = ['O']*natoms_O + ['H']*natoms_H
            coords_in = np.random.rand(natoms_H + natoms_O, 3, 30)
            cp = np.identity(3)*10
            sy = np.array(symbols)
            msk1 = sy=='O'
            msk2 = sy=='H'
            coords = [coords_in[msk1, ..., 10:], coords_in[msk1, ..., 10:]]
            np.savetxt(pj(dd, name + '.cp.txt'), cp)
            io.writetxt(pj(dd, name + '.coords0.txt'), coords[0])
            io.writetxt(pj(dd, name + '.coords1.txt'), coords[1])
        else:
            pp = parse.CifFile(pj(dd, name + '.cif'))
            pp.parse()
            coords = pp.coords
            cp = pp.cell_parameters
        rad, hist, num_int, rmax_auto = crys.rpdf(coords, 
                                                  rmax=5.0, 
                                                  cp=cp,
                                                  dr=0.05, 
                                                  pbc=True,
                                                  full_output=True)
        np.savetxt(pj(dd, "result.rad."         + name + ".txt"), rad) 
        np.savetxt(pj(dd, "result.hist."        + name + ".txt"), hist) 
        np.savetxt(pj(dd, "result.num_int."     + name + ".txt"), num_int) 
        np.savetxt(pj(dd, "result.rmax_auto."   + name + ".txt"), [rmax_auto]) 
