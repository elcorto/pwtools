#!/usr/bin/python
#
# gen_rpdf_ref.py
#
# For the structures pwtools/test/files/rpdf/*.cif, generate
# pwtools/test/files/rpdf/result.*.txt files with reference results. 
#
# The structures *.cif were generated with pwtools/examples/rpdf/rpdf.py .
#   $ python pwtools/examples/rpdf/rpdf.py
#   $ cp /tmp/rpdf_test/* pwtools/test/files/rpdf/
#   $ python pwtools/test/utils/gen_rpdf_ref.py 
#
# notes:
# ------
# If you'd like to re-generate the .cif files: 
# - back up the old structs - safety first
# - make sure that you use the same seed (np.random.seed()) for all random
#   structs! Otherwise, you will end up with a new structure and you will have
#   to "hg commit" that.

import os
import numpy as np
from pwtools import crys, parse
pj = os.path.join

if __name__ == '__main__':
    
    for name in ['randNx3', 'aln_ibrav0_sc', 'aln_ibrav2_sc']:
        dd = '../files/rpdf'
        pp = parse.CifFile(pj(dd, name + '.cif'))
        pp.parse()
        rad, hist, dens, num_int, rmax_auto = crys.rpdf(pp.coords, 
                                                        rmax=5.0, 
                                                        cp=pp.cell_parameters,
                                                        dr=0.05, 
                                                        pbc=True,
                                                        full_output=True)
        np.savetxt(pj(dd, "result.rad."         + name + ".txt"), rad) 
        np.savetxt(pj(dd, "result.hist."        + name + ".txt"), hist) 
        np.savetxt(pj(dd, "result.num_int."     + name + ".txt"), num_int) 
        np.savetxt(pj(dd, "result.rmax_auto."   + name + ".txt"), [rmax_auto]) 
