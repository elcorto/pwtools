# Test crys.rpdf() against reference results. See utils/gen_rpdf_ref.py for how
# the references were generated.

import os
import numpy as np
from pwtools import crys, parse
pj = os.path.join

if __name__ == '__main__':
    
    for name in ['randNx3', 'aln_ibrav0_sc', 'aln_ibrav2_sc']:
        dd = 'files/rpdf'
        pp = parse.CifFile(pj(dd, name + '.cif'))
        pp.parse()
        rad, hist, dens, num_int, rmax_auto = crys.rpdf(pp.coords, 
                                                        rmax=5.0, 
                                                        cp=pp.cell_parameters,
                                                        dr=0.05, 
                                                        pbc=True,
                                                        full_output=True)
        results = {'rad':       rad,
                   'hist':      hist, 
                   'num_int':   num_int,
                   'rmax_auto': np.array(rmax_auto),
                   }
        for key, val in results.iteritems():
            ref = np.loadtxt(pj(dd, "result.%s.%s.txt" %(key, name))) 
            np.testing.assert_array_almost_equal(ref, val)
