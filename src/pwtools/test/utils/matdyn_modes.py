#!/usr/bin/env python3

# Generate matdyn.modes file, filled with dummy data. Parsed and used in
# test/test_read_matdyn.py .

import numpy as np
from pwtools.common import str_arr

if __name__ == '__main__':
    natoms = 2
    qpoints = np.array([[0,0,0.], [0,0,0.5]])
    nqpoints = len(qpoints)
    nmodes = 3*natoms

    freqs = np.empty((nqpoints, nmodes))
    vecs = np.empty((nqpoints, nmodes, natoms, 3))
    num = 0
    for iqpoint in range(nqpoints):
        print("     diagonalizing the dynamical matrix ...\n")
        print("  q = %s" %str_arr(qpoints[iqpoint], fmt='%f'))
        print("*"*79)
        for imode in range(nmodes):
            freqs[iqpoint, imode] = num
            print("    omega(%i) = %f [THz] = %f [cm-1]" %(imode+1, num*0.1, num))
            num += 1
            for iatom in range(natoms):
                vec_str = " ("
                for icoord in range(3):
                    vecs[iqpoint,imode,iatom,icoord] = num
                    vec_str += "  %f  %f  " %(num, 0.03*num)
                    num += 1
                vec_str += ")"
                print(vec_str)
        print("*"*79)
