#!/usr/bin/python

# Create a PWscf MD "trajectory" where the CELL_PARAMETERS unit changes. This
# is for testing if the parser can handle that. The output file consists only
# of CELL_PARAMETERS blocks, which is enough for PwVCMDOutputFile.get_gell() .
#
# usage
# ------
# ./make-vc-md-cell.py > ../files/pw.vc-md.cell.out 

import numpy as np
from pwtools import common

if __name__ == '__main__':
    cell = np.arange(1,10).reshape((3,3))

    alat_lst = [2.0, 4.0]
    for ialat,alat in enumerate(alat_lst):
        for ii in range(5):
            cell_str = common.str_arr((cell + 0.02*ii + ialat)/alat)
            print("CELL_PARAMETERS (alat= %.5f)\n%s" %(alat, cell_str))
            
