import numpy as np
from pwtools.crys import cc2celldm, celldm2cc

def assrt(a,b):
    np.testing.assert_array_almost_equal(a, b)

def test():
    cc = np.array([3,4,5, 30, 50, 123.0])
    assrt(cc, celldm2cc(cc2celldm(cc)))
    
    cc = np.array([3,4,5, 30, 50, 123.0])
    assrt(cc, celldm2cc(cc2celldm(cc, fac=10), fac=0.1))

    cc = [3,3,3,90,90,90]
    assrt(cc2celldm(cc), np.array([3,1,1,0,0,0]))

    cc = [3,4,5, 90, 90, 120]
    assrt(cc2celldm(cc), 
          np.array([3, 4/3., 5/3., 0,0, -0.5]))
    
    cc = [3,4,5, 90, 90, 120]
    assrt(cc2celldm(cc, fac=10), 
          np.array([30, 4/3., 5/3., 0,0, -0.5]))
       
