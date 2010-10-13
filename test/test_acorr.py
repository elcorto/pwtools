import numpy as np
from pwtools.corr import acorr

arr = np.random.rand(100)
ref = acorr(arr, method=1)
for m in range(2,8):
    print "%i : 1" %m
    np.testing.assert_array_almost_equal(acorr(arr, method=m), ref)    
