import numpy as np
from pwtools.signal import acorr

def test_acorr():
    arr = np.random.rand(100)
    for norm in [True,False]:
        ref = acorr(arr, method=1, norm=norm)
        for m in range(2,8):
            print("%i : 1" %m)
            np.testing.assert_array_almost_equal(acorr(arr, method=m, 
                                                       norm=norm), 
                                                 ref)    
