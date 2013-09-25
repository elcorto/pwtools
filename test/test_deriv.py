# Test numerical derivatives.
#
# deriv_fd: Test only API
# 
# deriv_spl: Test also correctness of results for the case where we use (x,y)
# as input. Note the low "decimal" values for
# testing.assert_array_almost_equal:
#     n=1: 4
#     n=2: 2
# You see that the derivatives are not very accurate, using the default spline
# parameters (order, smoothing)! However, plotting y - yd reveals that the
# errors are only big near the x-range edges x[0] and x[-1], not in between, so
# it is safe to use the derivatitves after testing for further calculations.

import numpy as np
from pwtools import num
asrt = np.testing.assert_array_almost_equal

def test_deriv():
    x = np.linspace(0,10,100)
    y = np.sin(x)
    for n in [1,2]:
        xd, yd = num.deriv_fd(y, n=n)
        assert [len(xd), len(yd)] == [len(x)-n]*2
        xd, yd = num.deriv_fd(y, x, n=n)
        assert [len(xd), len(yd)] == [len(x)-n]*2
    for n, func, decimal in [(1, np.cos, 4), (2, lambda x: -np.sin(x), 2)]:
        print n, func
        xd, yd = num.deriv_spl(y, n=n)
        assert [len(xd), len(yd)] == [len(x)]*2
        xd, yd = num.deriv_spl(y, x, n=n)
        asrt(func(xd), yd, decimal=decimal)
        assert [len(xd), len(yd)] == [len(x)]*2
        
