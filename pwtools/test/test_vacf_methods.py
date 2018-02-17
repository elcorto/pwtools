#!/usr/bin/env python

def test_vacf_methods():
    import numpy as np
    import math
    from pwtools import pydos

    def assrt(a, b):
        np.testing.assert_array_almost_equal(a,b)

    # random velocity array: 10 atoms, 50 steps, 50 x  (10,3) arrays
    a = np.random.rand(50,10,3) + 1.0
    # random mass vector
    m = np.random.rand(10) * 10.0 + 1.0

    p1 = pydos.pyvacf(a, method=1)
    p2 = pydos.pyvacf(a, method=2)
    p3 = pydos.pyvacf(a, method=3)
    p1m = pydos.pyvacf(a, method=1, m=m)
    p2m = pydos.pyvacf(a, method=2, m=m)
    p3m = pydos.pyvacf(a, method=3, m=m)

    assrt(p1,  p2)
    assrt(p1,  p3)
    assrt(p2,  p3)
    assrt(p1m, p2m)
    assrt(p1m, p3m)
    assrt(p2m, p3m)

    f1 = pydos.fvacf(a, method=1)
    f2 = pydos.fvacf(a, method=2)
    f1m = pydos.fvacf(a, method=1, m=m)
    f2m = pydos.fvacf(a, method=2, m=m)

    assrt(f1,  f2)
    assrt(f1m, f2m)

    assrt(p1,  f1)
    assrt(p2,  f1)
    assrt(p3,  f1)
    assrt(p1,  f2)
    assrt(p2,  f2)
    assrt(p3,  f2)

    assrt(p1m,  f1m)
    assrt(p2m,  f1m)
    assrt(p3m,  f1m)
    assrt(p1m,  f2m)
    assrt(p2m,  f2m)
    assrt(p3m,  f2m)

