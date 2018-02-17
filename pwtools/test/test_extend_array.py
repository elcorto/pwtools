import numpy as np
import os
from pwtools import num, common
rand = np.random.rand

def equal(a,b):
    assert (a == b).all()

def test_extend_array():
    arr = rand(3,3)
    nrep = 5
    a0 = num.extend_array(arr, nrep, axis=0)
    a1 = num.extend_array(arr, nrep, axis=1)
    a2 = num.extend_array(arr, nrep, axis=2)    
    am1 = num.extend_array(arr, nrep, axis=-1)    
    assert a0.shape == (nrep,3,3)
    assert a1.shape == (3,nrep,3)
    assert a2.shape == (3,3,nrep)
    assert am1.shape == (3,3,nrep)
    equal(a2, am1)

    for axis, aa in enumerate([a0, a1, a2]):
        for ii in range(nrep):
            # slicetake(a0, 3, 0) -> a0[3,:,:]
            equal(arr, num.slicetake(aa, ii, axis=axis))
