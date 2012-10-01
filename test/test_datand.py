import numpy as np
from pwtools.num import DataND
rand = np.random.rand

def test_datand():
    x0 = np.sort(rand(3))
    x1 = np.sort(rand(2))
    x2 = np.sort(rand(5))
    x3 = np.sort(rand(7))
    print x0
    axes = [x0,x1,x2,x3]
    shape = tuple([len(x) for x in axes])
    a2 = np.empty((np.prod(shape), len(shape)+1), dtype=x0.dtype)

    an_ref = np.empty(shape, dtype=x0.dtype)
    idx = 0
    for i0,_x0 in enumerate(x0):
        for i1,_x1 in enumerate(x1):
            for i2,_x2 in enumerate(x2):
                for i3,_x3 in enumerate(x3):
                    val = _x0*_x1*_x2*_x3
                    an_ref[i0,i1,i2,i3] = val
                    a2[idx,0] = _x0
                    a2[idx,1] = _x1
                    a2[idx,2] = _x2
                    a2[idx,3] = _x3
                    a2[idx,4] = val
                    idx += 1

    nd = DataND(a2=a2)
    assert (nd.an == an_ref).all()
    
    for x,y in zip(axes, nd.axes):
        assert (x == y).all()
