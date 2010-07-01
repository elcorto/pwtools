import numpy as np

from pwtools.lib.pydos import velocity

a1 = np.arange(2*3*6).reshape(2,3,6)
a2 = a1.copy()
a3 = a1.copy()
v1 = velocity(a1, copy=True, tslice=slice(None), axis=-1)
v2 = a2[...,1:] - a2[...,:-1]
v3 = np.diff(a3, n=1, axis=-1)

assert v1.shape == v2.shape == v3.shape == (2,3,5)
assert (v1 == v2).all()
assert (v1 == v3).all()
assert (v2 == v3).all()

v1 = velocity(a1, copy=False, tslice=slice(None), axis=-1)
assert v1.shape == (2,3,5)
assert (v1 == a1[...,1:]).all()
