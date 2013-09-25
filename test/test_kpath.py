import numpy as np
from pwtools.kpath import kpath

def test_kpath():
    # only API
    vecs = np.random.rand(10,3)
    kpath(vecs, N=15)
