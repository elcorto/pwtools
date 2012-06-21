import numpy as np
from pwtools import _flib, crys, timer

def pydist(coords_frac, cell, mic=0):
    distvecs_frac = coords_frac[:,None,:] - coords_frac[None,...]
    if mic == 1:
        distvecs_frac = crys.min_image_convention(distvecs_frac)
    distvecs = np.dot(distvecs_frac, cell)
    distsq = (distvecs**2.0).sum(axis=2)
    return distsq, distvecs, distvecs_frac

def test_fdist():
    
    coords_frac = np.random.rand(5,3)
    cell = np.random.rand(3,3)*3
    
    for mic in [0,1]:
        print "mic:", mic
        pyret = pydist(coords_frac, cell, mic)
        fret = _flib.distsq_frac(coords_frac, cell, mic)
        for ii in [0,1,2]:
            print ii
            assert np.allclose(pyret[ii], fret[ii])
