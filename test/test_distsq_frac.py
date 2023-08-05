import numpy as np
from pwtools import _flib, crys, timer
from pwtools.num import fempty

def pydist(coords_frac, cell, pbc=0):
    distvecs_frac = coords_frac[:,None,:] - coords_frac[None,...]
    if pbc == 1:
        distvecs_frac = crys.min_image_convention(distvecs_frac)
    distvecs = np.dot(distvecs_frac, cell)
    distsq = (distvecs**2.0).sum(axis=2)
    return distsq, distvecs, distvecs_frac

def test_fdist():

    natoms = 5
    coords_frac = np.random.rand(natoms,3)
    cell = np.random.rand(3,3)*3
    struct = crys.Structure(coords_frac=coords_frac,
                            cell=cell)

    for pbc in [0,1]:
        print("pbc:", pbc)
        pyret = pydist(coords_frac, cell, pbc)
        # uses _flib.distsq_frac()
        pyret2 = crys.distances(struct, pbc=pbc, squared=True, fullout=True)
        for ii in [0,1,2]:
            print(ii)
            assert np.allclose(pyret[ii], pyret2[ii])
