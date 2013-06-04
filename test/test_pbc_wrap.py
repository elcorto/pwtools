import numpy as np
from pwtools import crys

def test_pbc_wrap_coords():
    coords_frac = np.array([[1.1, 0.9, -0.1], [-0.8, 0.5, 0.0]])
    coords_frac_tgt = np.array([[ 0.1,  0.9,  0.9],[ 0.2,  0.5,  0. ]])
    np.allclose(coords_frac_tgt, crys.pbc_wrap_coords(coords_frac))

    # 3d array, last index (-1) is xyz, i.e. 0,1,2
    coords_frac_orig = np.random.rand(20,100,3)
    # array of a.shape with 0, 1, -1 randomly distributed
    plus = np.random.randint(-1,1,coords_frac_orig.shape) 
    coords_frac = coords_frac_orig + plus

    coords_frac_wrap = crys.pbc_wrap_coords(coords_frac, xyz_axis=-1)
    # no wrapping here, values inside [0,1]
    np.allclose(coords_frac[plus == 0], coords_frac_wrap[plus == 0])
    np.allclose(coords_frac[plus == -1] + 1, coords_frac_wrap[plus == -1])
    np.allclose(coords_frac[plus == 1] - 1, coords_frac_wrap[plus == 1])
    # the PBC wrapping must restore coords_frac_orig
    np.allclose(coords_frac_orig, coords_frac_wrap)

    # pbc only in x-y, not z
    coords_frac_wrap = crys.pbc_wrap_coords(coords_frac, mask=[True,True,False], xyz_axis=-1)
    np.allclose(coords_frac[...,2], coords_frac_wrap[...,2])

def test_pbc_wrap():
    coords_frac_orig = np.array([[1.1, 0.9, -0.1], 
                                 [-0.8, 0.5,0.0]])
    st = crys.Structure(coords_frac=coords_frac_orig,
                        cell=np.identity(3)*2,
                        symbols=['H']*2)
    st_wrap = crys.pbc_wrap(st)
    # pbc_wrap() makes a copy, make sure the original st is unchanged
    assert (st.coords_frac == coords_frac_orig).all()
    assert (st.coords == coords_frac_orig*2.0).all()
    coords_frac_wrap = np.array([[ 0.1,  0.9,  0.9],[ 0.2,  0.5,  0. ]])
    np.allclose(coords_frac_wrap, st_wrap.coords_frac)
    np.allclose(coords_frac_wrap, st_wrap.coords/2.0)

    coords_frac_orig = np.random.rand(20,100,3)
    plus = np.random.randint(-1,1,coords_frac_orig.shape) 
    coords_frac = coords_frac_orig + plus
    tr = crys.Trajectory(coords_frac=coords_frac,
                         cell=np.identity(3)*2.0,
                         symbols=['H']*100)
                             
    tr_wrap = crys.pbc_wrap(tr, xyz_axis=-1)                      
    np.allclose(coords_frac[plus == 0],      tr_wrap.coords_frac[plus == 0])
    np.allclose(coords_frac[plus == -1] + 1, tr_wrap.coords_frac[plus == -1])
    np.allclose(coords_frac[plus == 1] - 1,  tr_wrap.coords_frac[plus == 1])
    np.allclose(coords_frac_orig, tr_wrap.coords_frac)
    np.allclose(coords_frac_orig, tr_wrap.coords/2.0)

    tr_wrap = crys.pbc_wrap(tr, mask=[True,True,False], xyz_axis=-1)                      
    np.allclose(tr.coords_frac[...,2], tr_wrap.coords_frac[...,2])
