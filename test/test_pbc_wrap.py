def test():
    import numpy as np
    from pwtools import crys
    a = np.array([[1.1, 0.9, -0.1], [-0.8, 0.5, 0.0]])
    atgt = np.array([[ 0.1,  0.9,  0.9],[ 0.2,  0.5,  0. ]])
    np.testing.assert_array_almost_equal(atgt, crys.pbc_wrap(a))

    # 3d array, last index (-1) is xyz, i.e. 0,1,2
    aorig = np.random.rand(20,100,3)
    # array of a.shape with 0, 1, -1 randomly distributed
    plus = np.random.randint(-1,1,aorig.shape) 
    a = aorig + plus

    awrap = crys.pbc_wrap(a, xyz_axis=-1)
    # no wrapping here, values inside [0,1]
    np.testing.assert_array_equal(a[plus == 0], awrap[plus == 0])
    np.testing.assert_array_almost_equal(a[plus == -1] + 1, awrap[plus == -1])
    np.testing.assert_array_almost_equal(a[plus == 1] - 1, awrap[plus == 1])
    # the PBC wrapping must restore aorig
    np.testing.assert_array_almost_equal(aorig, awrap)

    # pbc only in x-y, not z
    awrap = crys.pbc_wrap(a, mask=[True,True,False], xyz_axis=-1)
    np.testing.assert_array_equal(a[...,2], awrap[...,2])
