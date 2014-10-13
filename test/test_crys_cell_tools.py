from math import pi
import numpy as np
from pwtools import crys
from pwtools.test.tools import aaae, aae
rand = np.random.rand

def test_cell_tools():
    # test known values: simple cubic
    cell_a = 5.0
    cell = np.identity(3)*cell_a
    volume = cell_a**3.0
    cryst_const = np.array([cell_a]*3 + [90.0]*3)
    np.testing.assert_array_almost_equal(crys.cell2cc(cell), cryst_const)
    np.testing.assert_array_almost_equal(crys.cc2cell(cryst_const), cell)
    np.testing.assert_almost_equal(volume, crys.volume_cc(cryst_const))
    np.testing.assert_almost_equal(volume, crys.volume_cell(cell))
    np.testing.assert_array_almost_equal(crys.cc2cell(crys.cell2cc(cell)), cell)
    np.testing.assert_array_almost_equal(crys.cell2cc(crys.cc2cell(cryst_const)),
                                         cryst_const)
                                         
    # random
    #
    # volume : volume_cc() always returns positive values, whereas det() and
    #     volume_cell() may return the volume with negative sign but correct
    #     magnitude.
    # cell : A random cell does also have a random orientation in space. It
    #     does NOT conform the usual convention: a along x, b in x-y plane.
    #     However, the cryst_const and volume must be invariant.
    cell = np.random.rand(3,3)
    cryst_const = crys.cell2cc(cell)
    volume = abs(np.linalg.det(cell))
    np.testing.assert_almost_equal(volume, crys.volume_cc(cryst_const))
    np.testing.assert_almost_equal(volume, abs(crys.volume_cell(cell)))
    # this will and must always fail for random cells
    try:
        np.testing.assert_array_almost_equal(crys.cc2cell(crys.cell2cc(cell)), cell)
    except AssertionError:
        print("KNOWNFAIL")
    # Here, we convert cryst_const to a *different* cell which conforms to the
    # orientation convention, and back to cryst_const.
    np.testing.assert_array_almost_equal(crys.cell2cc(crys.cc2cell(cryst_const)),
                                         cryst_const)

    # 3d
    cell = rand(100,3,3)
    cc = crys.cell2cc3d(cell, axis=0)
    vol_cell = np.abs(crys.volume_cell3d(cell, axis=0))
    vol_cc = crys.volume_cc3d(cc, axis=0)

    assert crys.cell2cc3d(cell, axis=0).shape == (100,6)
    assert crys.cc2cell3d(cc, axis=0).shape == (100,3,3)
    
    assert vol_cc.shape == (100,)
    assert vol_cell.shape == (100,)
    aaae(vol_cell, vol_cc)
    aaae(crys.cell2cc3d(crys.cc2cell3d(cc)), cc)


def test_recip_cell():
    # reciprocal cell
    cell = rand(3,3)
    rcell = crys.recip_cell(cell)
    vol = crys.volume_cell(cell)
    try:
        assert np.allclose(crys.recip_cell(rcell), cell)
    except AssertionError:        
        assert np.allclose(crys.recip_cell(rcell), -1.0 * cell)
    assert np.allclose(crys.volume_cell(rcell), (2*pi)**3.0 / vol)


def test_kgrid():
    # `h` is very small, just to make all `size` entries odd
    cell = np.diag([3,4,5]) # Angstrom
    size = crys.kgrid(cell, h=0.23)
    assert (np.array(size) == np.array([9,7,5])).all()
    size = crys.kgrid(cell, h=0.23, even=True)
    assert (np.array(size) == np.array([10,8,6])).all()
    size, spacing = crys.kgrid(cell, h=0.23, fullout=True)
    assert np.allclose(spacing, crys.kgrid(cell, size=size))
    # big cell, assert Gamma = [1,1,1] or better
    size = crys.kgrid(cell*100, h=0.23, minpoints=2)
    assert (np.array(size) == np.array([2,2,2])).all()


def test_voigt():
    a = rand(3,3) 
    # symmetric tensor
    s = np.dot(a,a.T)
    v = crys.tensor2voigt(s)
    assert (crys.voigt2tensor(v) == s).all()
    assert v[0] == s[0,0]
    assert v[1] == s[1,1]
    assert v[2] == s[2,2]
    assert v[3] == s[1,2] == s[2,1]
    assert v[4] == s[0,2] == s[2,0]
    assert v[5] == s[0,1] == s[1,0]
    
    nstep = 10
    a = rand(nstep,3,3) 
    s = np.array([np.dot(a[i,...],a[i,...].T) for i in range(nstep)])
    v = crys.tensor2voigt3d(s)
    assert (v[:,0] == s[:,0,0]).all()
    assert (v[:,1] == s[:,1,1]).all()
    assert (v[:,2] == s[:,2,2]).all()
    assert (v[:,3] == s[:,1,2]).all()
    assert (v[:,4] == s[:,0,2]).all()
    assert (v[:,5] == s[:,0,1]).all()
    assert (v[:,3] == s[:,2,1]).all()
    assert (v[:,4] == s[:,2,0]).all()
    assert (v[:,5] == s[:,1,0]).all()

    assert (crys.voigt2tensor3d(v) == s).all()
    assert (crys.tensor2voigt3d(s) == \
            np.array([crys.tensor2voigt(s[i,...]) for i in \
            range(nstep)])).all()
    assert (crys.voigt2tensor3d(v) == \
            np.array([crys.voigt2tensor(v[i,...]) for i in \
            range(nstep)])).all()

