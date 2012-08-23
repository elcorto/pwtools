from math import pi
import numpy as np
from pwtools import crys
from pwtools.test.tools import aaae, aae
rand = np.random.rand

def test():
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

    # reciprocal cell
    cell = rand(3,3)
    rcell = crys.recip_cell(cell)
    vol = crys.volume_cell(cell)
    try:
        assert np.allclose(crys.recip_cell(rcell), cell)
    except AssertionError:        
        assert np.allclose(crys.recip_cell(rcell), -1.0 * cell)
    assert np.allclose(crys.volume_cell(rcell), (2*pi)**3.0 / vol)
