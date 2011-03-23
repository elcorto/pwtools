import numpy as np
from pwtools import crys

def test():
    # test known values: simple cubic
    cell_a = 5.0
    cell = np.identity(3)*cell_a
    volume = cell_a**3.0
    cryst_const = np.array([cell_a]*3 + [90.0]*3)
    np.testing.assert_array_almost_equal(crys.cell2cc(cell), cryst_const)
    np.testing.assert_array_almost_equal(crys.cc2cell(cryst_const), cell)
    np.testing.assert_equal(volume, crys.volume_cc(cryst_const))
    np.testing.assert_equal(volume, crys.volume_cell(cell))
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

