import os
import numpy as np
from pwtools.parse import CifFile
from pwtools import io
from testenv import testdir

def test():
    p1 = CifFile('files/cif_struct.cif').get_struct()
    assert p1.cell is not None
    assert p1.cryst_const is not None
    assert p1.symbols is not None
    assert p1.coords is not None
    assert p1.coords_frac is not None    

    # test writing
    filename = os.path.join(testdir, 'test_write_cif.cif')
    io.write_cif(filename, p1)
    p2 = CifFile(filename).get_struct()
    np.testing.assert_array_almost_equal(p1.coords, p2.coords)
    np.testing.assert_array_almost_equal(p1.cryst_const, p2.cryst_const)
    np.testing.assert_array_almost_equal(p1.cell, p2.cell)
    assert p1.symbols == p2.symbols

