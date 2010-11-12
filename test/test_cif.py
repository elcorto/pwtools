import os
import numpy as np
from pwtools.parse import CifFile
from pwtools import io
from testenv import testdir

def test():
    p1 = CifFile('files/cif_struct.cif')
    p1.parse()
    
    # test writing
    filename = os.path.join(testdir, 'test_write_cif.cif')
    io.write_cif(filename, p1.coords, p1.symbols, p1.cryst_const, conv=False)
    p2 = CifFile(filename)
    p2.parse()
    np.testing.assert_array_almost_equal(p1.coords, p2.coords)
    np.testing.assert_array_almost_equal(p1.cryst_const, p2.cryst_const)
    assert p1.symbols == p2.symbols
