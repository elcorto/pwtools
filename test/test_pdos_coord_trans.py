# Test which shows that pydos.*_dos() needs cartesian coords.
#
# This test shows that scaling the coords does not matter b/c we normalize the
# integral area in pydos.*_pdos(). But using a different coord sys does not work.
# One must convert coords to cartesian before calculating the PDOS.
#
# "cart" and "cart2" must be exactly the same. "cell1"
# must match in principle, but not overlay the other two. 


import numpy as np
from pwtools import pydos as pd
from pwtools.crys import coord_trans, velocity_traj
from pwtools.test import tools
rand = np.random.random

def pdos(coords_arr_3d, axis=0):
    f, d = pd.direct_pdos(velocity_traj(coords_arr_3d, axis=axis))
    return d

def test_pdos_coord_trans():
    coords = {}

    # cartesian: first axis is the time axis
    coords['cart'] = rand((100, 10, 3))

    # cartesian scaled, e.g. Angstrom instead of Bohr
    coords['cart2'] = coords['cart']*5

    # some other coord sys
    cell1 = rand((3,3))
    # coord_trans: axis=-1 specifies the "x,y,z"-axis of dimension 3
    coords['cell1'] = coord_trans(coords['cart'],
                                  old=np.identity(3), 
                                  new=cell1,
                                  axis=-1)

    dos = {}
    for key, val in coords.iteritems():
        dos[key] = pdos(val)

    np.testing.assert_array_almost_equal(dos['cart'], dos['cart2'])
    assert not tools.array_almost_equal(dos['cart'], dos['cell1'])
