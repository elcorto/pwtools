# This test shows that scaling the coords does not matter b/c we normalize the
# integral area in pydos.*_pdos(). But using a different coord sys does not work.
# One must convert coords to cartesian before calculating the PDOS.
#
# In the resulting plot, "cart" and "cart2" must be exactly the same. "cp1"
# must match in principle, but not overlay the other two. 

import numpy as np
from pwtools import pydos as pd
from pwtools.crys import coord_trans
from matplotlib import pyplot as plt

def pdos(coords_arr_3d):
    f, d = pd.direct_pdos(pd.velocity(coords_arr_3d, axis=-1))
    return d

rand = np.random.random
coords = {}

# cartesian, new convention: last axis is the time axis
coords['cart'] = rand((10, 3, 200))

# cartesian scaled, e.g. Angstrom instead of Bohr
coords['cart2'] = coords['cart']*5

# some other coord sys
#
# swapaxes : transform (10,3,200) -> (10,200,3) -> coord_trans -> transform
# back to (10,2,300); for coord_trans, the last axis must have dim 3
cp1 = rand((3,3))
coords['cp1'] = coord_trans(coords['cart'].swapaxes(-1,-2),
                            old=np.identity(3), 
                            new=cp1,
                            align='rows').swapaxes(-1,-2)

dos = {}
for key, val in coords.iteritems():
    dos[key] = pdos(val)

for key, val in dos.iteritems():
    plt.plot(val, label=key)

plt.legend()
plt.show()

print ">>> this test should pass ..."
np.testing.assert_array_almost_equal(dos['cart'], dos['cart2'])
print ">>> ... ok"

print ">>> this test should fail ..."
try:
    np.testing.assert_array_almost_equal(dos['cart'], dos['cp1'])
except AssertionError:
    print "OK KNOWNFAIL"
