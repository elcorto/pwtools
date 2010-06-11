# This test shows that scaling the coords does not matter b/c we normalize the
# integral area in pd.*_pdos(). But using a different coord sys does not work.
# One must convert coords to cartesian before calculating the PDOS.

import numpy as np
from pwtools import pydos as pd
from pwtools.lib.crys import coord_trans
from matplotlib import pyplot as plt

def pdos(coords_arr_3d):
    f, d = pd.direct_pdos(pd.velocity(coords_arr_3d))
    return d

rand = np.random.random
coords = {}

# cartesian
coords['cart'] = rand((10, 200, 3))

# cartesian scaled, e.g. Angstrom instead of Bohr
coords['cart2'] = coords['cart']*5

# some other coord sys
cp1 = rand((3,3))
coords['cp1'] = coord_trans(coords['cart'], old=np.identity(3), new=cp1)

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
