"""
Test speed of _fsymfunc.symfunc_45() against symfunc_45_fast().

Bottom line: 
* OpenMP scales, but for sure not linear. Don't expect too much speedup on many
  cores. Tested up to 4.
* We set rcut = rmax_smith, which is ~ dists.mean(). Only then, we skip enough
  dists to make the *_fast() routines faster. If we use rcut=dists.max(), then
  the *_fast() versions are slower, even with OpenMP.
"""

import numpy as np
from pwtools import _fsymfunc, _flib, timer, num, crys
rand = np.random.rand

tt = timer.TagTimer()

# Run tests `nn` times to get better average.
nn=10

# Cheap random struct: atoms in a cubic box, just using rand() and ignoring
# covalent radii. Seed the random generator to have the same stream in each
# run.
np.random.seed(1234)
natoms = 100 
npsets = 16
nparams = 4
st = crys.Structure(coords_frac=rand(natoms,3),
                    cell=np.identity(3)*5)
distsq, distvecs, distvecs_frac = crys.distances(st, pbc=True, squared=True,
                                                 fullout=True)
anglesijk = crys.angles(st, pbc=True, deg=False)                                                 
dists = np.sqrt(distsq)
params = rand(npsets,nparams)

rcut = crys.rmax_smith(st.cell)
print "dists: min=%f, max=%f, mean=%f, rcut=%f" %(dists.min(), dists.max(),
                                                  dists.mean(), rcut)
params[:,0] = rcut

ret = num.fempty((natoms,npsets))
tt.t('g4')
for i in range(nn):
    _fsymfunc.symfunc_45(distsq, anglesijk, ret, params, 4)
tt.pt('g4')
g4 = ret.copy()

del ret
ret = num.fempty((natoms,npsets))
tt.t('g4_fast')
for i in range(nn):
    _fsymfunc.symfunc_45_fast(distvecs, dists, ret, params, 4)
tt.pt('g4_fast')
g4_fast = ret.copy()
assert (np.abs(g4) > 0.0).any()
assert np.allclose(g4,g4_fast)

tt.t('g5')
for i in range(nn):
    _fsymfunc.symfunc_45(distsq, anglesijk, ret, params, 5)
tt.pt('g5')
g5 = ret.copy()

del ret
ret = num.fempty((natoms,npsets))
tt.t('g5_fast')
for i in range(nn):
    _fsymfunc.symfunc_45_fast(distvecs, dists, ret, params, 5)
tt.pt('g5_fast')
g5_fast = ret.copy()
assert (np.abs(g5) > 0.0).any()
assert np.allclose(g5,g5_fast)
