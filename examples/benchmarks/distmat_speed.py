#!/usr/bin/env python3

"""
About
=====

Micro-benchmarks comparing various methods for computing distance matrices for
N-dim points (npoints, ndim). Used in rbf, for example. We calculate *squared*
distances and skip the sqrt().

We calculate a square distance matrix (so all pairwise distances) since we also
want to benchmark scipy.spatial.distance.pdist. For that we use *one* array `arr`
of shape (npoints, ndim).

Our (slow) reference version is the pure numpy solution

    ((arr[:,None,:] - arr[None,...])**2.0).sum(-1)

We use similar methods in dist_speed_{struct,traj}.py but there we have
special-purpose code that deals with fractional coords and gives us the distance
vectors as well. Here we're only interested in collections of N-dim points w/o
dealing with their basis vectors.


Observations
============

serial
------
* use scipy.spatial.distance.cdist()
* serial numba is close, but still factor 2 slower

parallel
--------
* num.distsq (Fortran OpenMP) beats numba
* parallel numba = serial cdist

numba
-----

* Can't speed up numpy ndarray ops (like our reference code above) that already
  happen in numpy's C layer, but potentially use lots of temp arrays. So numba
  doesn't "unpack" those operations into their loop equivalents and then compiles
  those.
* Instead, for numba (and LLVM) to be effective, one needs to spell out all loops
  C/Fortran style. Don't mix/leave in numpy bits, that actually makes things
  worse!
* But: given that we don't need to write one line of C or Fortran and
  avoid building extensions, it's pretty cool!


Measurements
============

serial
------

* numba_loops_parallel_True skipped here, see "parallel" below

mean and std over 5 times 100 executions:
 2.757 +- 0.0092: ((arr[:,None,:] - arr[None,...])**2.0).sum(-1)
 0.410 +- 0.0071: squareform(pdist(arr, metric='sqeuclidean'))
 0.276 +- 0.0010: cdist(arr, arr, metric='sqeuclidean')
 0.274 +- 0.0011: num.distsq(arr, arr)
 3.154 +- 0.0156: numba_ndarray(arr, arr)
 0.515 +- 0.0011: funcs['numba_loops_parallel_False'](arr, arr)

parallel
--------
* 2 cores
* pwtools: num.distsq Fortran OpenMP,
  need to do
    cd src
    make clean
    make gfortran-omp
  before to recompile _flib extension
* numba: njit(parallel=True), use numba.prange

mean and std over 5 times 100 executions:
 0.146 +- 0.0017: num.distsq(arr, arr)
 0.282 +- 0.0018: funcs['numba_loops_parallel_True'](arr, arr)
"""


import timeit

from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np

from pwtools import num


statements = [
    "((arr[:,None,:] - arr[None,...])**2.0).sum(-1)",
    "squareform(pdist(arr, metric='sqeuclidean'))",
    "cdist(arr, arr, metric='sqeuclidean')",
    "num.distsq(arr, arr)",
]

try:
    from numba import float64, njit, prange
    print("compile using numba jit ...")

    # Adding explicit types allows eager compilation, i.e. at parse time rather
    # than runtime -> no need for numba to infer types and compile in a
    # separate first call to the function before running bench using the
    # compiled function.
    #
    # Use expand_dims() since numba doesn't support None indexing.
    #
    # Strange error with parallel=True in ndarray case:
    #   AssertionError: Sizes of $10call_method.4, $20call_method.9 do not \
    #                   match on ... line 57 ...
    # Not investigated further, so only serial here.
    @njit(float64[:, :](float64[:, :], float64[:, :]), parallel=False)
    def numba_ndarray(aa, bb):
        # aa[:,None,:] - bb[None,:,:]
        d3d = np.expand_dims(aa, 1) - np.expand_dims(bb, 0)
        return (d3d ** 2.0).sum(-1)

    statements.append("numba_ndarray(arr, arr)")

    def make_loop_func(parallel):
        @njit(float64[:, :](float64[:, :], float64[:, :]), parallel=parallel)
        def func(aa, bb):
            nx = aa.shape[0]
            ny = bb.shape[0]
            ndim = aa.shape[1]
            out = np.zeros((nx, ny))
            for ii in prange(nx):
                for jj in prange(ny):
                    # slow
                    ##out[ii, jj] = np.sum((aa[ii, :] - bb[jj, :]) ** 2.0)
                    # fast
                    for kk in prange(ndim):
                        out[ii, jj] += (aa[ii,kk] - bb[jj,kk])**2.0
            return out
        return func

    funcs = {}
    for parallel in [True, False]:
        funcs[f"numba_loops_parallel_{parallel}"] = make_loop_func(parallel)

    for fname in funcs.keys():
        statements.append(f"funcs['{fname}'](arr, arr)")

except ModuleNotFoundError:
    print("numba not found, skipping related tests ...")


arr = np.random.rand(1000, 3)

# dict with module-level global vars passed as context to timeit. That contains,
# besides stuff imported (scipy.spatial.distance functions) also the array
# `arr`, the pre-compiled numba function `numba_ndarray` as well as the dict
# `funcs` with pre-compiled numba functions. Cool!
globs = globals()


ref_stmt = statements[0]
ref = eval(ref_stmt)
for stmt in statements[1:]:
    diff = np.abs(ref - eval(stmt)).max()
    assert diff < 1e-15, f"ref={ref_stmt} stmt={stmt} diff={diff}"

print("running bench ...")
nloops = 100
nreps = 5
print(f"mean and std over {nreps} times {nloops} executions:")
for stmt in statements:
    times = np.array(
        timeit.repeat(stmt, globals=globs, number=nloops, repeat=nreps)
    )
    print(f"{times.mean():6.3f} +- {times.std():.4f}: {stmt}")
