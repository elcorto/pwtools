#!/usr/bin/env python3

# Compare speed of distance calculation: numpy vs. fortran. The Fortran version
# is ~10x faster. See also test/test_distsq_frac.py .

import timeit

import numpy as np
from pwtools import _flib, crys, timer, num


def pydist(arr, cell, pbc=0):
    distvecs_frac = arr[:, None, :] - arr[None, ...]
    if pbc == 1:
        distvecs_frac = crys.min_image_convention(distvecs_frac)
    distvecs = np.dot(distvecs_frac, cell)
    distsq = (distvecs ** 2.0).sum(axis=2)
    return distsq, distvecs, distvecs_frac


def fdist(arr, cell, pbc=0):
    natoms = arr.shape[0]
    distsq = num.fempty((natoms, natoms))
    dummy1 = num.fempty((natoms, natoms, 3))
    dummy2 = num.fempty((natoms, natoms, 3))
    return _flib.distsq_frac(arr, cell, pbc, distsq, dummy1, dummy2)


if __name__ == "__main__":

    pbc = 1

    arr = np.random.rand(100, 3)
    cell = np.random.rand(3, 3) * 3

    globs = globals()

    statements = [
        "pydist(arr, cell, pbc)",
        "fdist(arr, cell, pbc)",
    ]
    for stmt in statements:
        number = 500
        times = np.array(
            timeit.repeat(stmt, globals=globs, number=number, repeat=5)
        )
        print(
            f"{number} loops: {times.mean():6.3f} +- {times.std():.4f}: {stmt}"
        )
