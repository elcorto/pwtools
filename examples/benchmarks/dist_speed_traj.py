#!/usr/bin/env python3

import timeit

import numpy as np

from pwtools.crys import Trajectory
from pwtools import crys, num, timer

rand = np.random.rand

# example session on 4-core box, _flob compiled w/ OpenMP (make gfortran-omp)
# ---------------------------------------------------------------------------
#
# $ export OMP_NUM_THREADS=1
# --TagTimer--: py_bigmen:  time: 3.56550598145
# --TagTimer--: py_loop:  time: 0.456802129745
# --TagTimer--: f:  time: 0.437112092972
#
# $ export OMP_NUM_THREADS=2
# --TagTimer--: f:  time: 0.206064939499
#
# $ export OMP_NUM_THREADS=4
# --TagTimer--: f:  time: 0.125560998917


def pydist_bigmem(traj, pbc=True):
    # Pure numpy version w/ big temp arrays. Also slowest.
    #
    # (nstep, natoms, natoms, 3)
    distvecs_frac = (
        traj.coords_frac[:, :, None, :] - traj.coords_frac[:, None, :, :]
    )
    if pbc:
        distvecs_frac = crys.min_image_convention(distvecs_frac)
    distvecs = np.empty((nstep, natoms, natoms, 3))
    for ii in range(traj.nstep):
        distvecs[ii, ...] = np.dot(distvecs_frac[ii, ...], traj.cell[ii, ...])
    # (nstep, natoms, natoms)
    dists = np.sqrt((distvecs ** 2.0).sum(axis=-1))
    return dists


def pydist_loop(traj, pbc=True):
    dists = np.empty((nstep, natoms, natoms))
    for ii, struct in enumerate(traj):
        dists[ii, ...] = crys.distances(struct, pbc=pbc)
    return dists


def fdist(traj):
    return crys.distances_traj(traj, pbc=True)


if __name__ == "__main__":
    natoms = 100
    nstep = 1000
    cell = rand(nstep, 3, 3)
    stress = rand(nstep, 3, 3)
    forces = rand(nstep, natoms, 3)
    coords_frac = rand(nstep, natoms, 3)
    symbols = ["H"] * natoms
    traj = Trajectory(coords_frac=coords_frac, cell=cell, symbols=symbols)

    ##assert np.allclose(pydist_bigmem(traj), pydist_loop(traj))
    ##print("... ok")
    ##assert np.allclose(pydist_loop(traj), fdist(traj))
    ##print("... ok")

    globs = globals()

    statements = [
        "pydist_bigmem(traj)",
        "pydist_loop(traj)",
        "fdist(traj)",
    ]
    for stmt in statements:
        number = 1
        times = np.array(
            timeit.repeat(stmt, globals=globs, number=number, repeat=5)
        )
        print(
            f"{number} loops: {times.mean():6.3f} +- {times.std():.4f}: {stmt}"
        )
