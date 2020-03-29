import numpy as np
from pwtools.crys import Trajectory
from pwtools import crys, num
rand = np.random.rand

def test_dist_traj():
    natoms = 10
    nstep = 100
    cell = rand(nstep,3,3)
    stress = rand(nstep,3,3)
    forces = rand(nstep,natoms,3)
    etot=rand(nstep)
    cryst_const = crys.cell2cc3d(cell, axis=0)
    coords_frac = np.random.rand(nstep,natoms,3)
    coords = crys.coord_trans3d(coords=coords_frac,
                                old=cell,
                                new=num.extend_array(np.identity(3),
                                                     nstep,axis=0),
                                axis=1,
                                timeaxis=0)
    assert cryst_const.shape == (nstep, 6)
    assert coords.shape == (nstep,natoms,3)
    symbols = ['H']*natoms

    traj = Trajectory(coords_frac=coords_frac,
                      cell=cell,
                      symbols=symbols,
                      forces=forces,
                      stress=stress,
                      etot=etot,
                      timestep=1,
                      )

    for pbc in [True, False]:
        # (nstep, natoms, natoms, 3)
        distvecs_frac = traj.coords_frac[:,:,None,:] - \
                        traj.coords_frac[:,None,:,:]
        assert distvecs_frac.shape == (nstep, natoms, natoms, 3)
        if pbc:
            distvecs_frac = crys.min_image_convention(distvecs_frac)
        distvecs = np.empty((nstep, natoms, natoms, 3))
        for ii in range(traj.nstep):
            distvecs[ii,...] = np.dot(distvecs_frac[ii,...], traj.cell[ii,...])
        # (nstep, natoms, natoms)
        dists = np.sqrt((distvecs**2.0).sum(axis=-1))
        assert np.allclose(dists, crys.distances_traj(traj, pbc=pbc))
