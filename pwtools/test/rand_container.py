# Define a Structure and Trajectory filled with random data. All possible attrs
# are used (I hope :)

from pwtools.crys import Structure, Trajectory
import numpy as np
rand = np.random.rand

def get_rand_traj():
    natoms = 10
    nstep = 100
    cell = rand(nstep,3,3)
    stress = rand(nstep,3,3)
    forces = rand(nstep,natoms,3)
    etot=rand(nstep)
    coords_frac = rand(nstep,natoms,3)
    symbols = ['H']*natoms
    tr = Trajectory(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    forces=forces,
                    stress=stress,
                    etot=etot,
                    timestep=1.11,
                    )
    return tr


def get_rand_struct():
    natoms = 10
    symbols = ['H']*natoms
    st = Structure(coords_frac=rand(natoms,3),
                   symbols=symbols,
                   forces=rand(natoms,3),
                   cell=rand(3,3),
                   etot=3.14,
                   stress=rand(3,3))
    return st
