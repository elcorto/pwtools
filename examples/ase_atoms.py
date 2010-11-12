# Write animated xsf file with pwtools and ASE.

import numpy as np
from pwtools import io, crys
from ase.atoms import Atoms
from ase.io import write

natoms = 5
nstep = 10
# dummy size, Angstrom
alat = 7
cell = np.identity(3)*alat
# cartesian coords in Angstrom
coords_ang = np.random.rand(natoms, 3, nstep)*alat
# fractional
coords_frac = crys.coord_trans(coords_ang.swapaxes(-1,-2), 
                               old=np.identity(3), 
                               new=cell,
                               align='rows').swapaxes(-1,-2)
symbols = ['X']*natoms

atoms_lst = [Atoms('X'*natoms,
                   positions=coords_ang[...,i],
                   cell=cell,
                   pbc=[1,1,1]) \
             for i in range(nstep)]      

io.write_axsf('pwtools.axsf', coords_frac, cell, symbols)
write('ase.axsf', atoms_lst, format='xsf')
