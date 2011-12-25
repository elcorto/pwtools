# Write animated xsf file with pwtools and ASE, based on fractional coords. 
# Write file with original cartesian Angstrom coords, which is what you will
# find in the .axsf files.

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
coords_frac = crys.coord_trans(coords_ang, 
                               old=np.identity(3), 
                               new=cell,
                               axis=1)
symbols = ['X']*natoms

# symbols: ASE can use 'X'*natoms or ['X']*natoms
# positions: use positions = coords_ang or scaled_positions = coords_frac
atoms_lst = [Atoms(symbols=symbols,
                   scaled_positions=coords_frac[...,i], 
                   cell=cell,
                   pbc=[1,1,1]) \
             for i in range(nstep)]      

io.write_axsf(filename='pwtools.axsf', 
              coords_frac=coords_frac, 
              cell=cell, 
              symbols=symbols)
write('ase.axsf', atoms_lst, format='xsf')
io.writetxt('coords_ang.txt', coords_ang, axis=-1)
