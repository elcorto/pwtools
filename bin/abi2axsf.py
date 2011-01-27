#!/usr/bin/python

# Parse abinit output file (so far tested w/ ionmov 13 + optcell 2) and write
# .axsf file for xcrysden/VMD.
# 
# usage:
#   abi2axsf.py abi.out abi.axsf

import sys
from pwtools import parse, crys, constants, io

filename = sys.argv[1]
outfile = sys.argv[2]
pp = parse.AbinitVCMDOutputFile(filename)

#               Abinit      XSF         
# cell          Bohr        Ang
# cart. forces  Ha/Bohr     Ha/Ang

# Use pp.get_*() to parse only what we need.
coords_frac = pp.get_coords_frac()
cell = pp.get_cell()*constants.a0_to_A 
symbols = pp.get_symbols()
forces = pp.get_forces() / constants.a0_to_A

io.write_axsf(filename=outfile, 
              coords=coords_frac,
              cell=cell,
              forces=forces,
              symbols=symbols)
