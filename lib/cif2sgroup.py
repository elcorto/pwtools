#!/usr/bin/env python

import sys

import parse, crys
import pydos as pd

fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()

# If we ever use this for data parsed from other structure files:
#
# - Make sure that pp.coords are the *crystal" coords. This is true for cif
#   files. 
# - The unit of length of a,b,c is not important (Angstrom for Cif) if coords
#   are crystal.
print crys.wien_sgroup_input('P', pp.symbols, pp.coords, pp.cryst_const)
