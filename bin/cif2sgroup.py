#!/usr/bin/env python
# 
# cif2sgroup.py
#
# Extract information from a .cif file and print an input file for WIEN2k's
# "sgroup" symmetry analysis tool. Note that the Cif reader assumes P1
# symmetry, i.e. symmetry information in the cif file is ignored. Use
# ase.io.read(), which seems to parse symmetry information.
#
# usage::
#   $ cif2sgroup.py foo.cif > foo.sgroup.in
#   # Find primitive cell
#   $ sgroup -prim [-set-TOL=1e-4] foo.sgroup.in
# 
# See ``sgroup -help`` for more options.
#
# notes:
# ------
# The unit of length of a,b,c is not important (Angstrom, Bohr,...).

import sys
from pwtools import io
fn = sys.argv[1]
struct = io.read_cif(fn)
print io.wien_sgroup_input(struct, lat_symbol='P')
