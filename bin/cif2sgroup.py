#!/usr/bin/env python
# 
# cif2sgroup.py
#
# Extract information from a .cif file and print a input file for WIEN2k's
# "sgroup" symmetry analysis tool.
#
# usage:
#   cif2sgroup.py foo.cif > foo.sqroup.in
#
#
# notes:
# ------
#
# If we ever use this for data parsed from other structure files:
# - Make sure that pp.coords are the *crystal" coords. This is true for cif
#   files. 
# - The unit of length of a,b,c is not important (Angstrom for Cif) if coords
#   are crystal.


import sys
from pwtools import parse, io
fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()
print io.wien_sgroup_input(pp, lat_symbol='P')
