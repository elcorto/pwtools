#!/usr/bin/env python
# 
# cif2pwin.py
#
# Extract information from a .cif file and print them in a format suitable for
# inclusion in a pascf input file.
#
# usage:
#   cif2pwin.py foo.cif


import sys
import numpy as np

from pwtools import parse
from pwtools import crys
from pwtools.common import str_arr

fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()

print "celldm (a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab):\n%s\n" %str_arr(pp.celldm)
print "cell_parameters (div. by a):\n%s\n" %str_arr(pp.cell_parameters / pp.cryst_const[0])
print "atpos (crystal):\n%s\n" %pp.atpos_str
print "natoms:\n%s\n" %pp.natoms

cpr = crys.recip_cp(pp.cell_parameters)
print "recip. cell_parameters:\n%s\n" %str_arr(cpr)

norms = np.sqrt((cpr**2.0).sum(axis=1))
print "relation of recip. vector lengths (a:b:c):\n%s\n" %str(norms/norms.min())
