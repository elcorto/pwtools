#!/usr/bin/env python

import sys

import parse
import pydos as pd

fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()

print "celldm (a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab):\n%s\n" %pd.str_arr(pp.celldm)
print "cell_parameters (div. by a):\n%s\n" %pd.str_arr(pp.cell_parameters / pp.cryst_const[0])
print "atpos (crystal):\n%s\n" %pd.atpos_str(pp.symbols, pp.coords)
print "natoms:\n%s\n" %pp.natoms

