#!/usr/bin/env python

import sys
import numpy as np

import parse
import crys
import pydos as pd

fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()

print "celldm (a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab):\n%s\n" %pd.str_arr(pp.celldm)
print "cell_parameters (div. by a):\n%s\n" %pd.str_arr(pp.cell_parameters / pp.cryst_const[0])
##print "atpos (crystal):\n%s\n" %pd.atpos_str(pp.symbols, pp.coords)
print "atpos (crystal):\n%s\n" %pp.atpos_str
print "natoms:\n%s\n" %pp.natoms

cpr = crys.recip_cp(pp.cell_parameters)
print "recip. cell_parameters:\n%s\n" %pd.str_arr(cpr)

norms = np.sqrt((cpr**2.0).sum(axis=1))
print "relation of recip. vector lengths (a:b:c):\n%s\n" %str(norms/norms.min())
