#!/usr/bin/env python
# 
# cif2pwin.py
#
# Extract information from a .cif file and print them in a format suitable for
# inclusion in a pwscf and abinit input file.
#
# usage:
#   cif2pwin.py foo.cif


import sys
import numpy as np

from pwtools import parse, crys, periodic_table, pwscf, abinit
from pwtools.common import str_arr, seq2str
from pwtools import constants

fn = sys.argv[1]
pp = parse.CifFile(fn)
pp.parse()
cellr = crys.recip_cell(pp.cell)
norms = np.sqrt((cellr**2.0).sum(axis=1))
bar = '-'*78
cryst_const = pp.cryst_const.copy()
# in Bohr
cryst_const[:3] /= constants.a0_to_A
abiin = abinit.AbinitInput(pp.symbols)
mass_unique = [periodic_table.pt[sym]['mass'] for sym in abiin.symbols_unique]                 
atspec = pwscf.atspec_str(abiin.symbols_unique, 
                          mass_unique, 
                          [sym + '.UPF' for sym in abiin.symbols_unique])

print bar
print("""NOTE: celldm is valid only for ibrav=0 (pwscf)! The .cif file is
assumed to contain NO symmetry information linking primitive and conventional
cell!""")
print bar

print bar
print "PWSCF"
print bar

print "celldm (a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab):\n%s\n" %str_arr(pp.celldm)
for ii, cd in enumerate(pp.celldm):
    print "celldm(%i) = %.16e" %(ii+1, cd)
print "ibrav = 0"
print "nat = %i" %pp.natoms    
print "ntyp = %i" %abiin.ntypat
print "CELL_PARAMETERS alat\n%s" %str_arr(pp.cell / pp.cryst_const[0])
print "ATOMIC_SPECIES\n%s" %atspec
print "ATOMIC_POSITIONS crystal\n%s" %pp.get_atpos_str()

print bar
print "ABINIT"
print bar
print "natom %i" %pp.natoms    
print "ntypat %i" %abiin.ntypat
print "typat %s" %seq2str(abiin.typat)
print "znucl %s" %seq2str(abiin.znucl)
print "acell %s" %str_arr(cryst_const[:3])
print "angdeg %s" %str_arr(cryst_const[3:])
print "xred\n%s" %str_arr(pp.coords)
print "pseudopotential order:\n%s" %"\n".join(["%s %i" %(sym, zz) for sym,zz in 
                                               zip(abiin.symbols_unique,
                                                   abiin.znucl)])

print bar
print "GENERAL"
print bar
print "recip. cell:\n%s" %str_arr(cellr)
print "relation of recip. vector lengths (a:b:c):\n%s" %str_arr(norms/norms.min())
