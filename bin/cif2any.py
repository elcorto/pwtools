#!/usr/bin/env python
# 
# cif2pwin.py
#
# Extract information from a .cif file and print them in a format suitable for
# inclusion in a pwscf,abinit,cpmd input file.
#
# usage:
#   cif2pwin.py foo.cif

import sys
import numpy as np

from pwtools import parse, crys, periodic_table, pwscf
from pwtools.common import str_arr, seq2str
from pwtools.constants import Bohr, Angstrom

# All lengths in Bohr
pp = parse.CifFile(sys.argv[1], units={'length': Angstrom/Bohr})
struct = pp.get_struct()
cellr = crys.recip_cell(struct.cell)
norms = np.sqrt((cellr**2.0).sum(axis=1))
bar = '-'*78
mass_unique = [periodic_table.pt[sym]['mass'] for sym in struct.symbols_unique]                 
atspec = pwscf.atspec_str(struct.symbols_unique, 
                          mass_unique, 
                          [sym + '.UPF' for sym in struct.symbols_unique])
celldm = crys.cc2celldm(struct.cryst_const)

print bar
print("""\
The .cif file is assumed to contain NO symmetry information:
    _symmetry_space_group_name_H-M          'P 1'
    _symmetry_Int_Tables_number             1
    loop_
      _symmetry_equiv_pos_as_xyz
       x,y,z
Only a,b,c,alpha,beta,gamma and the fractional coords as found in the file are
used.

CPMD: 
* Instead of CELL ABSOLUTE DEGREE, one can also use
    CELL
        celldm(1) ... celldm(6)
  just like in PWscf.    
* The manual mentions that the code may only read lines up to 80 chars length.
  B/c of that and the fact that Fortran expects 6 numbers after CELL, no matter
  how many linebreaks, better use
    CELL
        celldm(1) 
        ... 
        celldm(6)
  to avoid long lines if each number is printed with "%.16e" or so.
""")
print bar+"\n"

#-------------------------------------------------------------------------------
# PWSCF
#-------------------------------------------------------------------------------
print bar
print "PWSCF"
print bar
print "celldm 1..6: [a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab)]:\n"
for ii, cd in enumerate(celldm):
    print "celldm(%i) = %.16e" %(ii+1, cd)
print "ibrav = 0"
print "nat = %i" %struct.natoms    
print "ntyp = %i" %struct.ntypat
print "CELL_PARAMETERS alat\n%s" %str_arr(struct.cell / struct.cryst_const[0])
print "ATOMIC_SPECIES\n%s" %atspec
print "ATOMIC_POSITIONS crystal\n%s" %pwscf.atpos_str(struct.symbols,
                                                      struct.coords_frac)

#-------------------------------------------------------------------------------
# ABINIT
#-------------------------------------------------------------------------------
print bar
print "ABINIT"
print bar
print "natom %i" %struct.natoms    
print "ntypat %i" %struct.ntypat
print "typat %s" %seq2str(struct.typat)
print "znucl %s" %seq2str(struct.znucl)
print "acell %s" %str_arr(struct.cryst_const[:3])
print "angdeg %s" %str_arr(struct.cryst_const[3:])
print "xred\n%s" %str_arr(struct.coords_frac)
print "pseudopotential order:\n%s" %"\n".join(["%s %i" %(sym, zz) for sym,zz in 
                                               zip(struct.symbols_unique,
                                                   struct.znucl)])

#-------------------------------------------------------------------------------
# CPMD
#-------------------------------------------------------------------------------
print bar
print "CPMD"
print bar
print """\
&SYSTEM
    SYMMETRY
        ???
    SCALE
    POINT GROUP
        AUTO
    CELL ABSOLUTE DEGREE
%s
&END"""\
    %("\n".join([" "*8 + "%.16e" %x for x in struct.cryst_const]),)
print "\n&ATOMS"
for sym, natoms in struct.nspecies.iteritems():
    mask = np.array(struct.symbols) == sym
    print "*%s.psp\n    LMAX=XXXLMAX_%s LOC=XXXLOC_%s" %((sym,) + (sym.upper(),)*2)
    print "    %i" %natoms
    print str_arr(struct.coords_frac[mask,:])    
print "&END"

print bar
print "GENERAL"
print bar
print "recip. cell:\n%s" %str_arr(cellr)
print "relation of recip. vector lengths (a:b:c):\n%s" %str_arr(norms/norms.min())
