#!/usr/bin/env python
# 
# Extract information from a .cif file and print them in a format suitable for
# inclusion in a pwscf,abinit,cpmd input file.
#
# usage:
#   cif2any.py foo.cif

import sys
import numpy as np

from pwtools import crys, atomic_data, pwscf, io
from pwtools.common import str_arr, seq2str
from pwtools.constants import Bohr, Angstrom

def indent(txt, num=4):
    """Indent text block by `num` white spaces."""
    space = " "*num
    return '\n'.join(space + line for line in txt.splitlines())


# All lengths in Bohr
struct = io.read_cif(sys.argv[1], units={'length': Angstrom/Bohr})
rcell = crys.recip_cell(struct.cell)
norms = np.sqrt((rcell**2.0).sum(axis=1))
bar = '-'*78
mass_unique = [atomic_data.pt[sym]['mass'] for sym in struct.symbols_unique]                 
atspec = pwscf.atspec_str(struct.symbols_unique, 
                          mass_unique, 
                          [sym + '.UPF' for sym in struct.symbols_unique])
celldm = crys.cc2celldm(struct.cryst_const)
atpos_frac = pwscf.atpos_str(struct.symbols, struct.coords_frac)

print("""\
Notes
-----
The .cif file is assumed to contain NO symmetry information:
    _symmetry_space_group_name_H-M          'P 1'
    _symmetry_Int_Tables_number             1
    loop_
      _symmetry_equiv_pos_as_xyz
       x,y,z
Only a,b,c,alpha,beta,gamma and the fractional coords as found in the file are
used.

CPMD
----
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

PWscf
-----
We always output ibrav=0, CELL_PARAMETERS and all celldm(1..6) even though you
don't need all that at the same time. Check pw.x's manual. 
""")

#-------------------------------------------------------------------------------
# PWSCF
#-------------------------------------------------------------------------------
print "\n"
out = """
{bar}
PWSCF
{bar}

celldm 1..6: [a[Bohr], b/a, c/a, cos(bc), cos(ac), cos(ab)]:

celldm1 != 1 case
-----------------
&system
    ibrav = 0
    celldm(1) = {cd1}
    celldm(2) = {cd2}
    celldm(3) = {cd3}
    celldm(4) = {cd4}
    celldm(5) = {cd5}
    celldm(6) = {cd6}
    nat = {nat}
    ntyp = {ntyp}
/
CELL_PARAMETERS alat
{cp_alat}

celldm1 = 1 case, no celldm(1) needed
-------------------------------------
&system
    ibrav = 0
    nat = {nat}
    ntyp = {ntyp}
/
CELL_PARAMETERS angstrom
{cp_ang}

CELL_PARAMETERS bohr
{cp_bohr}

ATOMIC_SPECIES
{atspec}
ATOMIC_POSITIONS crystal
{atpos_frac}
"""
rules = {'bar': bar,
         'nat': struct.natoms,
         'ntyp': struct.ntypat,
         'cp_alat': str_arr(struct.cell / struct.cryst_const[0]),
         'cp_bohr': str_arr(struct.cell),
         'cp_ang': str_arr(struct.cell * Bohr / Angstrom),
         'atspec': atspec,
         'atpos_frac': atpos_frac,
        }
for ii, cd in enumerate(celldm):
    rules['cd%i' %(ii+1,)] = cd
        
print out.format(**rules)

#-------------------------------------------------------------------------------
# ABINIT
#-------------------------------------------------------------------------------
print "\n"
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
print "\n"
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
    print indent(str_arr(struct.coords_frac[mask,:]),4)    
print "&END"


#-------------------------------------------------------------------------------
# CP2K
#-------------------------------------------------------------------------------
templ = """\
\n
{bar}
CP2K
{bar}
&force_eval
    &subsys
        &cell
            abc {abc}
            alpha_beta_gamma {angles}
        &end cell
        &coord
            scaled t
{atpos_frac}
        &end coord
{kinds}
    &end subsys
&end force_eval
"""
kind_templ = """\
        &kind {atom}
            basis_set XXXBASISSET_{atom}
            potential XXXPOTENTIAL_{atom}
        &end kind"""
kinds = ''
for atom in struct.symbols_unique:
    kinds += kind_templ.format(atom=atom)
txt = templ.format(abc=str_arr(struct.cryst_const[:3]*Bohr/Angstrom),
                   angles=str_arr(struct.cryst_const[3:]),
                   atpos_frac=indent(atpos_frac,12),
                   kinds=kinds,
                   bar=bar)
print txt

#-------------------------------------------------------------------------------
# general crystal information
#-------------------------------------------------------------------------------

out = """
{bar}
general crystal information
{bar}

reciprocal cell [1/Bohr]:
{rcell_bohr}

reciprocal cell [1/Ang]:
{rcell_ang}

relation of recip. vector lengths (a:b:c)
{rrel}

kpoint grids for some h [1/Ang] resolutions
{kpoints}
"""

cell_ang = struct.cell * Bohr/Angstrom
kpoints = ''
for h in np.arange(.2, 1, .1):
    kpoints += "h = %.3f  size = %s\n" \
        %(h, str_arr(crys.kgrid(cell_ang, h=h), fmt='%i'))

rules = {'bar': bar,
         'rcell_bohr': str_arr(rcell),
         'rcell_ang': str_arr(rcell / Bohr * Angstrom),
         'rrel': str_arr(norms/norms.min()),
         'kpoints': kpoints,
         }

print out.format(**rules)
