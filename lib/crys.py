#!/usr/bin/env python
# vim:ts=4:sw=4:et

# 
# Copyright (c) 2008, Steve Schmerler <mefx@gmx.net>.
# The pydos package. 
# 
# This is a hackish q'n'd script to construct supercells. It shows how to use
# the pydos.py module as a toolkit.

import numpy as np
from math import sqrt
import pydos

# [1] http://www.quantum-espresso.org/input-syntax/INPUT_PW.html#id53713
# [2] http://cst-www.nrl.navy.mil/lattice/struk/b4.html
# [3] http://www.quantum-espresso.org/input-syntax/INPUT_PW.html#id55830

# cartesian basis vecs
x = np.array([1.,0.,0.])
y = np.array([0.,1.,0.])
z = np.array([0.,0.,1.])
cart = np.array([x,y,z])
print "cartesian basis"
print pydos.str_arr(cart)


#----------------------------------------------------------------------------
# define lattice here
#----------------------------------------------------------------------------

# all in a.u. (Rydbery atomic units)

# celldm(1) == a == alat
alat = 5.90978704749766
# (celldm(3) = c/a = 1.6)
c = 1.6 * alat
# no domension
##u = 0.387
u=0.382 # claudi

# pwscf's def. for hexagonal primitive lattice vecs (ibrav=4), from [1]
#
# simple hexagonal and trigonal(p)
# ================================
# v1 = a(1,0,0),  v2 = a(-1/2,sqrt(3)/2,0),  v3 = a(0,0,c/a).
#
a1 = alat*x  
a2 = alat*np.array([-0.5, sqrt(3)/2.0, 0.0])  
a3 = c*z
# [[ --- a1 --- ]
#  [ --- a2 --- ]
#  [ --- a3 --- ]]
#
# divide by alat -> plv's in alat units 
plv = np.array([a1, a2, a3]) / alat
print "primitive lattice vecs [cell_parameters (alat)]"
print pydos.str_arr(plv)

# Wurzite structure, spacegrpup P6_3mc. See [2].
# Positions in crystal coords = in units of primitive lattice vectors.
# Al = Zn, N = S
# 
# rows:
#   Al1
#   Al2
#   N1
#   N2
apc = np.array([
    [1.0/3.0, 2.0/3.0, 0.0],
    [2.0/3.0, 1.0/3.0, 0.5],
    [1.0/3.0, 2.0/3.0, u],
    [2.0/3.0, 1.0/3.0, 0.5+u],
    ])
print "atomic_positions (crystal)"
print pydos.str_arr(apc)

# Coordinate transformation: crystal -> cartesian. See pydos.py.
#
# Al1 = ( apc[0,0]*a1 + apc[0,1]*a2 + apc[0,2]*a3 )/a  
# Al2 = ( apc[1,0]*a1 + apc[1,1]*a2 + apc[1,2]*a3 )/a
# N1  = ( apc[2,0]*a1 + apc[2,1]*a2 + apc[2,2]*a3 )/a
# N2  = ( apc[3,0]*a1 + apc[3,1]*a2 + apc[3,2]*a3 )/a
# 
apa = pydos.coord_trans(apc, old=plv, new=cart, align='rows')

print "atomic_positions (alat)"
print pydos.str_arr(apa)

# build supercell
#
# It's also possible to parse an 1cell input file with pydos tools to extract
# `plv` and `apa` and `atoms`.
# 
# fn = 'AlN.md.in'
# atpos_in = pydos.atomic_positions(fn)
# apa = atpos_in['R0'] # make sure atpos_in['unit'] == 'alat' !!!
# atoms = atpos_in['symbols']
# plv = pydos.cell_parameters(fn)
#
atoms = ['Al', 'Al', 'N', 'N']

scs, scell = pydos.scell(apa, plv, pydos.scell_mask(3,3,3), atoms)

print("supercell:")
for i in range(scell.shape[0]):    
    print scs[i], pydos.str_arr(scell[i,:])

print("new plv:")
print pydos.str_arr(plv * 3)
