#!/usr/bin/env python3

# Load phonon DOS as calculated by QE's matdyn.x (e.g. matdyn.phdos):
#   [f in cm^-1]  [dos]
# and int_f dos(f) = 3*natom.
#
# Print a file to stdout which is a suitable phonon dos input for F_QHA.f90 (as
# of QE 4.2). This file *must* be named PHDOS.out . This file has the form
#
#   natom
#   natom   nstep   emax    de
#   <matdyn.phdos>
# 
# where
#   natom : number of atoms in unit cell
#   nstep : number of rows in matdyn.phdos
#   emax : max. frequency (in cm^-1)
#   de : frequency axis spacing
#
# usage:
#   $ matdyn2fqha.py matdyn.phdos > PHDOS.out
#   $ f90 F_QHA.f90 -o fqha.x
#   # make input file "fqha.in" for fqha.x:
#   $ cat fqha.in
#     PHDOS.out   
#     fqha.out    
#     10,1500,10 
#     ! phonon dos file w/ special header
#     ! file where to write output
#     ! Tmin,Tmax,dT
#   $ ./fqha.x < fqha.in
# 
# Find results in fqha.out .

import sys
import numpy as np
from scipy.integrate import simps

filename = sys.argv[1]
arr = np.loadtxt(filename)
freq = arr[:,0]
dos = arr[:,1]

integral = simps(dos, freq)

natom = integral / 3.
nstep = len(freq)
emax = freq[-1]
de = freq[1] - freq[0]

sys.stderr.write("""\
integal: %f
natom: %f 
nstep: %i 
emax: %f 
de: %f
"""%(integral, natom, nstep, emax, de))

inatom = int(round(natom))
print("%i\n%i  %i  %f  %f" %(inatom, inatom, nstep, emax, de))
print(open(filename).read())
