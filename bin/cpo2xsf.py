#!/usr/bin/env python

import sys
print 'imports ...'
from cStringIO import StringIO
import os
import sys
import numpy as np
from pwtools import common as com
from pwtools import pydos as pd
from pwtools import constants as con
from pwtools import regex

FLOAT_RE = regex.float_re

# usage: <this_script> cp.in cp.out
cpi_fn = sys.argv[1]
cpo_fn = sys.argv[2]

##st = com.backtick(r"egrep 'Species.*atoms[ ]+=' %s | awk '{print $5}'" % cpo_fn)
##natoms = int(np.loadtxt(StringIO(st)).sum())
#----------------
##natoms = int(com.backtick(r"egrep 'Species.*atoms[ ]+=' %s | \
##    awk '{print $5}' | paste -s -d '+' | bc" %cpo_fn))

print 'parsing infile ...'
nl = pd.pwin_namelists(cpi_fn)
natoms   = int(nl['system']['nat'])
isave = int(nl['control']['isave'])
dt_ha = float(nl['control']['dt'])
symbols = pd.pwin_atomic_positions(cpi_fn)['symbols']

##dt_ryd = 2*dt_ha
dt = dt_ha * con.th

##print "natoms:", natoms
##print "isave:", isave

if not os.path.exists('pdos'):
    print "creating pdos/"
    os.makedirs('pdos')

# CP prints ATOMIC_POSITIONS in cartesian Bohr (even if input was in cartesian
# Angstrom). We get them from the output file (printed every `isave` steps),
# but could also get them from <outdir>/<prefix>.pos
#
# nstep from cpin_fn is not reliable due to restarts etc, get brute force from
# outfile instead

# atomic positions
print 'getting coords ...'
key = 'ATOMIC_POS'
nstep = int(com.backtick('grep %s %s | wc -l' %(key, cpo_fn)))
cmd = "sed -nre '/%s/,+%ip' %s | grep -v %s | \
    awk '{printf $2\"  \"$3\"  \"$4\"\\n\"}'" %(key, natoms, cpo_fn, key)
axis = 1
header ="# [array]\n# shape = %s\n# axis = %i" %(pd.tup2str((natoms, nstep, 3)), axis) 
coords_str = header + '\n' + com.backtick(cmd)
coords = pd.readtxt(StringIO(coords_str))
fn = 'pdos/' + cpo_fn + '.r.txt'                            
print 'writing %s ...' %fn
com.file_write(fn, coords_str)

# cell parameters
print 'getting cps ...'
key = 'CELL_PARA'
cmd = "sed -nre '/%s/,+3p' %s | grep -v %s" %(key, cpo_fn, key)
axis = 1
header ="# [array]\n# shape = %s\n# axis = %i" %(pd.tup2str((3, nstep, 3)), axis) 
cp_str = header + '\n' + com.backtick(cmd)
cps = pd.readtxt(StringIO(cp_str))
fn = 'pdos/cps.txt'                            
print 'writing %s ...' %fn
com.file_write(fn, coords_str)

# mass vector
print 'getting atpos from infile ...'
atpos_in = pd.pwin_atomic_positions(cpi_fn)
header = '# [array]\n # shape = %i\n # axis = -1' %natoms
mass_str = header + '\n' + '\n'.join(['%e' % x for x in atpos_in['massvec']])
fn = 'pdos/' + cpo_fn + '.m.txt'
print 'writing %s ...' %fn
com.file_write(fn, mass_str)

## # temperature
## print 'getting temperature ...'
## cmd = r"egrep 'temperature\s*=' %s " %cpo_fn + r"| sed -re 's/.*temp.*=\s*(" + FLOAT_RE + \
##       r")\s*K/\1/'"
## header = '# [array]\n # shape = %i\n # axis = -1' %nstep
## temp_str = header + '\n' + com.backtick(cmd)
## fn = 'pdos/' + cpo_fn + '.temp.txt'
## print 'writing %s ...' %fn
## com.file_write(fn, temp_str)

# .axsf file for xcrysden
print 'generating axsf ...'
axsftxt = ''
axsftxt += 'ANIMSTEPS %i\n' %nstep
axsftxt += 'CRYSTAL\n'
for i in range(coords.shape[axis]):
    axsftxt += 'PRIMVEC %i\n%s\n' %(i+1, \
        pd.str_arr(cps[:,i,:]*con.a0_to_A, fmt='%.10e'))
    axsftxt += 'PRIMCOORD %i\n' %(i+1)
    axsftxt += '%i 1\n' %natoms
    axsftxt += pd.atpos_str(symbols, 
                            coords[:,i,:]*con.a0_to_A, fmt='%.10e') + '\n'
fn = 'pdos/coords.axsf'                            
print 'writing %s ...' %fn
com.file_write(fn, axsftxt)
