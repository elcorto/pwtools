import numpy as np
from pwtools.lib import _flib
from pwtools.lib.pydos import fvacf
import os
import sys

OMP_DCT = {'num_threads': None}
def omp_num_threads(action='check', num=1, omp_dct=OMP_DCT, err=False):
    key = 'OMP_NUM_THREADS'
    has_key = os.environ.has_key(key)
    if action == 'check':
        if has_key:
            print "[omp_num_threads] os.environ['%s']: %s" %(key, os.environ[key])
            if err and os.environ[key] != '3':
                return 'err'
        else:
            print "[omp_num_threads] no os.environ['%s']" %key
            if err:
                return 'err'
    elif action == 'backup':
        if has_key:
            print "[omp_num_threads] backup os.environ['%s'] = '%s'" %(key, os.environ[key])
            omp_dct['num_threads'] = os.environ[key]
        else:            
            omp_dct['num_threads'] = None
    elif action == 'restore':
        if has_key:
            print "[omp_num_threads] restoring os.environ['%s'] = '%s'" \
                %(key, omp_dct['num_threads'])
            os.environ[key] = omp_dct['num_threads']
    elif action == 'set':
        print "[omp_num_threads] setting os.environ['%s'] = '%s'" %(key, str(num))
        os.environ[key] = str(num)

rand = np.random.rand

nat=100 
nstep=5000 
v=rand(nat, nstep, 3)
m=rand(nat) 
c=np.zeros((nstep,))

bar='-'*70

ret = omp_num_threads('check', err=True)
if ret == 'err':
    print bar
    print """Do 
    $ export OMP_NUM_THREADS=3
before running this test."""
    print bar + '\n'
    sys.exit(0)

omp_num_threads('check')
omp_num_threads('backup')
omp_num_threads('set', num=4)

#-----------------------------------------------------------------------------

print bar
print """testing _flib.vacf(v,m,c,1,1), no nthreads from Python -- extension
called directly ... does NOT read os.environ, reacts only if OMP_NUM_THREADS
has been set in the shell BEFORE this test script was called"""
omp_num_threads('check')
c = _flib.vacf(v,m,c,1,1)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
nthr = 2
print "testing _flib.vacf(v,m,c,1,1,nthr), setting nthr = %i" %nthr
c = _flib.vacf(v,m,c,1,1,nthr)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
print "testing _flib.vacf(v,m,c,1,1), no nthreads from Python, take two"
print "*" * 70
print """!!! POSSIBLE F2PY BUG !!! 
After calling omp_set_num_threads() in the last test, OMP_NUM_THREADS is no
longer recognized on the Fortran side!!! nthreads is still at the value from
the last test: %s, that is WRONG
!!! POSSIBLE F2PY BUG !!!""" %nthr
print "*" * 70
omp_num_threads('check')
c = _flib.vacf(v,m,c,1,1)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
nthr = 2
print """testing pydos.fvacf(v, m=m, nthreads=nthr), setting nthr = %i --
override any OMP_NUM_THREADS setting""" %nthr
c = fvacf(v, m=m, nthreads=nthr)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
print """testing pydos.fvacf(v, m=m, nthreads=None): no nthreads from Python -- It
reads os.environ (workaround for f2py bug)."""
omp_num_threads('check')
c = fvacf(v, m=m, nthreads=None)
print bar + '\n'

omp_num_threads('restore')
