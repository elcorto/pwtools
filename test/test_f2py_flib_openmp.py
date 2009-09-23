import numpy as np
from pwtools.lib import _flib
from pwtools.lib.pydos import fvacf
import os

def check_omp_num_threads():
    key = 'OMP_NUM_THREADS'
    if os.environ.has_key(key):
        print "Python: %s: %s" %(key, os.environ[key])

rand = np.random.rand

nat=100 
nstep=5000 
v=rand(nat, nstep, 3)
m=rand(nat) 
c=np.zeros((nstep,))

bar='-'*70

#-----------------------------------------------------------------------------

print bar
print "testing _flib.vacf(), no nthreads from Python"
check_omp_num_threads()
c = _flib.vacf(v,m,c,1,1)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
nthr = 2
print "testing _flib.vacf(), setting nthr = %i from Python" %nthr
c = _flib.vacf(v,m,c,1,1,nthr)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
print "testing _flib.vacf(), no nthreads from Python, take two"
print "*" * 70
print """!!! POSSIBLE F2PY BUG !!! 
After calling omp_set_num_threads() in the last test, OMP_NUM_THREADS is no
longer recognized !!! nthreads is still at the value from the last test: %s,
that is WRONG
!!! POSSIBLE F2PY BUG !!!""" %nthr
print "*" * 70
check_omp_num_threads()
c = _flib.vacf(v,m,c,1,1)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
nthr = 2
print "testing pydos.fvacf(), setting nthr = %i from Python" %nthr
c = fvacf(v, m=m, nthreads=nthr)
print bar + '\n'

#-----------------------------------------------------------------------------

print bar
print "testing pydos.fvacf(), no nthreads from Python"
check_omp_num_threads()
c = fvacf(v, m=m, nthreads=None)
print bar + '\n'

