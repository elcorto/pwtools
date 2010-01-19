#!/usr/bin/env python

import numpy as N
import math

from pwtools import pydos
##reload(pydos)

def dif(a, b):
	"""Norm of the diff between 1d arrays. Could also use
    numpy.testing.assert_array_almost_equal().
    """
	d = a-b
	return math.sqrt(N.dot(d,d))

# random velocity array
a = N.random.rand(10,50,3) + 1.0
# random mass vector
m = N.random.rand(10) * 10.0 + 1.0

p1 = pydos.pyvacf(a, method=1)
p2 = pydos.pyvacf(a, method=2)
p3 = pydos.pyvacf(a, method=3)
p1m = pydos.pyvacf(a, method=1, m=m)
p2m = pydos.pyvacf(a, method=2, m=m)
p3m = pydos.pyvacf(a, method=3, m=m)

print "p1-p2",   dif(p1,  p2)
print "p1-p3",   dif(p1,  p3)
print "p2-p3",   dif(p2,  p3)
print "p1m-p2m", dif(p1m, p2m)
print "p1m-p3m", dif(p1m, p3m)
print "p2m-p3m", dif(p2m, p3m)

f1 = pydos.fvacf(a, method=1)
f2 = pydos.fvacf(a, method=2)
f1m = pydos.fvacf(a, method=1, m=m)
f2m = pydos.fvacf(a, method=2, m=m)

print "f1-f2",   dif(f1,  f2)
print "f1m-f2m", dif(f1m, f2m)

print "p1-f1",   dif(p1,  f1)
print "p2-f1",   dif(p2,  f1)
print "p3-f1",   dif(p3,  f1)
print "p1-f2",   dif(p1,  f2)
print "p2-f2",   dif(p2,  f2)
print "p3-f2",   dif(p3,  f2)

print "p1m-f1m",   dif(p1m,  f1m)
print "p2m-f1m",   dif(p2m,  f1m)
print "p3m-f1m",   dif(p3m,  f1m)
print "p1m-f2m",   dif(p1m,  f2m)
print "p2m-f2m",   dif(p2m,  f2m)
print "p3m-f2m",   dif(p3m,  f2m)

