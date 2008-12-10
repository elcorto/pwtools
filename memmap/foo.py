import numpy as N
from scipy.io import npfile
from _foo import foo
from time import sleep

nx = 8
ny = 200
nz = 3
ash = (nx, ny, nz)

a = N.random.random(ash)
print "a.shape:", a.shape
print "a.flags:", a.flags
print "\na size: %f MB" %(a.nbytes / 1024**2.0)
print "---------------" 

print "write a ..."
# write in F order
npf = npfile('a.dat', permission='wb', order='F')
npf.write_array(a)
npf.close()
print "... finished"
print "---------------" 

print "b = memmap(...)"
b = N.memmap('a.dat', dtype='float64', mode='r', shape=ash, order='F')
print "b.shape:", b.shape
print "b.flags:", b.flags
print "---------------" 

print "c = b[1:, 1:, 1:]"
c = b[1:, 1:, 1:]
print "c.shape:", c.shape
print "c.flags:", c.flags
print "---------------" 

#####################

##print "foo(a)"
##foo(a)
##print "---------------" 
##
##print "foo(N.array(a))"
##foo(N.array(a))
##print "---------------" 
##
##print "foo(N.array(a, copy=0))"
##foo(N.array(a, copy=0))
##print "---------------" 
##
##print "foo(N.array(a, order='F'))"
##foo(N.array(a,  order='F'))
##print "---------------" 
##
##print "foo(N.array(a, order='F', copy=0))"
##foo(N.array(a,  order='F', copy=0))
##print "------------------------------------------" 

######################

print "foo(b)"
foo(b)
print "---------------" 

print "foo(N.array(b))"
foo(N.array(b))
print "---------------" 

print "foo(N.array(b, copy=0))"
foo(N.array(b, copy=0))
print "---------------" 

print "foo(N.array(b, order='F'))"
foo(N.array(b,  order='F'))
print "---------------" 

print "foo(N.array(b, order='F', copy=0))"
foo(N.array(b,  order='F', copy=0))
print "------------------------------------------" 

######################

print "foo(b[1:, 1:, 1:])"
foo(b[1:, 1:, 1:])
print "---------------" 

print "foo(N.array(b[1:, 1:, 1:]))"
foo(N.array(b[1:, 1:, 1:]))
print "---------------" 

print "foo(N.array(b[1:, 1:, 1:], copy=0))"
foo(N.array(b[1:, 1:, 1:], copy=0))
print "---------------" 

print "foo(N.array(b[1:, 1:, 1:], order='F'))"
foo(N.array(b[1:, 1:, 1:], order='F'))
print "---------------" 

print "foo(N.array(b[1:, 1:, 1:], order='F', copy=0))"
foo(N.array(b[1:, 1:, 1:], order='F', copy=0))
print "------------------------------------------" 

########################

print "foo(c)"
foo(c)
print "---------------" 

print "foo(N.array(c))"
foo(N.array(c))
print "---------------" 

print "foo(N.array(c, copy=0))"
foo(N.array(c, copy=0))
print "---------------" 

print "foo(N.array(c, order='F'))"
foo(N.array(c, order='F'))
print "---------------" 

print "foo(N.array(c, order='F', copy=0))"
foo(N.array(c, order='F', copy=0))
print "------------------------------------------" 
