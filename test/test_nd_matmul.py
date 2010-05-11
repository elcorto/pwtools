import numpy as np

# some random 3d array
natoms = 5
nstep = 10
a = np.ones((natoms, nstep, 3))
for j in xrange(a.shape[1]):
    a[:,j,:] *= (j+1)
    print a[:,j,:]    

print "-----------------------"

# transformation matrix b, transform from cartesian with basis vecs
# np.identity(3) to b
#
# primitive lattice vectors (rows):
# a1 = b[0,:]
# a2 = b[1,:]
# a3 = b[2,:]
#
b = np.array([[1,2,3],[4,5,6],[7,8,9]])

# reference implementation: a[i,j,:] = old postion vector of atom i at time
# step j
print "c0"
c0 = np.zeros(a.shape)
# time steps
for j in xrange(a.shape[1]):
    # atoms
    for i in xrange(a.shape[0]):
        # vector c0[i,j,:] in new coords:
        # a[i,j,0]*a0 + a[i,j,1]*a1 + a[i,j,2]*a2
        for k in xrange(a.shape[2]):
            c0[i,j,:] += a[i,j,k] * b[k,:]
    print c0[:,j,:] 

print "-----------------------"

print "c1"
c1 = np.empty(a.shape)
for j in xrange(a.shape[1]):
    for i in xrange(a.shape[0]):
        c1[i,j,:] = np.dot(a[i,j,:], b)
    print c1[:,j,:] 

print "-----------------------"

print "c2"
c2 = np.empty(a.shape)
for i in xrange(a.shape[0]):
    c2[i,...] = np.dot(a[i,...], b)
for j in xrange(a.shape[1]):
    print c2[:,j,:]

print "-----------------------"

# (m, n, 3) x (3, 3) = (m, n, 3) .. so simple
# vectorization rocks!
print "c3"
c3 = np.dot(a, b)
for j in xrange(a.shape[1]):
    print c3[:,j,:]

print "-----------------------"

print "c4"
c4 = np.empty(a.shape)
for j in xrange(a.shape[1]):
    c4[:,j,:] = np.dot(a[:,j,:], b)
    print c4[:,j,:]

from scipy.linalg import norm
print norm(c0-c1)
print norm(c0-c2)
print norm(c0-c3)
print norm(c0-c4)
