#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Some combinatorics stuff.
#-----------------------------------------------------------------------------

import numpy as np
from common import assert_cond as _assert

def fac(n):
    """Factorial n!. Returns integer."""
    _assert(isinstance(n, int), '`n` must be int')
    _assert(n >= 0, '`n` must be >= 0')
    if n == 1 or n == 0:
        return 1
    return n * fac(n-1) 

#-----------------------------------------------------------------------------

def binom(n, k):
    """Binomial coefficient.
    fac(n)/(fac(k)*fac(n-k))
    """
    _assert(n >= k >= 0, 'illegal input, only n >= k >= 0')
    return fac(n)/(fac(k)*fac(n-k))

#-----------------------------------------------------------------------------

def _swap(a, i, j):
    """Swap a[i] <-> a[j]. Return i-j-swapped copy of `a`.
    
    args:
        a -- 1D numpy array
    
    notes:
        `a` is global in permute() and is the current permutation. Must use
        copy() b/c in-place operation on `a` would reflect to *all* other `a`
        arrays which are already in `lst`. Think of `a` as a pointer like in C.
        Stuff like `for i in range(..): lst.append(a)` would result in a "list
        of pointers" which all point to the same array.
    """        
    aa = a.copy()
    tmp = aa[i]
    aa[i] = aa[j]
    aa[j] = tmp
    return aa

#-----------------------------------------------------------------------------

def permute(a, id=True):
    """Store all N! permutations of `a` in a list.
    
    args:
        a -- 1D numpy array, list, ...
        id -- bool: if False, store all perms but not the identity (i.e. a
              itself), so len(lst) = N!-1
    notes:
        We use the Countdown QuickPerm Algorithm [1]. Works like Heap's
        Algorithm [2,3] so it's a nonrecursive version of it. Uses N!-1 swaps.
     
        [1] http://www.geocities.com/permute_it/ 
        [2] www.cs.princeton.edu/~rs/talks/perms.pdf, page 12
        [3] http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
    """
    a = np.asarray(a)
    if id:
        lst = [a]
    else:
        lst = []
    n = len(a)
    p = np.arange(n+1)
    i = 1
    while i < n:
        p[i] -= 1
        if i % 2 == 1:
            j = p[i]
        else:
            j = 0
        a = _swap(a, j, i)
        lst.append(a)
        # restore p to start value [0, 1, ..., n]
        i = 1
        while p[i] == 0:
            p[i] = i
            i += 1
    return lst


#-----------------------------------------------------------------------------

def kron(a, b):
    """Kronecker symbol for scalars and arrays.
    
    >>> kron(1, 2)
    0
    >>> kron(2, 2)
    1
    >>> a = array([1,2,3])
    >>> b = array([1,2,0])
    >>> kron(a,b)
    [1,1,0]
    """
    if np.isscalar(a):
        if a == b:
            return 1
        else:        
            return 0        
    else:
        # >>> vstack((a, -b))
        # array([[1,  2,  3],
        #       [-1, -2,  0]])
        # >>> vstack((a,-b)).sum(axis=0)
        # array([0, 0, 3])
        # >>> z
        # array([0, 0, 0])
        # >>> z[tmp==0] = 1
        # array([1, 1, 0])
        tmp = np.vstack((a, -b)).sum(axis=0)
        z = np.zeros(len(tmp), dtype=int)
        z[tmp==0] = 1
    return z

#-----------------------------------------------------------------------------

##def nest(lst, limit, i, cnt):
##    
##    cnt += 1
##    if cnt > 27:
##        return
##    
##
##    if lst == limit:
##        return
##    
##    if i > len(lst)-1:
##        i=0 
##    
##    if lst[i] == limit[i]:
##        lst[i] = 0
##        if i < len(lst)-1:
##            lst[i+1] += 1
##        nest(lst, limit, i+1, cnt)
##    else:
##        lst[i] += 1
##        nest(lst, limit, i, cnt)

##def nest():
##    for i in range(2):
##        for j in range(2):
##            for k in range(2):
##                print i,j,k
##
##def nest2(lev):
##    if lev <= 0:
##        print lev
##    else:
##        for i in range(2):
##            nest2(lev-1)
##

#-----------------------------------------------------------------------------

def main():
    a = np.array([0,1,2])
    lst = permute(a)
    for aa in lst:
        print aa

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(main())
