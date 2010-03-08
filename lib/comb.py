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


def binom(n, k):
    """Binomial coefficient ("n over k").
    fac(n)/(fac(k)*fac(n-k))
    """
    _assert(n >= k >= 0, 'illegal input, only n >= k >= 0')
    return fac(n)/(fac(k)*fac(n-k))


def _swap(a, i, j):
    """Swap a[i] <-> a[j]. Return i-j-swapped copy of `a`.
    
    args:
    -----
    a : 1D numpy array
    
    notes:
    ------
    `a` is global in permute() and is the current permutation. Must use copy()
    b/c in-place operation on `a` would reflect to *all* other `a` arrays which
    are already in `lst`. Think of `a` as a pointer like in C. Stuff like `for
    i in range(..): lst.append(a)` would result in a "list of pointers" which
    all point to the same array.
    """        
    aa = a.copy()
    tmp = aa[i]
    aa[i] = aa[j]
    aa[j] = tmp
    return aa


def permute(a, id=True):
    """Store all N! permutations of `a` in a list.
    
    args:
    -----
    a : 1D numpy array, list, ...
    id : bool 
        False : store all perms but not the identity (i.e. a
            itself), so len(lst) = N!-1
    notes:
    ------
    We use the Countdown QuickPerm Algorithm [1]. Works like Heap's Algorithm
    [2,3] so it's a nonrecursive version of it. Uses N!-1 swaps.
 
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


def unique2d(arr, what='row'):
    """Reduce 2d array `arr` to a 2d array with unique rows (or cols).

    args:
    -----
    arr : 2d-like
    what : str
        {'row', 'col'}

    example:
    --------
    >>> a=array([[1,2,3], [1,2,3], [1,2,4]])
    >>> reduce_equal(a, 'row')
    array([[1, 2, 3],
           [1, 2, 4]])
    """
    if what == 'row':
        arr = np.asarray(arr)
    elif what == 'col':
        arr = np.asarray(arr).T
    else:
        raise ValueError("illegal value of 'what': %s" %what)
    uniq = [arr[0,:]]
    for row_a in arr:
        is_in = False
        for row_u in uniq:
            if (row_a == row_u).all():
                is_in = True
                break
        if not is_in:
            uniq.append(row_a)
    if what == 'row':            
        return np.asarray(uniq)            
    else:        
        return np.asarray(uniq).T


def nested_loops(lists, ret_all=False):
    """Nonrecursive version of nested loops of arbitrary depth. Pure Python
    version (no numpy).
    
    args:
    -----
    lists : list of lists 
        The objects to permute. len(lists) == the depth (nesting levels) of the
        equivalent nested loops. Individual lists may contain a mix of
        different types/objects, e.g. [['a', 'b'], [Foo(), Bar(), Baz()],
        [1,2,3,4,5,6,7]].
    ret_all : bool
        True: return perms, perm_idxs
        False: return perms
    
    returns:
    --------
    perms : list of lists with permuted objects
    perm_idxs : list of lists with indices of the permutation

    example:
    --------
    >>> a=[1,2]; b=[3,4]; c=[5,6];
    >>> perms=[]
    >>> for aa in a:
    ....:   for bb in b:
    ....:       for cc in c:
    ....:           perms.append([aa,bb,cc])
    ....:             
    >>> perms
    [[1, 3, 5],
     [1, 3, 6],
     [1, 4, 5],
     [1, 4, 6],
     [2, 3, 5],
     [2, 3, 6],
     [2, 4, 5],
     [2, 4, 6]]
    >>> nested_loops([a,b,c], ret_all=True)
    ([[1, 3, 5],
      [1, 3, 6],
      [1, 4, 5],
      [1, 4, 6],
      [2, 3, 5],
      [2, 3, 6],
      [2, 4, 5],
      [2, 4, 6]],
     [[0, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
      [0, 1, 1],
      [1, 0, 0],
      [1, 0, 1],
      [1, 1, 0],
      [1, 1, 1]])
    >>> nested_loops([[1,2], ['a','b','c'], [sin, cos]])
    [[1, 'a', <ufunc 'sin'>],
     [1, 'a', <ufunc 'cos'>],
     [1, 'b', <ufunc 'sin'>],
     [1, 'b', <ufunc 'cos'>],
     [1, 'c', <ufunc 'sin'>],
     [1, 'c', <ufunc 'cos'>],
     [2, 'a', <ufunc 'sin'>],
     [2, 'a', <ufunc 'cos'>],
     [2, 'b', <ufunc 'sin'>],
     [2, 'b', <ufunc 'cos'>],
     [2, 'c', <ufunc 'sin'>],
     [2, 'c', <ufunc 'cos'>]]
    # if values of different lists should be varied together
    >>> nested_loops([zip([1,2], ['a', 'b']), [88, 99]])
    [[(1, 'a'), 88], [(1, 'a'), 99], [(2, 'b'), 88], [(2, 'b'), 99]]
    >>> from pwtools import common
    >>> for x in nested_loops([zip([1,2], ['a', 'b']), [88, 99]]): 
    ...     print common.flatten(x)
    ...
    [1, 'a', 88]
    [1, 'a', 99]
    [2, 'b', 88]
    [2, 'b', 99]
    """
    lens = map(len, lists)
    mx_idxs = [x - 1 for x in lens]
    # nperms = numpy.prod(lens)
    nperms = reduce(lambda x,y: x*y, lens)
    # number of nesting levels
    nlevels = len(lists)
    # index into `lists`: lists[i][j] -> lists[i][idxs[i]], i.e.
    # idxs[i] is the index into the ith list
    idxs = [0]*nlevels
    perm_idxs = []
    perms = []
    # e.g. [2,1,0]
    rev_rlevels = range(nlevels)[::-1]
    for i in range(nperms):         
        for pos in rev_rlevels:
            if idxs[pos] > mx_idxs[pos]:
                idxs[pos] = 0
                # pos - 1 never gets < 0 before all possible `nlevels`
                # permutations are generated.
                idxs[pos-1] += 1
        # [:] to append a copy                
        perm_idxs.append(idxs[:])
        perms.append([lists[j][k] for j,k in enumerate(idxs)])
        idxs[-1] += 1
    if ret_all:
        return perms, perm_idxs
    else:
        return perms


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


def main():
    a = np.array([0,1,2])
    lst = permute(a)
    for aa in lst:
        print aa


if __name__ == '__main__':
    import sys
    sys.exit(main())
