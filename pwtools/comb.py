import numpy as np
from pwtools.common import assert_cond as _assert
from pwtools import common
from functools import reduce
import itertools

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


# XXX remove? seems unused
def _swap(arr, i, j):
    """Swap arr[i] <-> arr[j]. Return i-j-swapped copy of `arr`.
    
    Parameters
    ----------
    arr : 1d list
    
    Notes
    -----
    Must return a copy of arr b/c the swap is an in-place operation and lists
    are mutable.
    """        
    # aa = a.copy() for np arrays
    # aa = a[:]     for lists
    aa = a[:]
    tmp = aa[i]
    aa[i] = aa[j]
    aa[j] = tmp
    return aa

# XXX remove? use itertools.permutations()??
def ipermute(seq):
    """Calculate all N! permutations of sequence `seq` (return generator
    object). This function was written before itertools.permutations()
    existed, which is actually much faster.

    Parameters
    ----------
    seq : 1d list to permute, len(seq) = N
    
    Examples
    --------
    >>> import itertools
    >>> [x for x in ipermute([1,2,3])]
    [[1, 2, 3], [2, 1, 3], [3, 1, 2], [1, 3, 2], [2, 3, 1], [3, 2, 1]]
    >>> [x for x in itertools.permutations([1,2,3])]
    [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    # only unique ones
    >>> set([x for x in itertools.permutations([1,0,0])])
    set([(0, 0, 1), (0, 1, 0), (1, 0, 0)])

    Notes
    -----
    algo : We use the Counting QuickPerm Algorithm [1]_. Works like Heap's
        Algorithm [2]_ [3]_  but not recursive. Uses N!-1 swaps.

    References
    ----------
    .. [1] http://permute.tchs.info/quickperm.php
    .. [2] www.cs.princeton.edu/~rs/talks/perms.pdf, page 12
    .. [3] http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
    """
    # copy input
    aa = list(seq)[:]
    n = len(aa)
    p = [0]*n
    i = 1
    # first permutation is the input
    yield aa[:]
    while i < n:
        if p[i] < i:
            if (i % 2) == 0:
                j = 0
            else:
                j = p[i]
            # copy
            # swap a[i] <-> a[j]
            tmp = aa[j]
            aa[j] = aa[i]
            aa[i] = tmp
            p[i] += 1
            i = 1
            yield aa[:]
        else:
            p[i] = 0
            i += 1

# XXX remove? use itertools.permutations()??
def permute(*args, **kwargs):
    """See ipermute()."""
    return [x for x in ipermute(*args, **kwargs)]


def unique2d(arr, what='row'):
    """Reduce 2d array `arr` to a 2d array with unique rows (or cols).

    Parameters
    ----------
    arr : 2d-like
    what : str
        {'row', 'col'}
    
    Returns
    -------
    numpy 2d array

    Examples
    --------
    >>> a=array([[1,2,3], [1,2,3], [1,2,4]])
    >>> unique2d(a, 'row')
    array([[1, 2, 3],
           [1, 2, 4]])
    
    Notes
    -----
    # These do the same:
    >>> unique2d(permute(a)) # uses more memory, slower
    >>> permute(a, skip_equal=True))
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


def _ensure_list(arg):
    if common.is_seq(arg):
        return [_ensure_list(xx) for xx in arg]
    else:
        return arg


# XXX return iterator (py3)
def nested_loops(lists, flatten=False):
    """Nested loops, optional flattening.
    
    Parameters
    ----------
    lists : list of sequences
        The objects to permute. len(lists) == the depth (nesting levels) of the
        equivalent nested loops. Individual lists may contain a mix of
        different types/objects, e.g. [['a', 'b'], [Foo(), Bar(), Baz()],
        [1,2,3,4,5,6,7]].
    flatten : bool
        Flatten each entry in returned list. 
    
    Returns
    -------
    list : nested lists 

    Examples
    --------
    >>> from pwtools import comb
    >>> comb.nested_loops([[1,2],['a','b']])
    [[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b']]

    # If values of different lists should be varied together, use zip(). Note
    # that you get nested lists back. Use flatten=True to get flattened lists.
    >>> comb.nested_loops([(1,2), zip(['a','b'],(np.sin,np.cos))])
    [[1, ['a', <ufunc 'sin'>]],
     [1, ['b', <ufunc 'cos'>]],
     [2, ['a', <ufunc 'sin'>]],
     [2, ['b', <ufunc 'cos'>]]]

    >>> comb.nested_loops([(1,2), zip(['a','b'],(np.sin,np.cos))], flatten=True)
    [[1, 'a', <ufunc 'sin'>],
     [1, 'b', <ufunc 'cos'>],
     [2, 'a', <ufunc 'sin'>],
     [2, 'b', <ufunc 'cos'>]]
    """
    perms = itertools.product(*lists)
    ret = [common.flatten(xx) for xx in perms] if flatten else perms
    return _ensure_list(ret)


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
        print(aa)

# XXX remove
# alias
factorial = fac

# XXX remove
if __name__ == '__main__':
    import sys
    sys.exit(main())
