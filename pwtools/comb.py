import numpy as np
from pwtools.common import assert_cond as _assert
from pwtools import common
from functools import reduce
import itertools


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
# legacy, we keep it for now b/c it is used in batch.py
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
