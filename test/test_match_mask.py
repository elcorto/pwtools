import numpy as np
from pwtools import num    

def test_match_mask():
    msk = np.array([ True, False,  True, False, False], dtype=bool)
    idx = np.array([0, 2])
    arr = np.array([1,2,3,4,5]) 
    values = np.array([1,3])
    assert (num.match_mask(arr, values) == msk).all()
    ret = num.match_mask(arr, values, fullout=True)
    assert (ret[0] == msk).all()
    assert (ret[1] == idx).all()
    assert (arr[msk] == np.array([1, 3])).all()
    
    # handle cases where len(values) > len(arr) and values not contained in arr
    values = np.array([1,3,3,3,7,9,-3,-4,-5])
    ret = num.match_mask(arr, values, fullout=True)
    assert (ret[0] == msk).all()
    assert (ret[1] == idx).all()
    
    # float values: use eps
    ret = num.match_mask(arr+0.1, values, fullout=True, eps=0.2)
    assert (ret[0] == msk).all()
    assert (ret[1] == idx).all()
