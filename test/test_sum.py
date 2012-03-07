import numpy as np
from pwtools.num import sum
from pwtools.test.tools import aaae, aae
rand = np.random.rand

def test():
    arr = rand(2,3,4)
    
    # this all goes thru np.sum(), must produce exact same result
    assert sum(arr) == sum(arr, axis=None) == arr.sum()
    
    # must use aae() b/c summing order is apparently different in
    # np.sum(axis=None) -> small numerical noise
    aae(sum(arr, axis=(0,1,2)), arr.sum())

    aaae(sum(arr, axis=0), arr.sum(axis=0))
    aaae(sum(arr, axis=1), arr.sum(axis=1))
    aaae(sum(arr, axis=2), arr.sum(axis=2))

    aaae(sum(arr, axis=-1), arr.sum(axis=-1))
    aaae(sum(arr, axis=-2), arr.sum(axis=-2))
    
    aaae(sum(arr, axis=(0,)), arr.sum(axis=0))
    aaae(sum(arr, axis=(1,)), arr.sum(axis=1))
    aaae(sum(arr, axis=(2,)), arr.sum(axis=2))
    
    assert sum(arr, axis=(0,1)).shape == (4,)
    assert sum(arr, axis=(0,2)).shape == (3,)
    assert sum(arr, axis=(1,2)).shape == (2,)

    aaae(sum(arr, axis=(0,1)), arr.sum(axis=0).sum(axis=0))
    aaae(sum(arr, axis=(0,2)), arr.sum(axis=0).sum(axis=1))
    aaae(sum(arr, axis=(1,2)), arr.sum(axis=1).sum(axis=1))
    
    assert sum(arr, axis=(0,), keepdims=True).shape == (2,)
    assert sum(arr, axis=(1,), keepdims=True).shape == (3,)
    assert sum(arr, axis=(2,), keepdims=True).shape == (4,)
    
    assert sum(arr, axis=(0,1), keepdims=True).shape == (2,3)
    assert sum(arr, axis=(0,2), keepdims=True).shape == (2,4)
    assert sum(arr, axis=(1,2), keepdims=True).shape == (3,4)
    
    aaae(sum(arr, axis=(0,1)), sum(arr, axis=(1,0)))
    aaae(sum(arr, axis=(0,2)), sum(arr, axis=(2,0)))
    aaae(sum(arr, axis=(1,2)), sum(arr, axis=(2,1)))
