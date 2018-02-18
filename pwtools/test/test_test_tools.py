# Test for testing the test tools ... uhhh boy!

import numpy as np
import copy
from pwtools.test import tools

def test_tools():
    x1 = {'a': 3,
          type(1): 'foo',
          'b': {'aa': 1,  
                'bb': np.array([1,2]),
                'cc': {'aaa': np.array([1,2.0]),
                       'bbb': np.array([2,4.0])}}}
    x2 = copy.deepcopy(x1)
    tools.assert_dict_with_all_types_equal(x1, x2)
    tools.assert_all_types_equal(x1, x2)
    
    # not equal array
    x2['b']['cc']['bbb'] *= 2.0
    assert not tools.all_types_equal(x1, x2)
    
    # almost equal array and float
    x2 = copy.deepcopy(x1)
    x2['b']['cc']['bbb'] += 1e-5
    x2['b']['aa'] += 1e-5
    tools.assert_all_types_almost_equal(x1, x2)
    
    # sub-dict different (keys don't match)
    x2 = copy.deepcopy(x1)
    x2['b']['cc']['fff'] = 'extra'
    assert not tools.all_types_equal(x1, x2)
    
    # test only some keys of a dict
    tools.assert_dict_with_all_types_equal({'a':1,'b':1}, {'a':1, 'b':3},
        keys=['a'])

    # simple stuff
    tools.assert_all_types_equal(1, 1)
    tools.assert_all_types_equal(1.0, 1.0)
    tools.assert_all_types_equal(1.0, 1)
    tools.assert_all_types_equal(1, 1.0)
    tools.assert_all_types_equal([1], [1])
    tools.assert_all_types_equal([1], [1.0])
    tools.assert_all_types_equal('a', 'a')
    tools.assert_all_types_almost_equal(1.0, 1.0+1e-5)
    tools.assert_all_types_almost_equal(np.array([1.0]), np.array([1.0+1e-5]))
    assert not tools.all_types_equal(1, 2)
    assert not tools.all_types_equal(1.0, 1.1)
    assert not tools.all_types_equal([1], [1,2])
    assert not tools.all_types_equal('a', 'b')
   
    # different types not allowed if strict=True
    assert not tools.all_types_equal(1.0, 1, strict=True)
    
    # test keys=[...], i.e. ignore some keys in both dicts
    x2 = copy.deepcopy(x1)
    x2['c'] = 1.0
    assert tools.dict_with_all_types_equal(x1, x2, keys=['a','b',type(1)])

    # fail on same array content but different dimensions
    a = np.random.rand(1,2,3)
    b = a[None,...]
    # this is True and IMHO a numpy bug b/c the dimensions are
    # different
    assert (a==b).all()
    # must catch that here
    assert not tools.array_equal(a,b)

    a = np.random.rand(1,2,3)
    b = (a + 1e-8)[None,...]
    assert not (a==b).all() # make sure they are numerically differrent
    assert np.allclose(a,b) # True but should be False
    assert not tools.array_almost_equal(a,b) # ok

    flt = 1.0
    np_flt = np.array([1.0+1e-9])[0]
    assert not tools.all_types_equal(flt, np_flt) 
    assert not tools.all_types_equal(np_flt, flt) 
    assert tools.all_types_almost_equal(flt, np_flt) 
    assert tools.all_types_almost_equal(np_flt, flt) 
