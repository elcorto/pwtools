import os
import numpy as np
from pwtools import io
from pwtools.test import tools
from testenv import testdir
rand = np.random.rand

def test_h5():
    try:
        import h5py
        # all keys start without '/' b/c that is what read_h5() returns
        dct = {'a': rand(4),
               'b/c/x1': 3,
               'b/c/x2': rand(2,3),
               }
        h5fn = os.path.join(testdir, 'test.h5')
        io.write_h5(h5fn, dct)
        dct1 = dct       
        dct2 = io.read_h5(h5fn)
        tools.assert_all_types_equal(dct1, dct2)
        
        dct1 = dict(((k, dct[k]) for k in ['b/c/x1', 'b/c/x2']))       
        for group in ['/b/c/', '/b/c', 'b/c']:
            dct2 = io.read_h5(h5fn, group=group)
            tools.assert_all_types_equal(dct1, dct2)
    
        dct1 = {'x1': dct['b/c/x1'], 'x2': dct['b/c/x2']}      
        for group in ['/b/c/', '/b/c', 'b/c']:
            dct2 = io.read_h5(h5fn, group=group, rel=True)
            tools.assert_all_types_equal(dct1, dct2)
    
    except ImportError:
        print "skipping test_h5, no h5py importable"
