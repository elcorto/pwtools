import os
import numpy as np
from pwtools import io
from pwtools.test import tools
from testenv import testdir
rand = np.random.rand

def test_h5():
    try:
        import h5py
        dct1 = \
            {'/a': 'abcgs',
             '/b/c/x1': 3,
             '/b/c/x2': rand(2,3),
             }
        # writing a dct w/o leading slash will always be read back in *with*
        # leading slash             
        dct2 = \
            {'a': 'abciqo4iki',
             'b/c/x1': 3,
             'b/c/x2': rand(2,3),
             }
        for idx,dct in enumerate([dct1, dct2]):             
            h5fn = os.path.join(testdir, 'test_%i.h5' %idx)
            io.write_h5(h5fn, dct)
            read_dct = io.read_h5(h5fn)
            for kk in read_dct.keys():
                assert kk.startswith('/')
            for kk in dct.keys():
                key = '/'+kk if not kk.startswith('/') else kk
                tools.assert_all_types_equal(dct[kk], read_dct[key])
        
        # write mode='a', test appending
        h5fn = os.path.join(testdir, 'test_append.h5')
        io.write_h5(h5fn, {'/a': 1.0})
        read_dct = io.read_h5(h5fn)
        assert read_dct.keys() == ['/a']
        assert read_dct['/a'] == 1.0
        # append '/b', using {'/a': 1.0, '/b': 2.0} would be an error since /a
        # already exists, use mode='w' then, but this overwrites all!
        io.write_h5(h5fn, {'/b': 2.0}, mode='a')
        read_dct2 = io.read_h5(h5fn)
        # sort(...): sort possible [/b, /a] -> [/a, /b]
        assert np.sort(np.array(read_dct2.keys())).tolist() == ['/a', '/b']
        assert read_dct2['/a'] == 1.0
        assert read_dct2['/b'] == 2.0
        # overwrite
        io.write_h5(h5fn, {'/b': 22.0, '/c': 33.0}, mode='w')
        read_dct3 = io.read_h5(h5fn)
        assert np.sort(np.array(read_dct3.keys())).tolist() == ['/b', '/c']

    except ImportError:
        tools.skip("skipping test_h5, no h5py importable")
