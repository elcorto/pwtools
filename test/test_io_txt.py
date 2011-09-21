# Test new text input and output

from StringIO import StringIO

def test():
    from pwtools import io
    import os
    import numpy as np
    from testenv import testdir
    pj = os.path.join

    def write_read_check(fn, arr, type='txt', axis=-1):
        print fn + ' ...'
        io.writearr(fn, arr, type=type, axis=axis)
        a = io.readarr(fn, type=type)
        assert (a == arr).all()

    def write_read_check_raw(fn, arr, axis=None, shape=None):
        print fn + ' ...'
        io.writetxt(fn, arr, axis=axis)
        # ignore file header
        a = io.readtxt(fn, axis=axis, shape=shape)
        assert (a == arr).all()

    # 1d
    a = np.arange(0, 3)
    fn = pj(testdir, 'a1d.txt')
    write_read_check(fn, a)

    # 2d
    shape = (3, 5)
    a = np.arange(0, np.prod(shape)).reshape(shape) 
    fn = pj(testdir, 'a2d.txt')
    write_read_check(fn, a)

    # 3d
    shape = (3, 5, 7)
    a = np.arange(0, np.prod(shape)).reshape(shape)

    fn = pj(testdir, 'a3d0.txt')
    write_read_check(fn, a, axis=0)

    fn = pj(testdir, 'a3d1.txt')
    write_read_check(fn, a, axis=1)

    fn = pj(testdir, 'a3d2.txt')
    write_read_check(fn, a, axis=2)

    fn = pj(testdir, 'a3dm1.txt')
    write_read_check(fn, a, axis=-1)

    fn = pj(testdir, 'a3d0r.txt')
    write_read_check_raw(fn, a, axis=0, shape=shape)

    fn = pj(testdir, 'a3d1r.txt')
    write_read_check_raw(fn, a, axis=1, shape=shape)

    fn = pj(testdir, 'a3d2r.txt')
    write_read_check_raw(fn, a, axis=2, shape=shape)
    
    # API
    shape = (3, 5)
    arr = np.arange(0, np.prod(shape)).reshape(shape)
    fn = pj(testdir, 'a2d_api.txt')
    fh = open(fn, 'w')
    fh.write('@@ some comment\n')
    fh.write('@@ some comment\n')
    fh.write('@@ some comment\n')
    np.savetxt(fh, arr)
    fh.close()
    a = io.readtxt(fn, shape=shape, axis=-1, comments='@@')
    assert (a == arr).all()

    txt = "1.0 2.0 3\n4   5   6\n"
    arr = io.readtxt(StringIO(txt), shape=(2,3), axis=-1, dtype=float)
    assert arr.dtype == np.array([1.0]).dtype
    arr = io.readtxt(StringIO(txt), shape=(2,3), axis=-1, dtype=int)
    assert arr.dtype == np.array([1]).dtype
