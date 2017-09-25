# Test array text input and output

from io import StringIO
from pwtools import arrayio, parse
import os
import numpy as np
from .testenv import testdir
pj = os.path.join
rand = np.random.rand

def write_read_check(fn, arr, axis=-1, shape=None):
    print(fn + ' ...')
    arrayio.writetxt(fn, arr, axis=axis)
    a = arrayio.readtxt(fn, axis=axis, shape=shape)
    assert (a == arr).all()


def test_io_txt():
    # 1d
    a = np.arange(0, 3)
    fn = pj(testdir, 'a1d.txt')
    write_read_check(fn, a)

    # 2d
    shape = (3, 5)
    a = np.arange(0, np.prod(shape)).reshape(shape) 
    fn = pj(testdir, 'a2d.txt')
    write_read_check(fn, a)

    # 3d, store array along axis 0,1,2,-1
    shape = (3, 5, 7)
    a = np.arange(0, np.prod(shape)).reshape(shape)
    
    # ignore file header if shape != None
    for sh in [None, shape]:
        fn = pj(testdir, 'a3d0.txt')
        write_read_check(fn, a, axis=0, shape=sh)
        fn = pj(testdir, 'a3d1.txt')
        write_read_check(fn, a, axis=1, shape=sh)
        fn = pj(testdir, 'a3d2.txt')
        write_read_check(fn, a, axis=2, shape=sh)
        fn = pj(testdir, 'a3dm1.txt')
        write_read_check(fn, a, axis=-1, shape=sh)
    
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
    a = arrayio.readtxt(fn, shape=shape, axis=-1, comments='@@')
    assert (a == arr).all()

    txt = "1.0 2.0 3\n4   5   6\n"
    arr = arrayio.readtxt(StringIO(txt), shape=(2,3), axis=-1, dtype=float)
    assert arr.dtype == np.array([1.0]).dtype
    # Apparently in Python 2.7: 
    #   float('1.0') -> 1.0
    #   int('1.0')  -> ValueError: invalid literal for int() with base 10: '1.0'
    #   int(float('1.0')) -> 1
    # We need to (ab)use converters. Ugh.  
    arr = arrayio.readtxt(StringIO(txt), shape=(2,3), axis=-1, dtype=int,
                          converters=dict((x,lambda a: int(float(a))) for x in [0,1,2]))
    assert arr.dtype == np.array([1]).dtype


def test_traj_from_txt():
    shape = (10,20,30)
    arr3d_orig = np.random.rand(*shape)
    axis = 0
    # general stuff for axis != 0, here for axis=0 we have written_shape==shape
    shape_2d_chunk = shape[:axis] + shape[(axis+1):]
    written_shape = (shape[axis],) + shape_2d_chunk
    print("axis, written_shape:", axis, written_shape)
    fn = pj(testdir, 'arr_test_traj_from_txt_axis%i.txt' %axis)
    arrayio.writetxt(fn, arr3d_orig, axis=axis, header=False)
    with open(fn) as fd:
        arr3d = parse.traj_from_txt(fd.read(), axis=axis,
                                    shape=written_shape)
        fd.seek(0)                                    
        # test if the "old" way of reading also works 
        arr3d_readtxt = arrayio.readtxt(fd, shape=written_shape, axis=axis)
        assert (arr3d_readtxt == arr3d_orig).all()    
    # now test traj_from_txt result 
    assert arr3d.shape == arr3d_orig.shape, \
           ("axis={0}, shapes: read={1} written={2} orig={3}".format(axis,
                  arr3d.shape, written_shape, arr3d_orig.shape))
    assert (arr3d == arr3d_orig).all()

