# Test new text input and output

from pwtools import io
import os
import numpy as np

pj = os.path.join
##rand = np.random.rand

def write_read_check(fn, arr, type='txt', axis=-1):
    print fn + ' ...'
    io.writearr(fn, arr, type=type, axis=axis)
    a = io.readarr(fn, type=type)
    if (a == arr).all():
        print "... ok"
    else:
        print "... FAIL!"

def write_read_check_raw(fn, arr, axis=None, shape=None):
    print fn + ' ...'
    io.writetxt(fn, arr, axis=axis)
    # ignore file header
    a = io.readtxt(fn, axis=axis, shape=shape)
    if (a == arr).all():
        print "... ok"
    else:
        print "... FAIL!"

dir = '/tmp/pwtools_test'
if not os.path.exists(dir):
    os.makedirs(dir)

# 1d
a = np.arange(0, 3)
fn = pj(dir, 'a1d.txt')
write_read_check(fn, a)

# 2d
shape = (3, 5)
a = np.arange(0, np.prod(shape)).reshape(shape) 
fn = pj(dir, 'a2d.txt')
write_read_check(fn, a)

# 3d
shape = (3, 5, 7)
a = np.arange(0, np.prod(shape)).reshape(shape)

fn = pj(dir, 'a3d0.txt')
write_read_check(fn, a, axis=0)

fn = pj(dir, 'a3d1.txt')
write_read_check(fn, a, axis=1)

fn = pj(dir, 'a3d2.txt')
write_read_check(fn, a, axis=2)

fn = pj(dir, 'a3dm1.txt')
write_read_check(fn, a, axis=-1)

fn = pj(dir, 'a3d0r.txt')
write_read_check_raw(fn, a, axis=0, shape=shape)

fn = pj(dir, 'a3d1r.txt')
write_read_check_raw(fn, a, axis=1, shape=shape)

fn = pj(dir, 'a3d2r.txt')
write_read_check_raw(fn, a, axis=2, shape=shape)

