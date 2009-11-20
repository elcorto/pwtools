# Test new text input and output

from pwtools import pydos as pd
import os
import numpy as np

pj = os.path.join
rand = np.random.rand

def write_read_check(fn, arr, type='txt', axis=-1):
    print fn + ' ...'
    pd.writearr(fn, arr, type=type, axis=axis)
    a = pd.readarr(fn, type=type)
    if (a == arr).all():
        print "... ok"
    else:
        print "... FAIL!"

dir = '/tmp/pwtools_test'
if not os.path.exists(dir):
    os.makedirs(dir)

# 1d
a = rand(10)
fn = pj(dir, 'a1d.txt')
write_read_check(fn, a)

# 2d
a = rand(10, 20)
fn = pj(dir, 'a2d.txt')
write_read_check(fn, a)

# 3d
a = rand(10, 20, 30)

fn = pj(dir, 'a3d0.txt')
write_read_check(fn, a, axis=0)

fn = pj(dir, 'a3d1.txt')
write_read_check(fn, a, axis=1)

fn = pj(dir, 'a3d2.txt')
write_read_check(fn, a, axis=2)

fn = pj(dir, 'a3dm1.txt')
write_read_check(fn, a, axis=-1)

