import os
from cStringIO import StringIO
import numpy as np
from pwtools import crys, common
from testenv import testdir
pj = os.path.join

def test():
    cell = np.identity(3)*2.0

    c2d = np.array([[0.5, 0.5, 0.5],
                  [1,1,1]])

    # (2,3,2) = (natoms, 3, nstep)
    coords = np.dstack((c2d[...,None],c2d[...,None]))
    coords_cart = np.dot(coords.swapaxes(-1,-2), cell).swapaxes(-1,-2)
    c2d_cart = coords_cart[...,0]
    symbols = ['H']*2

    axsf_fn = pj(testdir, 'foo.axsf')
    xyz_fn = pj(testdir, 'foo.xyz')
    crys.write_axsf(axsf_fn, 
                    coords=coords, 
                    cell=cell,
                    symbols=symbols)
    crys.write_xyz(xyz_fn, 
                    coords=coords, 
                    cell=cell,
                    symbols=symbols,
                    name='foo') 

    # We have no functions yet to read those formats. Therefore, we do only some
    # very basic checks here.

    # axsf
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | tail -n3" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, cell)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.concatenate((c2d_cart, c2d_cart), axis=0)
    arr2 = np.concatenate((arr2, np.zeros_like(arr2)), axis=1)
    np.testing.assert_array_almost_equal(arr, arr2)

    # xyz
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.concatenate((c2d_cart, c2d_cart), axis=0)
    np.testing.assert_array_almost_equal(arr, arr2)

