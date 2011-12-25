# Test writing AXSF (xcrysden) and XYZ files.
#
# We have no functions yet to read those formats, but try our best to verify
# the written files.

import os
from cStringIO import StringIO
import numpy as np
from pwtools import io, common, crys
from testenv import testdir
pj = os.path.join


def test():
    # --- AXSF ---------------------------------------------------------------
    cell2d = np.random.rand(3,3)
    cell3d = np.random.rand(3,3,2)
    # fractional
    coords2d_frac = np.array([[0.5, 0.5, 0.5],
                              [1,1,1]])
    # fractional, 2 time steps: (2,3,2) = (natoms, 3, nstep)
    coords3d_frac = np.dstack((coords2d_frac[...,None],
                               coords2d_frac[...,None]*0.8))
    # cartesian = coords3d_frac + cell2d (fixed cell). For varialbe cell cases
    # below, cell3d is used!
    coords3d_cart = crys.coord_trans(coords3d_frac, 
                                     old=cell2d, 
                                     new=np.identity(3),
                                     axis=1)
    coords2d_cart = coords3d_cart[...,0]
    symbols = ['H']*2
    forces2d = np.random.random(coords2d_frac.shape)
    forces3d = np.random.random(coords3d_frac.shape)

    # fixed cell, forces=0
    axsf_fn = pj(testdir, 'foo.axsf')
    io.write_axsf(axsf_fn, 
                  coords_frac=coords3d_frac, 
                  cell=cell2d,
                  symbols=symbols)
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | tail -n3" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, cell2d)

    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 CONVVEC %s | tail -n3" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, cell2d)
    
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.vstack((coords3d_cart[...,0],coords3d_cart[...,1]))
    arr2 = np.concatenate((arr2, np.zeros_like(arr2)), axis=1)
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # fixed cell, forces2d, coords_frac (same force for each time step,
    # possible but actually useless :)
    axsf_fn = pj(testdir, 'foo2.axsf')
    io.write_axsf(axsf_fn, 
                  coords_frac=coords3d_frac, 
                  cell=cell2d,
                  symbols=symbols,
                  forces=forces2d)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    tmp_c = np.vstack((coords3d_cart[...,0], coords3d_cart[...,1]))
    tmp_f = np.vstack((forces2d,)*2)
    arr2 = np.hstack((tmp_c, tmp_f))
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)

    # fixed cell, forces3d, coords_frac
    axsf_fn = pj(testdir, 'foo3.axsf')
    io.write_axsf(axsf_fn, 
                  coords_frac=coords3d_frac, 
                  cell=cell2d,
                  symbols=symbols,
                  forces=forces3d)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    t0 = np.concatenate((coords3d_cart[...,0], forces3d[...,0]), axis=1)
    t1 = np.concatenate((coords3d_cart[...,1], forces3d[...,1]), axis=1)
    arr2 = np.vstack((t0,t1))
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # variable cell, forces3d, coords_frac
    axsf_fn = pj(testdir, 'foo4.axsf')
    io.write_axsf(axsf_fn, 
                  coords_frac=coords3d_frac, 
                  cell=cell3d,
                  symbols=symbols,
                  forces=forces3d)
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -v -e '--' -e 'PRIMVEC'" %axsf_fn)))
    arr2 = np.vstack((cell3d[...,0], cell3d[...,1]))           
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    t0 = np.concatenate((np.dot(coords3d_frac[...,0], cell3d[...,0]), 
                         forces3d[...,0]), axis=1)
    t1 = np.concatenate((np.dot(coords3d_frac[...,1], cell3d[...,1]), 
                         forces3d[...,1]), axis=1)
    arr2 = np.vstack((t0,t1))
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # variable cell, forces3d, coords_cart (pure random data, cell3d and
    # coords3d_cart are not related, just make sure that no coords trans is
    # done, cell is only written to PRIMVEC etc)
    axsf_fn = pj(testdir, 'foo5.axsf')
    io.write_axsf(axsf_fn, 
                  coords_cart=coords3d_cart, 
                  cell=cell3d,
                  symbols=symbols,
                  forces=forces3d)
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -v -e '--' -e 'PRIMVEC'" %axsf_fn)))
    arr2 = np.vstack((cell3d[...,0], cell3d[...,1]))           
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    t0 = np.concatenate((coords3d_cart[...,0], forces3d[...,0]), axis=1)
    t1 = np.concatenate((coords3d_cart[...,1], forces3d[...,1]), axis=1)
    arr2 = np.vstack((t0,t1))
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # single struct, coords_cart
    axsf_fn = pj(testdir, 'foo6.axsf')
    io.write_axsf(axsf_fn, 
                  coords_cart=coords2d_cart, 
                  cell=cell2d,
                  symbols=symbols,
                  forces=forces2d)
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -v -e '--' -e 'PRIMVEC'" %axsf_fn)))
    arr2 = cell2d           
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.concatenate((coords2d_cart, forces2d), axis=1)
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)
    

    # --- XYZ ----------------------------------------------------------------
    # Use cell, coords from above

    # input: coords_frac
    symbols = ['H']*2
    xyz_fn = pj(testdir, 'foo_frac_input.xyz')
    io.write_xyz(xyz_fn, 
                 coords_frac=coords3d_frac, 
                 cell=cell2d,
                 symbols=symbols,
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.vstack((coords3d_cart[...,0], coords3d_cart[...,1]))
    np.testing.assert_array_almost_equal(arr, arr2)

    # input: coords_cart, cell=None
    xyz_fn = pj(testdir, 'foo_cart_input.xyz')
    io.write_xyz(xyz_fn, 
                 coords_cart=coords3d_cart, 
                 symbols=symbols,
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.vstack((coords3d_cart[...,0], coords3d_cart[...,1]))
    np.testing.assert_array_almost_equal(arr, arr2)

