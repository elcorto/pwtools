# Test writing AXSF (xcrysden) and XYZ files.
#
# We have no functions yet to read those formats, but try our best to verify
# the written files.

import os
from cStringIO import StringIO
import numpy as np
from pwtools import io, common
from testenv import testdir
pj = os.path.join


def test():
    # --- AXSF ---------------------------------------------------------------
    # cubic, lattice constant = 2
    cell = np.identity(3)*2.0
    # fractional
    coords2d = np.array([[0.5, 0.5, 0.5],
                         [1,1,1]])
    # fractional, 2 time steps: (2,3,2) = (natoms, 3, nstep)
    coords3d= np.dstack((coords2d[...,None],coords2d[...,None]*3))
    # cartesian
    coords3d_cart = np.dot(coords3d.swapaxes(-1,-2), cell).swapaxes(-1,-2)
    coords2d_cart = coords3d_cart[...,0]
    symbols = ['H']*2
    forces2d = np.random.random(coords2d.shape)
    forces3d = np.random.random(coords3d.shape)

    # fixed cell, forces=0
    axsf_fn = pj(testdir, 'foo.axsf')
    io.write_axsf(axsf_fn, 
                  coords=coords3d, 
                  cell=cell,
                  symbols=symbols)
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | tail -n3" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, cell)

    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 CONVVEC %s | tail -n3" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, cell)
    
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.vstack((coords3d_cart[...,0],coords3d_cart[...,1]))
    arr2 = np.concatenate((arr2, np.zeros_like(arr2)), axis=1)
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # fixed cell, forces2d (same force for each time step, possible but
    # actually useless :)
    axsf_fn = pj(testdir, 'foo2.axsf')
    io.write_axsf(axsf_fn, 
                  coords=coords3d, 
                  cell=cell,
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

    # fixed cell, forces3d
    axsf_fn = pj(testdir, 'foo3.axsf')
    io.write_axsf(axsf_fn, 
                  coords=coords3d, 
                  cell=cell,
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
    
    # variable cell, forces3d
    axsf_fn = pj(testdir, 'foo4.axsf')
    cell3d = np.random.rand(3,3,2)
    io.write_axsf(axsf_fn, 
                  coords=coords3d, 
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
    
    # single struct
    axsf_fn = pj(testdir, 'foo5.axsf')
    io.write_axsf(axsf_fn, 
                  coords=coords2d_cart, 
                  cell=cell,
                  symbols=symbols)
    

    # --- XYZ ----------------------------------------------------------------
    cell = np.identity(3)*2.0
    coords2d = np.array([[0.5, 0.5, 0.5],
                         [1,1,1]])
    # (2,3,2) = (natoms, 3, nstep)
    coords3d = np.dstack((coords2d[...,None],coords2d[...,None]*3))
    coords3d_cart = np.dot(coords3d.swapaxes(-1,-2), cell).swapaxes(-1,-2)
    coords2d_cart = coords3d_cart[...,0]
    symbols = ['H']*2
    forces2d = np.random.random(coords2d.shape)
    forces3d = np.random.random(coords3d.shape)
    
    xyz_fn = pj(testdir, 'foo.xyz')
    io.write_xyz(xyz_fn, 
                 coords=coords3d, 
                 cell=cell,
                 symbols=symbols,
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.vstack((coords3d_cart[...,0], coords3d_cart[...,1]))
    print arr
    print arr2
    print "----------------"
    np.testing.assert_array_almost_equal(arr, arr2)

