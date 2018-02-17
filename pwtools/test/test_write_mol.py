# Test writing AXSF (xcrysden) and XYZ files.
#
# We have no functions yet to read those formats, but try our best to verify
# the written files.
#
# units: write_axsf() converts forces from Structure / Trajectory's eV / Ang to
# Ha / Ang (for XSF). We use `units` to revert that transformarion for easier
# testing.

import os
from io import StringIO
import numpy as np
from pwtools import io, common, crys
from .testenv import testdir
from pwtools.crys import Structure, Trajectory
from pwtools.constants import Ha, eV
pj = os.path.join


def test_write_mol():
    units={'forces': Ha / eV}
    nstep = 2
    cell2d = np.random.rand(3,3)
    cell3d = np.random.rand(nstep,3,3)
    # fractional
    coords2d_frac = np.array([[0.5, 0.5, 0.5],
                              [1,1,1]])
    # fractional, 2 time steps: (2,2,3) = (nstep, natoms, 3)
    coords3d_frac = np.array([coords2d_frac, coords2d_frac*0.8])
    # cartesian = coords3d_frac + cell2d (fixed cell). For varialbe cell cases
    # below, cell3d is used!
    coords3d_cart = crys.coord_trans(coords3d_frac, 
                                     old=cell2d, 
                                     new=np.identity(3),
                                     axis=-1)
    coords2d_cart = coords3d_cart[0,...]
    symbols = ['H']*2
    forces2d = np.random.random(coords2d_frac.shape) 
    forces3d = np.random.random(coords3d_frac.shape)

    # --- AXSF ---------------------------------------------------------------
    # fixed cell, forces=0
    axsf_fn = pj(testdir, 'foo.axsf')
    io.write_axsf(axsf_fn, 
                  Trajectory(units=units,
                             coords_frac=coords3d_frac, 
                             cell=cell2d,
                             symbols=symbols),
                 )                                    
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -vE -e '--|PRIMVEC'" %axsf_fn)))
    np.testing.assert_array_almost_equal(arr, np.concatenate((cell2d, cell2d), axis=0))

    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.vstack((coords3d_cart[0,...],coords3d_cart[1,...]))
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # fixed cell, forces3d, coords_frac
    axsf_fn = pj(testdir, 'foo3.axsf')
    io.write_axsf(axsf_fn, 
                  Trajectory(units=units,coords_frac=coords3d_frac, 
                             cell=cell2d,
                             symbols=symbols,
                             forces=forces3d),
                 )                             
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    t0 = np.concatenate((coords3d_cart[0,...], forces3d[0,...]), axis=-1)
    t1 = np.concatenate((coords3d_cart[1,...], forces3d[1,...]), axis=-1)
    arr2 = np.vstack((t0,t1))
    print(arr)
    print(arr2)
    print("----------------")
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # variable cell, forces3d, coords_frac
    axsf_fn = pj(testdir, 'foo4.axsf')
    io.write_axsf(axsf_fn, 
                  Trajectory(units=units,coords_frac=coords3d_frac, 
                             cell=cell3d,
                             symbols=symbols,
                             forces=forces3d))
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -v -e '--' -e 'PRIMVEC'" %axsf_fn)))
    arr2 = np.vstack((cell3d[0,...], cell3d[1,...]))           
    print(arr)
    print(arr2)
    print("----------------")
    np.testing.assert_array_almost_equal(arr, arr2)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    t0 = np.concatenate((np.dot(coords3d_frac[0,...], cell3d[0,...]), 
                         forces3d[0,...]), axis=-1)
    t1 = np.concatenate((np.dot(coords3d_frac[1,...], cell3d[1,...]), 
                         forces3d[1,...]), axis=-1)
    arr2 = np.vstack((t0,t1))
    print(arr)
    print(arr2)
    print("----------------")
    np.testing.assert_array_almost_equal(arr, arr2)
    
    # single struct, coords_cart
    axsf_fn = pj(testdir, 'foo6.axsf')
    io.write_axsf(axsf_fn, 
                  Structure(units=units,coords=coords2d_cart, 
                            cell=cell2d,
                            symbols=symbols,
                            forces=forces2d))
    arr = np.loadtxt(StringIO(
            common.backtick("grep -A3 PRIMVEC %s | grep -v -e '--' -e 'PRIMVEC'" %axsf_fn)))
    arr2 = cell2d           
    print(arr)
    print(arr2)
    print("----------------")
    np.testing.assert_array_almost_equal(arr, arr2)
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %axsf_fn)))
    arr2 = np.concatenate((coords2d_cart, forces2d), axis=1)
    print(arr)
    print(arr2)
    print("----------------")
    np.testing.assert_array_almost_equal(arr, arr2)
    

    # --- XYZ ----------------------------------------------------------------
    # Use cell, coords, etc from above

    # input: coords_frac
    xyz_fn = pj(testdir, 'foo_frac_input.xyz')
    io.write_xyz(xyz_fn, 
                 Trajectory(units=units,coords_frac=coords3d_frac, 
                            cell=cell2d,
                            symbols=symbols),
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.concatenate([coords3d_cart[0,...], coords3d_cart[1,...]], axis=0)
    np.testing.assert_array_almost_equal(arr, arr2)

    # input: coords_cart, cell=None
    xyz_fn = pj(testdir, 'foo_cart_input.xyz')
    io.write_xyz(xyz_fn, 
                 Trajectory(units=units,coords=coords3d_cart, 
                            symbols=symbols),
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = np.concatenate((coords3d_cart[0,...], coords3d_cart[1,...]), axis=0)
    np.testing.assert_array_almost_equal(arr, arr2)

    # input: coords2d_frac, cell=cell2d
    xyz_fn = pj(testdir, 'foo_cart_input.xyz')
    io.write_xyz(xyz_fn, 
                 Structure(units=units,coords_frac=coords2d_frac, 
                           cell=cell2d,
                           symbols=symbols),
                 name='foo') 
    arr = np.loadtxt(StringIO(
            common.backtick("sed -nre 's/^H(.*)/\\1/gp' %s" %xyz_fn)))
    arr2 = coords2d_cart
    np.testing.assert_array_almost_equal(arr, arr2)

