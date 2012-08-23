import numpy as np
from pwtools import crys

def test_rms():
    natoms = 5
    nstep = 10
    arr1 = np.random.rand(nstep, 3, natoms)
    arr2 = np.random.rand(natoms, nstep, 3)
    arr3 = np.random.rand(natoms, 3, nstep)

    # Test if rms() works.
    r3_rms_all1 = crys.rms(arr3, nitems='all')
    r3_rms_natoms1 = crys.rms(arr3, nitems=natoms)
    r3_rms_all2 = np.sqrt((arr3**2.0).sum() / float(3*natoms*nstep))
    r3_rms_natoms2 = np.sqrt((arr3**2.0).sum() / float(natoms))
    np.testing.assert_almost_equal(r3_rms_all1, r3_rms_all2)
    np.testing.assert_almost_equal(r3_rms_natoms1, r3_rms_natoms2)

    # Test if rms3d() operates correctly along each axis.
    r1_3d = crys.rms3d(arr1, axis=0, nitems='all')
    r2_3d = crys.rms3d(arr2, axis=1, nitems='all')
    r3_3d = crys.rms3d(arr3, axis=2, nitems='all')
    r1_loop = np.empty((nstep,), dtype=float)
    r2_loop = np.empty((nstep,), dtype=float)
    r3_loop = np.empty((nstep,), dtype=float)
    for k in range(nstep):
        r1_loop[k] = crys.rms(arr1[k,...], nitems='all')
        r2_loop[k] = crys.rms(arr2[:,k,:], nitems='all')
        r3_loop[k] = crys.rms(arr3[...,k], nitems='all')
    np.testing.assert_array_almost_equal(r1_3d,r1_loop)
    np.testing.assert_array_almost_equal(r2_3d,r2_loop)
    np.testing.assert_array_almost_equal(r3_3d,r3_loop)

    # Test if rmsd() works.
    #
    # NOTE: Subtle numpy issue here:
    # It is very important NOT to use
    #     R -= R[...,0][...,None]  
    # or 
    #     for k in range(R.shape[-1]):
    #         R[...,k] -= R[...,0][...,None]
    # because R itself is changed in the loop! You have to copy the reference
    # R[...,0] first and then broadcast it for subtracting. What also works is
    # this:
    #     R = R - R[...,0][...,None]
    # HOWEVER, THIS DOES NOT:
    #     for k in range(R.shape[-1]):
    #         R[...,k] = R[...,k] - R[...,0][...,None]
    traj = crys.Trajectory(coords=np.random.rand(nstep, natoms, 3))
    assert traj.timeaxis == 0
    assert traj.nstep == nstep
    from_rmsd = crys.rmsd(traj, ref_idx=0)
    from_loop = np.empty((nstep,), dtype=float)
    from_rms3d = crys.rms3d(traj.coords - traj.coords[0,...][None,...],
                            nitems=natoms, axis=0)
    R = traj.coords.copy()
    ref = R[0,...].copy()
    for k in range(nstep):
        R[k,...] -= ref
        from_loop[k] = np.sqrt((R[k,...]**2.0).sum() / natoms)

    np.testing.assert_array_almost_equal(from_rmsd, from_loop)
    np.testing.assert_array_almost_equal(from_rmsd, from_rms3d)
