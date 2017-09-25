import os
import numpy as np
from pwtools import common, io, crys

def test_velocity_traj():
    # Test Trajectory.get_velocity() against velocities output from CP2K. The
    # agreement is very good. Works only for fixed-cell MDs, however!
    dr = 'files/cp2k/md/nvt_print_low'
    base = os.path.dirname(dr) 
    fn = '%s/cp2k.out' %dr
    print(common.backtick('tar -C {0} -xzf {1}.tgz'.format(base,dr)))
    tr = io.read_cp2k_md(fn)
    # read from data file
    v1 = tr.velocity.copy()
    # If tr.velocity != None, then get_velocity() doesn't calculate it. Then,
    # it simply returns tr.velocity, which is what we of course usually want.
    tr.velocity = None
    # calculate from coords + time step, b/c of central diffs, only steps 1:-1
    # are the same
    v2 = tr.get_velocity()
    print(">>>> np.abs(v1).max()", np.abs(v1).max())
    print(">>>> np.abs(v1).min()", np.abs(v1).min())
    print(">>>> np.abs(v2).max()", np.abs(v2).max())
    print(">>>> np.abs(v2).min()", np.abs(v2).min())
    print(">>>> np.abs(v1-v2).max()", np.abs(v1-v2).max())
    print(">>>> np.abs(v1-v2).min()", np.abs(v1-v2).min())
    assert np.allclose(v1[1:-1,...], v2[1:-1,...], atol=1e-4)
    
    ##from pwtools import mpl
    ##fig,ax = mpl.fig_ax()
    ##ax.plot(v1[1:-1,:,0], 'b')
    ##ax.plot(v2[1:-1,:,0], 'r')
    ##mpl.plt.show()
    
    shape = (100,10,3)
    arr = np.random.rand(*shape)
    assert crys.velocity_traj(arr, axis=0).shape == shape
    assert crys.velocity_traj(arr, axis=0, endpoints=False).shape == (98,10,3)
