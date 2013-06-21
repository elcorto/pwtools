import os
import numpy as np
from pwtools.pydos import velocity
from pwtools import common, io


def test_velocity_n1():
    # Test most simple minded finite diff velocities.
    coords = np.arange(2*3*6).reshape(2,3,6)
    v1 = velocity(coords, axis=-1)
    v2 = coords[...,1:] - coords[...,:-1]
    v3 = np.diff(coords, n=1, axis=-1)
    assert v1.shape == v2.shape == v3.shape == (2,3,5)
    assert (v1 == v2).all()
    assert (v1 == v3).all()

def test_velocity_traj():
    # Test Trajectory.get_velocity() against velocities output from CP2K. The
    # agreement is very good. Works only for fixed-cell MDs, however!
    dr = 'files/cp2k/md/nvt_print_low'
    base = os.path.dirname(dr) 
    fn = '%s/cp2k.out' %dr
    print common.backtick('tar -C {} -xzf {}.tgz'.format(base,dr))
    tr = io.read_cp2k_md(fn)
    # read from data file
    v1 = tr.velocity.copy()
    # If tr.velocity != None, then get_velocity() doesn't calculateb it. The,
    # it simply returns tr.velocity, which is what we of course usually want.
    tr.velocity = None
    # calculate from coords + time step, b/c of central diffs, only steps 1:-1
    # are the same
    v2 = tr.get_velocity()
    assert np.allclose(v1[1:-1,...], v2[1:-1,...], atol=1e-4)

    ##from pwtools import mpl
    ##fig,ax = mpl.fig_ax()
    ##ax.plot(v1[1:-1,:,0], 'b')
    ##ax.plot(v2[1:-1,:,0], 'r')
    ##mpl.plt.show()
