import itertools
import numpy as np
from pwtools import mpl
from pwtools.test import tools
rand = np.random.rand

def assert_equal(d1, d2, keys=None):
    if keys is None:
        keys = ['x','y','xx','yy','zz','X','Y','Z','XY']
    tools.assert_dict_with_all_types_equal(d1.__dict__, d2.__dict__,
                                           keys=keys)

def test_input():
    d1 = mpl.get_2d_testdata()
    for k in [d1.x, d1.y, d1.xx, d1.yy, d1.zz, d1.X, d1.Y, d1.Z, d1.XY]:
        assert k is not None

    XY = np.array([k for k in itertools.product(d1.x, d1.y)])
    zz = np.empty(np.prod(d1.Z.shape))
    nx = d1.Z.shape[0]
    ny = d1.Z.shape[1]
    for ix in range(nx):
        zz[ix*ny:(ix+1)*ny] = d1.Z[ix,:]
    xx = XY[:,0]
    yy = XY[:,1]
    assert nx == d1.nx
    assert ny == d1.ny
    assert (d1.xx == xx).all()
    assert (d1.yy == yy).all()
    assert (d1.XY == XY).all()
    assert (d1.zz == d1.Z.flatten()).all()
    assert (d1.zz == zz).all()

    # test various forms of x-y input
    d2 = mpl.Data2D(x=d1.x, y=d1.y, zz=d1.zz)
    assert_equal(d1, d2)
    d2 = mpl.Data2D(x=d1.x, y=d1.y, Z=d1.Z)
    assert_equal(d1, d2)
    d2 = mpl.Data2D(xx=d1.xx, yy=d1.yy, Z=d1.Z)
    assert_equal(d1, d2)
    d2 = mpl.Data2D(X=d1.X, Y=d1.Y, Z=d1.Z)
    assert_equal(d1, d2)
    d2 = mpl.Data2D(XY=d1.XY, Z=d1.Z)
    assert_equal(d1, d2)

    # Z data is optional
    d2 = mpl.Data2D(XY=d1.XY)
    assert_equal(d1, d2, keys=['x','y','xx','yy','X','Y','XY'])


def test_unique_order():
    for x,y in [(np.array([1,2,3,4,5,6]), np.array([1,3,-3,-7,-8])),
                (rand(12), rand(222))]:

        d1 = mpl.Data2D(x=x, y=y)
        d2 = mpl.Data2D(xx=d1.xx, yy=d1.yy)

        # test Data2D._unique()
        #
        # with np.unique()
        #   d1.y = array([1,  3,  -3, -7, -8])
        #   d2.y = np.unique(d1.yy)
        #        = array([-8, -7, -3,  1,  3])
        # instead of
        #   d2.y = Data2D._unique(d1.yy)
        #        = array([1,  3,  -3, -7, -8])
        #        = d1.y
        assert_equal(d1, d2)


def test_copy():
    d1 = mpl.Data2D(x=rand(10), y=rand(12), zz=rand(10*12))
    d2 = d1.copy()
    assert_equal(d1, d2)
    assert d1.zz is not None
    assert d2.zz is not None
    assert d1.Z is  not None
    assert d2.Z is  not None

    # zz and Z is None
    d1 = mpl.Data2D(x=rand(10), y=rand(12))
    d2 = d1.copy()
    assert_equal(d1, d2)
    assert d1.zz is None
    assert d2.zz is None
    assert d1.Z is None
    assert d2.Z is None

    # arrays are not views, changing d2.<attr> will not alter d1.<attr>
    d2.xx *= 2.0
    assert (d1.xx * 2.0 == d2.xx).all()


def test_update():
    x1 = rand(10)
    y1 = rand(12)
    d1 = mpl.Data2D(x=x1, y=y1)
    assert d1.zz is None
    assert d1.Z is None
    d1.update(Z=rand(10,12))
    assert d1.zz is not None
    assert d1.Z is not None
    d2 = d1.copy()
    d2.update(x=rand(10), y=rand(12))
    assert (d1.xx != d2.xx).all()
    assert (d1.yy != d2.yy).all()
    assert (d1.X != d2.X).all()
    assert (d1.Y != d2.Y).all()
    assert (d1.Z == d2.Z).all()
    assert (d1.zz == d2.zz).all()
