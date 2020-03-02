from itertools import product

import numpy as np
from pwtools import num

rand = np.random.rand


def test_round_mult():
    assert num.round_up_next_multiple(144,8) == 144
    assert num.round_up_next_multiple(145,8) == 152


def test_euler_matrix():
    # http://mathworld.wolfram.com/EulerAngles.html

    # 0 degree rotation
    assert np.allclose(num.euler_matrix(0,0,0), np.identity(3))

    # degree and radians API
    degs = rand(3)
    degs[0] *= 360
    degs[1] *= 180
    degs[2] *= 360
    rads = np.radians(degs)
    assert np.allclose(num.euler_matrix(*rads),
                       num.euler_matrix(*degs, deg=True))
    # rotation about z
    vec = np.array([0,0,1.0])
    rr = rand(50)
    for ri in rr:
        rot = num.euler_matrix(ri*2*np.pi,0,0)
        assert (np.dot(rot,vec) == vec).all()

    # rotation about x'
    vec = np.array([1.0,0,0])
    rr = rand(50)
    for ri in rr:
        rot = num.euler_matrix(0, ri*np.pi,0)
        assert (np.dot(rot,vec) == vec).all()

    # rotation about z'
    vec = np.array([0,0,1.0])
    rr = rand(50)
    for ri in rr:
        rot = num.euler_matrix(0, 0, ri*2*np.pi)
        assert (np.dot(rot,vec) == vec).all()


def test_inner_points_mask():
    # ndim = dimension of the domain, works for > 3 of course, but this is
    # just a test. ndim > 1 uses qhull. ndim==1 requires ordered points.
    for ndim in [1,2,3]:
        a = np.array([x for x in product([0,1,2,3],repeat=ndim)])
        ai = a[num.inner_points_mask(a)]
        assert (ai == np.array([x for x in product([1,2],repeat=ndim)])).all()
