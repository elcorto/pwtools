from itertools import product

import numpy as np
from pwtools import num
from pwtools.test.tools import assert_all_types_equal

rand = np.random.rand

def make_kwd_lst(*args):
    kwd_lst = []
    for tup in product(*args):
        tmp = {}
        for dct in tup:
            tmp.update(dct)
        kwd_lst.append(tmp)
    return kwd_lst


def test_polyfit():
    scale = [{'scale': False}, {'scale': True}]
    scale_vand = [{'scale_vand': False}, {'scale_vand': True}]
    degs = [{'deg': 2}, {'deg': 3}]
    kwd_lst = make_kwd_lst(scale, degs, scale_vand)
    for kwds in kwd_lst:
        print(kwds)
        # 1D
        x = np.linspace(-5,5,100)
        y = x**2.0 - 1.0
        f = num.PolyFit1D(x,y,**kwds)
        assert np.allclose(y, f(x))
        assert np.allclose(2.0*x, f(x, der=1))
        assert np.allclose(2.0, f(x, der=2))
        assert np.allclose(-1.0, f(f.get_min()))
        assert np.allclose(-1.0, f(f.get_min(xtol=1e-10)))
        assert np.allclose(-1.0, f(f.get_min(x0=1.0, tol=1e-10)))
        assert np.allclose(-1.0, f(f.get_min(xab=[-1,1], xtol=1e-10, rtol=1e-14)))

        # API: PolyFit1D __call__ arg: scalar, 1d, 2d
        for xs in [2.0, np.array([2.0]), np.array([[2.0]])]:
            assert np.allclose(3.0, f(xs))
            assert type(np.array([3.0])) == type(np.array([f(xs)]))

        # API: 3rd arg is always 'deg'
        fit1 = num.polyfit(x[:,None],y,2)
        fit2 = num.polyfit(x[:,None],y,deg=2)
        assert_all_types_equal(fit1, fit2)

    degs = [{'deg': 4}]
    kwd_lst = make_kwd_lst(scale, degs, scale_vand)
    for kwds in kwd_lst:
        print(kwds)
        # 2D
        x = np.linspace(-5,6,20)
        y = np.linspace(-2,4,22)
        points = np.array([xy for xy in product(x,y)])
        xx = points[:,0]
        yy = points[:,1]
        zz = (xx-2)**2.0 + (yy-1)**4.0 - 1.0
        f = num.PolyFit(points, zz, **kwds)
        assert np.allclose(zz, f(points))
        print(f.get_min(xtol=1e-10, ftol=1e-10))
        assert np.allclose(np.array([2.0, 1.0]), f.get_min(xtol=1e-10, ftol=1e-8), atol=1e-3)
        assert np.allclose(-1.0, f(f.get_min()))

        for xs in [np.array([4.0,3.0]), np.array([[4.0,3.0]])]:
            assert np.allclose(19.0, f(xs))
            assert type(np.array([19.0])) == type(np.array([f(xs)]))

        # mix terms
        zz = xx**2.0 + yy**2.0 + xx*yy**2.0
        f = num.PolyFit(points, zz, **kwds)
        assert np.allclose(zz, f(points),atol=1e-3)


def test_compare_numpy():
    x = np.sort(np.random.rand(10))
    y = np.random.rand(10)
    yy1 = np.polyval(np.polyfit(x, y, 3), x)
    scale = [{'scale': False}, {'scale': True}]
    scale_vand = [{'scale_vand': False}, {'scale_vand': True}]
    kwd_lst = make_kwd_lst(scale, scale_vand)
    for kwds in kwd_lst:
        yy2 = num.PolyFit1D(x, y, 3, **kwds)(x)
        assert np.allclose(yy1, yy2)


def test_scale_get_min():
    # two local minima separated by a max at ~6.5
    x = np.linspace(2,10,50)
    y = np.cos(x)+x*.2
    xlo = 2.94080857
    xhi = 9.2237442
    scale = [{'scale': False}, {'scale': True}]
    scale_vand = [{'scale_vand': False}, {'scale_vand': True}]
    kwd_lst = make_kwd_lst(scale, scale_vand)
    for kwds in kwd_lst:
        f = num.PolyFit(x[:,None], y, deg=9, **kwds)
        assert np.allclose(f.get_min(x0=4), np.array([xlo]))
        assert np.allclose(f.get_min(x0=8), np.array([xhi]))
        f = num.PolyFit1D(x, y, deg=9, **kwds)
        assert np.allclose(f.get_min(x0=4), xlo)
        assert np.allclose(f.get_min(x0=8), xhi)


def test_api():
    # PolyFit's call method returns a scalar when given a "scalar" input (a
    # single x point). All other interpolators return a length-1 array instead.
    # We need to ckeck where this specific behavior is used in code using
    # pwtools before we can deal with it. Until then, we have this test to make
    # sure it behaves as it does.
    #
    #   >>> f=num.Interpol2D(dd=dd, what='bispl')
    #   >>> f.get_min()
    #   array([ 4.58624188, -3.59544779])
    #   >>> f(f.get_min())
    #   array([-2.13124442])
    #
    # BUT
    #
    #   >>> f=num.PolyFit(dd.XY, dd.zz, deg=5)
    #   >>> f(f.get_min())
    #   -2.1312444200439651
    #
    # ... scalar (numpy.float64)
    #
    # Note that PolyFit1D.get_min() returns a float instead!
    #
    # This has not been a problem, at least in tests, since
    # >>> np.allclose(1.0, array([1.0]))
    # True
    #
    x = np.linspace(-5,5,20)
    y = x**2
    f = num.PolyFit(x[:,None], y, deg=2)
    assert f.get_min().shape == (1,)
    assert f(x[:,None]).shape == x.shape
    assert f(np.array([1.23])).shape == ()
    assert f(np.array([[1.23]])).shape == (1,)

    f = num.PolyFit1D(x, y, deg=2)
    assert f(1.23).shape == ()
    # vectorized eval
    assert f(rand(3)).shape == (3,)
