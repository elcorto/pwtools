from itertools import product
import numpy as np
from pwtools import num

def test_polyfit():
    # 1D
    x = np.linspace(-5,5,100) 
    y = x**2.0 + 1.0
    for deg in [2,3,4,5,6]:
        f = num.PolyFit1D(x,y,deg=deg)
        assert np.allclose(y, f(x))
        assert np.allclose(2.0*x, f(x, der=1))
    
    y = x**2.0 + 1.0
    for deg in [2,3,4,5,6]:
        f = num.PolyFit1D(x,y,deg=deg, levels=5)
        assert np.allclose(y, f(x))
        assert np.allclose(2.0*x, f(x, der=1))
        assert np.allclose(0.0, f.get_min())
    
    y = x**2.0 + 1.0
    f = num.PolyFit1D(x,y,degrange=[4,5,6], levels=5)
    assert np.allclose(y, f(x))
    assert np.allclose(2.0*x, f(x, der=1))
    assert np.allclose(0.0, f.get_min())
    
    # API
    for xs in [2.0, np.array([2.0]), np.array([[2.0]])]:
        assert np.allclose(5.0, f(xs))
        print xs, f(xs), type(f(xs))
        assert type(np.array([5.0])) == type(np.array([f(xs)]))
    
    # API
    y = x**2.0 + 1.0
    f = num.PolyFit1D(x,y,degrange=[4,5,6], levels=5)
    assert np.allclose(f(f.get_min(xtol=1e-10)), 1.0)
    assert np.allclose(f(f.get_min(x0=1.0, tol=1e-10)), 1.0)
    assert np.allclose(f(f.get_min(xab=[-1,1], xtol=1e-10, rtol=1e-16)), 1.0)

    # API
    y = x**2.0 + 1.0
    f = num.PolyFit1D(x,y,deg=4)
    ret = f(x)
    for deg_kwds in [{'degmin': 1, 'degmax': 5}, \
                     {'degrange': [4,5]}]:
        assert np.allclose(ret, num.PolyFit1D(x,y,**deg_kwds)(x))
                             
    y = np.sin(x)
    f = num.PolyFit1D(x,y,degrange=[7,10,15])
    assert np.allclose(y, f(x))
    assert np.allclose(np.cos(x), f(x, der=1), atol=1e-6)
    assert np.allclose(-np.sin(x), f(x, der=2), atol=1e-4)
    
    y = np.sin(x)
    f = num.PolyFit1D(x,y,degmin=2, degmax=15)
    assert np.allclose(y, f(x))
    assert np.allclose(np.cos(x), f(x, der=1), atol=1e-6)   
    assert np.allclose(-np.sin(x), f(x, der=2), atol=1e-4)
    
    # copy self.fit and call avgpolyfit
    y = np.sin(x)
    f = num.PolyFit1D(x,y,degmin=2, degmax=15)
    ret = f(x)
    for i in range(10):
        assert (ret == f(x)).all()
    
    # 2D
    x = np.linspace(-5,5,20)
    y = x
    points = np.array([xy for xy in product(x,y)])
    xx = points[:,0]
    yy = points[:,1]
    
    zz = xx**2.0 + yy**4.0
    f = num.PolyFit(points, zz, degrange=[4])
    assert np.allclose(zz, f(points))
    assert np.allclose(np.array([0.0, 0.0]), f.get_min(), atol=1e-5)

    zz = xx**2.0 + yy**4.0
    f = num.PolyFit(points, zz, degmin=1, degmax=5)
    assert np.allclose(zz, f(points))
    assert np.allclose(np.array([0.0, 0.0]), f.get_min(xtol=1e-10, ftol=1e-8), atol=1e-5)

    for xs in [np.array([2.0,2.0]), np.array([[2.0,2.0]])]:
        assert np.allclose(20.0, f(xs))
        assert type(np.array([20.0])) == type(np.array([f(xs)]))
    
    # mix terms
    zz = xx**2.0 + yy**2.0 + xx*yy**2.0
    f = num.PolyFit(points, zz, deg=2)
    assert np.allclose(zz, f(points))


def test_inner_points_mask():
    # ndim = dimension of the domain, works for > 3 of course, but this is
    # just a test. ndim > 1 uses qhull. ndim==1 requires ordered points.
    for ndim in [1,2,3]:
        a = np.array([x for x in product([0,1,2,3],repeat=ndim)])
        ai = a[num.inner_points_mask(a)]
        assert (ai == np.array([x for x in product([1,2],repeat=ndim)])).all()
    
