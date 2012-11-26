from itertools import product
import numpy as np
from pwtools import num

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
    levels = [{}, {'levels': 2}]
    degs = [{'deg': 2}, {'degrange': [2,3,4,5]}, 
            {'degmin': 1, 'degmax': 5}]
    kwd_lst = make_kwd_lst(scale, levels, degs)        
    for kwds in kwd_lst:
        print kwds
        # 1D
        x = np.linspace(-5,5,100) 
        y = x**2.0 + 1.0
        f = num.PolyFit1D(x,y,**kwds)
        assert np.allclose(y, f(x))
        assert np.allclose(2.0*x, f(x, der=1))
        assert np.allclose(2.0, f(x, der=2))
        assert np.allclose(1.0, f(f.get_min()))
        assert np.allclose(1.0, f(f.get_min(xtol=1e-10)))                         
        assert np.allclose(1.0, f(f.get_min(x0=1.0, tol=1e-10)))                  
        assert np.allclose(1.0, f(f.get_min(xab=[-1,1], xtol=1e-10, rtol=1e-16))) 
        
        # API
        for xs in [2.0, np.array([2.0]), np.array([[2.0]])]:
            assert np.allclose(5.0, f(xs))
            assert type(np.array([5.0])) == type(np.array([f(xs)]))

        # copy self.fit and call avgpolyfit
        y = np.sin(x)
        f = num.PolyFit1D(x,y,**kwds)
        ret = f(x)
        for i in range(10):
            assert (ret == f(x)).all()
    
    
    scale = [{'scale': False}, {'scale': True}]
    levels = [{}, {'levels': 1}]
    degs = [{'deg': 4}, {'degrange': [3,4]}, 
            {'degmin': 2, 'degmax': 5}]
    kwd_lst = make_kwd_lst(scale, levels, degs)        
    for kwds in kwd_lst:
        print kwds
        # 2D
        x = np.linspace(-5,6,20)
        y = np.linspace(-2,4,22)
        points = np.array([xy for xy in product(x,y)])
        xx = points[:,0]
        yy = points[:,1]
        zz = (xx-2)**2.0 + (yy-1)**4.0 + 1.0
        f = num.PolyFit(points, zz, **kwds)
        assert np.allclose(zz, f(points))
        print f.get_min(xtol=1e-10, ftol=1e-10)
        assert np.allclose(np.array([2.0, 1.0]), f.get_min(xtol=1e-10, ftol=1e-8), atol=1e-3)
        assert np.allclose(1.0, f(f.get_min()))

        for xs in [np.array([4.0,3.0]), np.array([[4.0,3.0]])]:
            assert np.allclose(21.0, f(xs))
            assert type(np.array([21.0])) == type(np.array([f(xs)]))
        
        # mix terms
        zz = xx**2.0 + yy**2.0 + xx*yy**2.0
        f = num.PolyFit(points, zz, **kwds)
        assert np.allclose(zz, f(points),atol=1e-3)

def test_inner_points_mask():
    # ndim = dimension of the domain, works for > 3 of course, but this is
    # just a test. ndim > 1 uses qhull. ndim==1 requires ordered points.
    for ndim in [1,2,3]:
        a = np.array([x for x in product([0,1,2,3],repeat=ndim)])
        ai = a[num.inner_points_mask(a)]
        assert (ai == np.array([x for x in product([1,2],repeat=ndim)])).all()
    
