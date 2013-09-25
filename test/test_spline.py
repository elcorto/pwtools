import numpy as np
from scipy.interpolate import splev
from pwtools.num import Spline

def test_spline():
    x = np.linspace(0,10,100)
    y = np.sin(x)
    spl = Spline(x,y)
    assert (y - spl(x) < spl.eps).all()
    assert (y - spl.splev(x) < spl.eps).all()
    assert (y - splev(x, spl.tck) < spl.eps).all()
    assert not spl.is_mono()
    np.testing.assert_array_almost_equal(spl(x,der=1), np.cos(x), decimal=4)

    x = np.linspace(0,10,100)
    y = x**2.0 - 5
    spl = Spline(x,y)
    assert spl.is_mono()
    
    y = -(x**2.0 - 5)
    spl = Spline(x,y)
    assert spl.is_mono()
    
    y0s = [5,0,-40]
    xabs = [[0,2], [1,3], [6,8]]
    x0s = [1,2,7]
    # use bracket [x[0], x[-1]] for brentq()
    for y0 in y0s:
        np.testing.assert_almost_equal(y0, spl(spl.invsplev(y0)))
    # use smaller bracket
    for y0,xab in zip(y0s, xabs):
        np.testing.assert_almost_equal(y0, spl(spl.invsplev(y0, xab=xab)))
    # use start guess for newton() 
    for y0,x0 in zip(y0s, x0s):
        np.testing.assert_almost_equal(y0, spl(spl.invsplev(y0, x0=x0)))
    
    # root    
    np.testing.assert_almost_equal(spl.invsplev(0.0), spl.get_root())
    
    # min
    x = np.linspace(-10,10,100) 
    y = (x-5)**2.0 + 1.0
    spl = Spline(x,y)
    xmin = spl.get_min()
    ymin = spl(xmin)
    np.testing.assert_almost_equal(xmin, 5.0)
    np.testing.assert_almost_equal(ymin, 1.0)

    # API
    spl = Spline(x,y,k=2,s=0.1,eps=0.11)
    for kw in ['k', 's']:
        assert kw in spl.splrep_kwargs.keys()
    assert spl.splrep_kwargs['k'] == 2       
    assert spl.splrep_kwargs['s'] == 0.1

