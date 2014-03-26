import numpy as np
from pwtools import mpl, num

def return_min(inter):
    # Return scalar minimum instead of array (5.0 instead of [5.0]).    
    return inter(inter.get_min(maxfun=1e6, maxiter=1e2))[0]

def test_interpol2d():
    x = np.linspace(-5,5,20) 
    y = x 
    X,Y = np.meshgrid(x,y); X=X.T; Y=Y.T 
    Z = (X+3)**2+(Y+4)**2 + 5 
    dd = mpl.Data2D(X=X,Y=Y,Z=Z)
    
    tgt = np.array([  5.0 ,  30])
    inter = num.Interpol2D(dd=dd, what='rbf_multi') 
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='rbf_inv_multi') 
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='rbf_gauss')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0, atol=1e-5)
    
    inter = num.Interpol2D(dd=dd, what='bispl')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)
    
    # linear, ct and nearest are very inaccurate, use only for plotting!
    inter = num.Interpol2D(dd=dd, what='linear')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt, atol=5e-1)
    assert np.allclose(return_min(inter), 5.0, atol=1e-1)
    
    # don't even test accuracy here
    inter = num.Interpol2D(dd=dd, what='nearest')
    
    try:
        from scipy.interpolate import CloughTocher2DInterpolator
        inter = num.Interpol2D(dd=dd, what='ct')
        assert np.allclose(inter([[-3,-4],[0,0]]), tgt,
            atol=1e-1)
        assert np.allclose(return_min(inter), 5.0, atol=1e-1)
    except ImportError:
        import warnings
        warnings.warn("couldn't import "
            "scipy.interpolate.CloughTocher2DInterpolator")
   
    # API
    inter = num.Interpol2D(xx=dd.xx, yy=dd.yy, values=dd.zz, what='bispl')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)

    inter = num.Interpol2D(points=dd.XY, values=dd.zz, what='bispl')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)

    inter = num.Interpol2D(dd.XY, dd.zz, what='bispl')
    assert np.allclose(inter([[-3,-4],[0,0]]), tgt)
    assert np.allclose(return_min(inter), 5.0)

