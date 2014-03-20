import numpy as np
from pwtools import mpl, num

def test_interpol2d():
    x = np.linspace(-5,5,20) 
    y = x 
    X,Y = np.meshgrid(x,y); X=X.T; Y=Y.T 
    Z = (X+3)**2+(Y+4)**2 + 5 
    dd = mpl.Data2D(X=X,Y=Y,Z=Z)
    
    tgt = np.array([  5.0 ,  30])
    inter = num.Interpol2D(dd=dd, what='rbf_multi') 
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='rbf_gauss')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='bispl')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='linear')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)
    
    inter = num.Interpol2D(dd=dd, what='nearest')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)
    
    try:
        from scipy.interpolate import CloughTocher2DInterpolator
        inter = num.Interpol2D(dd=dd, what='ct')
        np.allclose(inter([[-3,-4],[0,0]]), tgt)
    except ImportError:
        import warnings
        warnings.warn("couldn't import "
            "scipy.interpolate.CloughTocher2DInterpolator")
   
    # API
    inter = num.Interpol2D(xx=dd.xx, yy=dd.yy, values=dd.zz, what='bispl')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)

    inter = num.Interpol2D(points=dd.XY, values=dd.zz, what='bispl')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)

    inter = num.Interpol2D(dd.XY, dd.zz, what='bispl')
    np.allclose(inter([[-3,-4],[0,0]]), tgt)
    np.allclose(inter.get_min(), 5.0)

