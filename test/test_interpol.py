import numpy as np
from pwtools import mpl, num
from pwtools.test.tools import aaae

def test_interpol2d():
    x = np.linspace(-5,5,20) 
    y = x 
    X,Y = np.meshgrid(x,y); X=X.T; Y=Y.T 
    Z = (X+3)**2+(Y+4)**2 + 5 
    dd = mpl.Data3D(X=X,Y=Y,Z=Z)
    
    tgt = np.array([  5.0 ,  30])
    inter = num.Interpol2D(dd, what='rbf_multi') 
    aaae(inter([[-3,-4],[0,0]]), tgt)
    inter = num.Interpol2D(dd, what='rbf_gauss')
    aaae(inter([[-3,-4],[0,0]]), tgt, decimal=5)
    inter = num.Interpol2D(dd, what='ct')
    aaae(inter([[-3,-4],[0,0]]), tgt, decimal=2)
    inter = num.Interpol2D(dd, what='bispl')
    aaae(inter([[-3,-4],[0,0]]), tgt)

