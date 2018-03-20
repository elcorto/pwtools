import numpy as np
from pwtools import rbf
rand = np.random.rand

def test_interpol_high_dim():
    # 100 points in 40-dim space
    #
    # Test only API. Interpolating random points makes no sense, of course.
    # However, on the center points (= data points = training points),
    # interpolation must be "perfect".
    X = rand(100,40)
    z = rand(100)
    
    rbfi = rbf.RBFInt(X,z) 
    rbfi.fit(solver='solve') 
    assert np.abs(z - rbfi(X)).max() < 1e-13

def test_2d():
    # 2d example, y = f(x1,x2) = sin(x1) + cos(x2)
    x1 = np.linspace(-1,1,20)
    X1,X2 = np.meshgrid(x1,x1)
    X1 = X1.T
    X2 = X2.T
    z = (np.sin(X1)+np.cos(X2)).flatten()
    X = np.array([X1.flatten(), X2.flatten()]).T
    print(X.shape, z.shape)
    rbfi = rbf.RBFInt(X,z) 
    rbfi.fit()
    # test fit at 300 random points within [-1,1]^2
    Xr = -1.0 + 2*np.random.rand(300,2)
    zr = np.sin(Xr[:,0]) + np.cos(Xr[:,1]) 
    err = np.abs(rbfi(Xr) - zr).max()
    print(err)
    assert err < 1e-6
    # Big errors occur only at the domain boundary: -1, 1, errs at the points
    # should be smaller
    err = np.abs(z - rbfi(X)).max()
    print(err)
    assert err < 1e-7

def test_1d_with_deriv():
    # 1d example, deriv test
    x = np.linspace(0,10,30)
    z = np.sin(x)
    xx = np.linspace(0,10,100)
    rbfi = rbf.RBFInt(x[:,None],z)
    cases = [
        dict(solver='solve'),
        dict(solver='solve', reg=0),
        dict(solver='solve', reg=1e-11),
        dict(solver='lstsq'),
        ]
    for kwds in cases:
        rbfi.fit(**kwds)
        assert np.allclose(rbfi(xx[:,None]), np.sin(xx), rtol=0, atol=1e-4)
        assert np.allclose(rbfi(xx[:,None], der=1)[:,0], np.cos(xx), rtol=0, atol=1e-3)

def test_rbf_func_api():
    X = rand(100,3)
    z = rand(100)
    r1 = rbf.RBFInt(X, z, rbf='multi')
    r2 = rbf.RBFInt(X, z, rbf=rbf.RBFMultiquadric())
    r1.fit()
    r2.fit()
    assert (r1(X) == r2(X)).all()
    
    # make sure we don't share the same RBFFunction
    assert not r1.rbf is r2.rbf
    r1.rbf.param *= 1000
    assert r1.rbf.param != r2.rbf.param
