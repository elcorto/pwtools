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
    Y = rand(100)
    
    rbfi = rbf.RBFInt(X,Y) 
    rbfi.train(mode='linalg', solver='solve') 
    assert np.abs(Y - rbfi(X)).max() < 1e-13

def test_2d():
    # 2d example, y = f(x1,x2) = x1**2 + x2**2
    x = np.linspace(-1,1,20)
    y = x
    X1,X2 = np.meshgrid(x,y)
    X1 = X1.T
    X2 = X2.T
    Y = (np.sin(X1)+np.cos(X2)).flatten()
    X = np.array([X1.flatten(), X2.flatten()]).T
    print(X.shape, Y.shape)
    rbfi = rbf.RBFInt(X,Y) 
    rbfi.train()
    tol = 1e-7
    err = rbfi(np.atleast_2d([0,0]))[0] - 1.0
    print(err)
    assert err < tol
    err = rbfi(np.atleast_2d([.5,.3]))[0] - (np.sin(0.5)+np.cos(0.3))
    print(err)
    assert err < tol
    # Big errors occur only at the domain boundary: -1, 1
    err = np.abs(Y - rbfi(X)).max()
    print(err)
    assert err < tol

def test_1d_with_deriv():
    # 1d example, deriv test
    X = np.linspace(0,10,20)
    Y = np.sin(X)
    XI = np.linspace(0,10,100)
    rbfi = rbf.RBFInt(X[:,None],Y)
    for solver in ['solve', 'lstsq']:
        rbfi.train(mode='linalg', solver=solver)
        assert np.allclose(rbfi(XI[:,None]), np.sin(XI), rtol=1e-6, atol=1e-3)
        assert np.allclose(rbfi(XI[:,None], der=1)[:,0], np.cos(XI), rtol=1e-2, atol=1e-4)
