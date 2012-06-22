import numpy as np
from pwtools import rbf
rand = np.random.rand

def test_rbf():
    # 100 points in 40-dim space
    #
    # Test only API. Interpolating random points makes no sense, of course.
    # However, on the center points (= data points = training points),
    # interpolation must be "perfect".
    X = rand(100,40)
    Y = rand(100)
    
    rbfi = rbf.RBFInt(X,Y) 
    rbfi.train() 
    assert np.abs(Y - rbfi(X)).max() < 1e-13

    # 2d example, y = f(x1,x2) = x1**2 + x2**2
    x = np.linspace(-1,1,20)
    y = x
    X1,X2 = np.meshgrid(x,y)
    X1 = X1.T
    X2 = X2.T
    Y = (np.sin(X1)+np.cos(X2)).flatten()
    X = np.array([X1.flatten(), X2.flatten()]).T
    print X.shape, Y.shape
    rbfi = rbf.RBFInt(X,Y) 
    rbfi.train()
    tol = 1e-7
    err = rbfi(np.atleast_2d([0,0]))[0] - 1.0
    print err
    assert err < tol
    err = rbfi(np.atleast_2d([.5,.3]))[0] - (np.sin(0.5)+np.cos(0.3))
    print err
    assert err < tol
    # Big errors occur only at the domain boundary: -1, 1
    err = np.abs(Y - rbfi(X)).max()
    print err
    assert err < tol
    
    # Test train_param(). Note that in general, the `param` from param='est'
    # is already quite good and fmin() doesn't gain much. In fact, we have to
    # use a *higher* `tol` in order for all tests to pass :) This is only an API
    # test mostly, which we want to be fast, not accurate. 
    # Play w/ fmin keywords (ftol, ...) to tune the optimization.
    rbfi = rbf.train_param(X,Y,pattern='rand', randskip=0.2, shuffle=False)
    tol = 1e-5
    err = rbfi(np.atleast_2d([0,0]))[0] - 1.0
    print err
    assert err < tol
    err = rbfi(np.atleast_2d([.5,.3]))[0] - (np.sin(0.5)+np.cos(0.3))
    print err
    assert err < tol
    # Big errors occur only at the domain boundary: -1, 1
    err = np.abs(Y - rbfi(X)).max()
    print err
    assert err < tol    
    
    # 1d example, deriv test
    X = np.linspace(0,10,20)
    Y = np.sin(X)
    XI = np.linspace(0,10,100)
    rbfi = rbf.RBFInt(X[:,None],Y)
    rbfi.train()
    assert np.allclose(rbfi(XI[:,None]), np.sin(XI), rtol=1e-3, atol=1e-3)
    assert np.allclose(rbfi(XI[:,None], der=1)[:,0], np.cos(XI), rtol=1e-2, atol=1e-3)


