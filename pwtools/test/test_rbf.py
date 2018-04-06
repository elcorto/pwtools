import numpy as np
from pwtools import rbf
rand = np.random.rand

def test_interpol_high_dim():
    # 100 points in 40-dim space
    #
    # Test only API. Interpolating random points makes no sense, of course.
    # However, on the center points (= data points = training points),
    # interpolation must be "perfect".
    X = rand(10,40)
    z = rand(10)
    
    rbfi = rbf.Rbf(X,z) 
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
    rbfi = rbf.Rbf(X, z, p=1.5)  
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
    assert err < 1e-6


def test_api_and_all_types_and_1d_with_deriv():
    # 1d example, deriv test
    x = np.linspace(0,10,50)
    z = np.sin(x)
    xx = np.linspace(0,10,100)
    for name in rbf.rbf_dct.keys():
        print(name)
        cases = [
            (True,  dict(rbf=name)),
            (True,  dict(p='mean')),
            (False, dict(p='scipy')), # not accurate enough, only API test
            (True,  dict(rbf=name, r=1e-11)),
            (True,  dict(rbf=name, p=rbf.estimate_p(x[:,None]))),         
            (True,  dict(rbf=name, p=rbf.estimate_p(x[:,None]), r=1e-11)),
            ]
        for go, kwds in cases:
            rbfi = rbf.Rbf(x[:,None], z, **kwds)
            if go:
                assert np.allclose(rbfi(xx[:,None]), np.sin(xx), rtol=0, atol=1e-4)
                assert np.allclose(rbfi(xx[:,None], der=1)[:,0], np.cos(xx), rtol=0, atol=1e-3)


def test_p_api():
    X = rand(10,3)
    z = rand(10)
    for name in ['scipy', 'mean']:
        f = rbf.Rbf(X, z, p=name)
        assert f.p == rbf.estimate_p(X, method=name)


def test_func_api():
    X = rand(10,3)
    z = rand(10)
    r1 = rbf.Rbf(X, z, rbf='multi')
    r2 = rbf.Rbf(X, z, rbf=rbf.rbf_dct['multi'])
    assert (r1(X) == r2(X)).all()


def test_opt_api():
    X = rand(10,3)
    z = rand(10)
    cv_kwds = dict(ns=5, nr=1)
    rbf_kwds = dict(rbf='inv_multi')
    rbf.fit_opt(X, z, method='fmin', opt_kwds=dict(disp=True, x0=5, maxiter=3), what='p',
                rbf_kwds=rbf_kwds)
    rbf.fit_opt(X, z, method='fmin', opt_kwds=dict(disp=True, x0=[5, 1e-8], maxiter=3),
                what='pr', cv_kwds=cv_kwds, rbf_kwds=rbf_kwds)
    rbf.fit_opt(X, z, method='de', opt_kwds=dict(bounds=[(1,3), (1e-6,1)],
                maxiter=3))
