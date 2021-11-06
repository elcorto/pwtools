import numpy as np
import pytest

from pwtools import rbf
rand = np.random.rand


@pytest.mark.parametrize("p", ['mean', 'scipy'])
@pytest.mark.parametrize("r", [None, 0, 1e-14])
def test_interpol_high_dim(p, r):
    # 10 points in 40-dim space
    #
    # Test only API. Interpolating random points makes no sense, of course.
    # However, on the center points (= data points = training points),
    # interpolation must be "perfect", even for low regularization parameter r.
    X = rand(10,40)
    z = rand(10)

    rbfi = rbf.Rbf(X,z, p=p, r=r)
    assert np.abs(z - rbfi(X)).max() < 1e-13

@pytest.mark.parametrize("r", [None, 1e-12])
def test_2d(r):
    # 2d example, y = f(x1,x2) = sin(x1) + cos(x2)
    x1 = np.linspace(-1,1,20)
    X1,X2 = np.meshgrid(x1, x1, indexing='ij')
    z = (np.sin(X1)+np.cos(X2)).flatten()
    X = np.array([X1.flatten(), X2.flatten()]).T
    rbfi = rbf.Rbf(X, z, p=1.5, r=r)
    # test fit at 300 random points within [-1,1]^2
    Xr = -1.0 + 2*np.random.rand(300,2)
    zr = np.sin(Xr[:,0]) + np.cos(Xr[:,1])
    err = np.abs(rbfi(Xr) - zr).max()
    assert err < 1e-6
    # Error at center points: big errors occur only at the domain boundary: -1,
    # 1, errs at the points should be smaller
    err = np.abs(z - rbfi(X)).max()
    assert err < 1e-7
    # derivative at random points
    dzr_dx = np.cos(Xr[:,0])
    dzr_dy = -np.sin(Xr[:,1])
    drbfi = rbfi(Xr, der=1)
    assert np.allclose(dzr_dx, drbfi[:,0], rtol=0, atol=1e-4)
    assert np.allclose(dzr_dy, drbfi[:,1], rtol=0, atol=1e-4)


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


def test_predict_api():
    X = rand(100,3)
    x = X[0,:]
    z = rand(100)
    f = rbf.Rbf(X, z)
    assert f(X).shape == (100,)
    assert f(x).shape == ()
    assert f(x[None,:]).shape == (1,)


def test_grad_api():
    X = rand(100,3)
    x = X[0,:]
    z = rand(100)
    f = rbf.Rbf(X, z)
    f(X, der=1).shape == X.shape
    f.deriv(X).shape == X.shape
    f(x, der=1).shape == x.shape
    f.deriv(x[None,:]).shape == (1,3)


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
    rnd = np.random.RandomState(seed=1234)
    cv_kwds = dict(n_splits=5, n_repeats=1, random_state=rnd)
    rbf_kwds = dict(rbf='inv_multi')
    rbf.fit_opt(X, z,
                method='fmin',
                opt_kwds=dict(disp=True,
                              x0=5,
                              maxiter=3),
                what='p',
                rbf_kwds=rbf_kwds,
                )
    rbf.fit_opt(X, z,
                method='fmin',
                what='pr',
                cv_kwds=cv_kwds,
                rbf_kwds=rbf_kwds,
                opt_kwds=dict(disp=True, x0=[5, 1e-8], maxiter=3),
                )
    rbf.fit_opt(X, z,
                method='de',
                opt_kwds=dict(bounds=[(1,3), (1e-6,1)],
                              maxiter=3),
                )
    rbf.fit_opt(X, z,
                method='brute',
                opt_kwds=dict(ranges=[(1,3), (1e-6,1)],
                              Ns=5),
                )
    rbf.fit_opt(X, z,
                method='brute',
                cv_kwds=cv_kwds,
                rbf_kwds=rbf_kwds,
                opt_kwds=dict(ranges=[(1,3), (1e-6,1)],
                              Ns=5),
                )
