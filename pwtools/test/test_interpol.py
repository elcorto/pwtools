import tempfile
import pickle

import numpy as np

from pwtools import mpl, num
from pwtools.test import tools
from .testenv import testdir


def return_min(inter):
    # Return scalar minimum instead of array (5.0 instead of [5.0]).
    return inter(inter.get_min(maxfun=1e6, maxiter=1e2))[0]


def dump(obj, testdir=testdir):
    fn = tempfile.mktemp(dir=testdir, prefix="pwtools_test_interpol_")
    fd = open(fn, "wb")
    pickle.dump(obj, fd, 2)
    fd.close()


def get_data():
    x = np.linspace(-5, 5, 20)
    y = x
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T
    Z = (X + 3) ** 2 + (Y + 4) ** 2 + 5
    dd = mpl.Data2D(X=X, Y=Y, Z=Z)
    tgt = np.array([5.0, 30.0])
    return dd, tgt


class TestInterpol2D:
    # Oh come on!
    #   PytestCollectionWarning: cannot collect test class 'TestInterpol2D' because
    #   it has a __init__ constructor
    # What!? This is Python, right? I mean, this here is valid code. Used to work
    # with nose, thank you very much.
    ##    def __init__(self):
    ##        self.dd, self.tgt = get_data()
    dd, tgt = get_data()
    rbf_tol_kwds = dict(rtol=1e-4, atol=1e-5)
    rbf_param_r = 1e-10

    def test_rbf_multi(self):
        inter = num.Interpol2D(
            dd=self.dd, what="rbf_multi", r=self.rbf_param_r
        )
        np.testing.assert_allclose(
            inter([[-3, -4], [0, 0]]), self.tgt, **self.rbf_tol_kwds
        )
        np.testing.assert_allclose(return_min(inter), 5.0, **self.rbf_tol_kwds)
        np.testing.assert_allclose(
            inter(self.dd.XY), self.dd.zz, **self.rbf_tol_kwds
        )
        dump(inter)

    def test_rbf_inv_multi(self):
        inter = num.Interpol2D(
            dd=self.dd, what="rbf_inv_multi", r=self.rbf_param_r
        )
        np.testing.assert_allclose(
            inter([[-3, -4], [0, 0]]), self.tgt, **self.rbf_tol_kwds
        )
        np.testing.assert_allclose(return_min(inter), 5.0, **self.rbf_tol_kwds)
        np.testing.assert_allclose(
            inter(self.dd.XY), self.dd.zz, **self.rbf_tol_kwds
        )
        dump(inter)

    def test_rbf_gauss(self):
        inter = num.Interpol2D(
            dd=self.dd, what="rbf_gauss", r=self.rbf_param_r
        )
        np.testing.assert_allclose(
            inter([[-3, -4], [0, 0]]), self.tgt, **self.rbf_tol_kwds
        )
        np.testing.assert_allclose(return_min(inter), 5.0, **self.rbf_tol_kwds)
        np.testing.assert_allclose(
            inter(self.dd.XY), self.dd.zz, **self.rbf_tol_kwds
        )
        dump(inter)

    def test_poly(self):
        inter = num.Interpol2D(dd=self.dd, what="poly", deg=5)
        np.testing.assert_allclose(inter([[-3, -4], [0, 0]]), self.tgt)
        np.testing.assert_allclose(return_min(inter), 5.0, atol=1e-5)
        np.testing.assert_allclose(inter(self.dd.XY), self.dd.zz)
        dump(inter)

    def test_bispl(self):
        # pickling this fails:
        #
        #  ERROR: pwtools.test.test_interpol.test_interpol2d
        #  PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed
        #  FAILED (errors=1)
        #
        # The reason is that a function _call() inside __init__ is defined and
        # pickle cannot serialize nested functions. But also defining a class-level
        # function self._call_bispl instead doesn't work. So there is no solution
        # here. You need to store the to-be-interpolated data and build the
        # interpolator again as needed. Not a speed problem, just inconvenient. All
        # other interpolator types can be pickled.
        inter = num.Interpol2D(dd=self.dd, what="bispl")
        np.testing.assert_allclose(inter([[-3, -4], [0, 0]]), self.tgt)
        np.testing.assert_allclose(return_min(inter), 5.0)
        np.testing.assert_allclose(inter(self.dd.XY), self.dd.zz)
        ##dump(inter)

    def test_linear(self):
        # linear, ct and nearest are very inaccurate, use only for plotting!
        inter = num.Interpol2D(dd=self.dd, what="linear")
        np.testing.assert_allclose(
            inter([[-3, -4], [0, 0]]), self.tgt, atol=5e-1
        )
        np.testing.assert_allclose(return_min(inter), 5.0, atol=1e-1)
        np.testing.assert_allclose(inter(self.dd.XY), self.dd.zz)
        dump(inter)

    def test_nearest(self):
        # don't even test accuracy here
        inter = num.Interpol2D(dd=self.dd, what="nearest")
        dump(inter)

    def test_ct(self):
        try:
            from scipy.interpolate import CloughTocher2DInterpolator

            inter = num.Interpol2D(dd=self.dd, what="ct")
            np.testing.assert_allclose(
                inter([[-3, -4], [0, 0]]), self.tgt, atol=1e-1
            )
            np.testing.assert_allclose(return_min(inter), 5.0, atol=1e-1)
            np.testing.assert_allclose(inter(self.dd.XY), self.dd.zz)
            dump(inter)
        except ImportError:
            tools.skip(
                "couldn't import scipy.interpolate.CloughTocher2DInterpolator"
            )

    def test_api_bispl(self):
        # API
        inter = num.Interpol2D(
            xx=self.dd.xx, yy=self.dd.yy, values=self.dd.zz, what="bispl"
        )
        np.testing.assert_allclose(inter([[-3, -4], [0, 0]]), self.tgt)
        np.testing.assert_allclose(return_min(inter), 5.0)

        inter = num.Interpol2D(
            points=self.dd.XY, values=self.dd.zz, what="bispl"
        )
        np.testing.assert_allclose(inter([[-3, -4], [0, 0]]), self.tgt)
        np.testing.assert_allclose(return_min(inter), 5.0)

        inter = num.Interpol2D(self.dd.XY, self.dd.zz, what="bispl")
        np.testing.assert_allclose(inter([[-3, -4], [0, 0]]), self.tgt)
        np.testing.assert_allclose(return_min(inter), 5.0)
