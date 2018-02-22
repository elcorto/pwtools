"""
Radial Basis Functions Neural Network for interpolation or fitting of n-dim
data points.

Refs:

.. [1] http://en.wikipedia.org/wiki/Radial_basis_function_network
.. [2] Numerical Recipes, 3rd ed., ch 3.7

Training
--------
For our RBF network, we use the traditional approach and simply solve a
linear system for the weights. Calculating the distance matrix w/
scipy.spatial.distance.cdist() or num.distsq() is handled efficiently,
even for many points. But we get into trouble for many points (order 1e4) b/c
solving a big dense linear system (1e4 x 1e4) with plain numpy.linalg on a
single core is possible but painful (takes some minutes) -- the traditional
RBF problem. Maybe use numpy build against threaded MKL, or even scalapack?
For the latter, look at how GPAW does this.

rbf parameter
-------------
Each RBF has a single parameter (`param`), which can be tuned. This is
usually a measure for the "width" of the function.

* What seems to work best is RBFMultiquadric + param='est' (mean distance of
  all points), i.e. the "traditional" RBF approach. This is exactly what's done
  in scipy.interpolate.Rbf .

* It seems that for very few data points (say 5x5 for x**2 + x**2), the
  default param='est' is too small, which results in wildly fluctuating
  interpolation, b/c there is no data support in between the data points. We
  need to have wider RBFs. Usually, bigger (x10), sometimes much bigger
  (x100) params work.

* However, we also had a case where we found that the square of a
  small param (param (from 'est') = 0.3, param**2 = 0.1) was actually much
  better than the one from param='est'.

Interpolation vs. fitting
-------------------------
For smooth noise-free data, RBF works perfect. But for noisy data, we would
like to do some kind of fit instead, like the "s" parameter to
scipy.interpolate.bisplrep(). scipy.interpolate.Rbf has a "smooth" parameter
and what they do is some form of regularization (solve (G-I*smooth) . w = y
instead of G . w = y; G = RBF matrix, w = weights to solve for, y = data).

We found (see examples/rbf.py) that scipy.linalg.solve() often gives an
ill-conditioned matrix warning, which shows numerical instability and results
in bad interpolation. It seems that problems start to show as soon as the noise
level (y + noise) is in the same order or magnitude as the mean point distance.
Then we see wildly fluctuating data points which are hard to interpolate. In
that case, the mean-distance estimate for the rbf param breaks down and one
needs to use smaller values to interpolate all fluctuations. However, in most
cases, on does actually want to perform a fit instead in such situations.

If we switch from scipy.linalg.solve() to scipy.linalg.lstsq() and solve the
system in a least squares sense, we get much more stable solutions. With
lstsq(), we have the smoothness by construction, b/c we do *not* perform
interpolation anymore -- this is a fit now. The advantage of using least
squares is that we don't have a smoothness parameter which needs to be tuned.

If the noise is low relative to the point distance, we get interpolation-like
results, which cannot be distinguished from the solutions obtained with a
normal linear system solver. The method will try its best to do interpolation,
but will smoothly transition to fitting as noise increases, which is what we
want. Hence, lstsq is the default solver.

other codes
-----------

* scipy.interpolate.Rbf
  Essentially the same as we do. We took some ideas from there.
* http://pypi.python.org/pypi/PyRadbas/0.1.0
  Seems to work with Gaussians hardcoded. Not what we want.
* http://code.google.com/p/pyrbf/
  Seems pretty useful for large problems (many points), but not
  documented very much.

input data
----------
It usually helps if all data ranges (points X and values Y) are in the same
order of magnitude, e.g. all have a maximum close to unity or so. If, for
example, X and Y have very different scales (say X -0.1 .. 0.1 and Y
0...1e4), you may get bad interpolation between points. This is b/c the RBFs
also live on more or less equal x-y scales.
"""

import warnings
import math
from scipy import optimize
import numpy as np
import scipy.linalg as linalg
from scipy.spatial import distance
from pwtools import num

warnings.simplefilter('always')


class RBFFunction:
    """Radial basis function base class."""
    def __init__(self, param=None):
        """
        Parameters
        ----------
        param : scalar or None
            The RBF's free parameter.
        """
        self.param = param

    def _get_param(self, param):
        p = self.param if param is None else param
        assert p is not None, ("param is None")
        return p

    def __call__(self, rsq, param=None):
        """
        Parameters
        ----------
        rsq : scalar or nd array
            Squared distances. Squared b/c this avoids using sqrt() in the
            distance matrix (R) calculation just to square them here again.
            bad:  R   = sqrt((X - C)**2)
            good: Rsq = (X - C)**2
            Note that with scipy.spatial, we get R instead of Rsq anyway.
        param : scalar or None, optional
            If None, then self.param is used.
        """
        pass


class RBFGauss(RBFFunction):
    r"""Gaussian RBF :math:`\exp\left(-\frac{r^2}{2\,p^2}\right)`
    """
    def __call__(self, rsq, param=None):
        return np.exp(-0.5*rsq / self._get_param(param)**2.0)


class RBFMultiquadric(RBFFunction):
    r"""Multiquadric RBF :math:`\sqrt{r^2 + p^2}`"""
    def __call__(self, rsq, param=None):
        return np.sqrt(rsq + self._get_param(param)**2.0)


class RBFInverseMultiquadric(RBFFunction):
    r"""Inverse Multiquadric RBF :math:`\frac{1}{\sqrt{r^2 + p^2}}`"""
    def __call__(self, rsq, param=None):
        return (rsq + self._get_param(param)**2.0)**(-0.5)


class RBFInt:
    """Radial basis function neural network interpolator."""
    def __init__(self, X, Y=None, C=None, rbf=RBFMultiquadric(),
                 verbose=False, distmethod='fortran'):
        """
        Parameters
        ----------
        X : 2d array, (M,N)
            data points : M points in N-dim space, training set points
        Y: 1d array, (M,)
            function values at training points
        C : 2d array (K,N)
            K N-dim center vectors, for usual interpolation training X == C
            (default)
        rbf : RBFFunction instance, optional
        verbose : bool, optional
            print some messages
        distmethod : str, optional
            Choose method for distance matrix calculation.
                fortran : Fortran implementation (OpenMP, fastest)
                spatial : scipy.spatial.distance.cdist
                numpy : pure numpy (memory hungry, slowest)

        Examples
        --------
        >>> from pwtools import mpl, rbf
        >>> import numpy as np
        >>> # 1D example w/ derivatives. For 1d, we need to use X[:,None] b/c the
        >>> # input array containing training (X) and interpolation (XI) points must
        >>> # be 2d.
        >>> fig,ax = mpl.fig_ax()
        >>> X=linspace(0,10,20)     # shape (M,), M=20 points
        >>> Y=sin(X)                # shape (M,)
        >>> rbfi=rbf.RBFInt(X[:,None],Y)
        >>> rbfi.fit()
        >>> XI=linspace(0,10,100)   # shape (M,), M=100 points
        >>> ax.plot(X,Y,'o', label='data')
        >>> ax.plot(XI, sin(XI), label='sin(x)')
        >>> ax.plot(XI, rbfi(XI[:,None]), label='rbf')
        >>> ax.plot(XI, cos(XI), label='cos(x)')
        >>> ax.plot(XI, rbfi(XI[:,None],der=1)[:,0], label='d(rbf)/dx')
        >>> ax.legend()
        >>> # 2D example
        >>> x1=np.linspace(-3,3,10); x2=x1
        >>> X1,X2=np.meshgrid(x1,x2); X1,X2 = X1.T,X2.T
        >>> X = np.array([X1.flatten(), X2.flatten()]).T
        >>> Y = (np.sin(X1)+np.cos(X2)).flatten()
        >>> x1i=linspace(-3,3,50); x2i=x1i
        >>> X1I,X2I=np.meshgrid(x1i,x2i); X1I,X2I = X1I.T,X2I.T
        >>> XI = np.array([X1I.flatten(), X2I.flatten()]).T
        >>> rbfi=rbf.RBFInt(X,Y)
        >>> rbfi.fit()
        >>> fig1,ax1 = mpl.fig_ax3d()
        >>> ax1.scatter(X[:,0], X[:,1], Y, label='data', color='r')
        >>> ax1.plot_wireframe(X1I, X2I, rbfi(XI).reshape(50,50), label='rbf')
        >>> ax1.set_xlabel('x'); ax1.set_ylabel('y');
        >>> ax1.legend()
        >>> fig2,ax2 = mpl.fig_ax3d()
        >>> offset=2
        >>> ax2.plot_wireframe(X1I, X2I, rbfi(XI).reshape(50,50), label='rbf',
        ...                    color='b')
        >>> ax2.plot_wireframe(X1I, X2I, rbfi(XI,der=1)[:,0].reshape(50,50)+offset,
        ...                    color='g', label='d(rbf)/dx')
        >>> ax2.plot_wireframe(X1I, X2I, rbfi(XI,der=1)[:,1].reshape(50,50)+2*offset,
        ...                    color='r', label='d(rbf)/dy')
        >>> ax2.set_xlabel('x'); ax2.set_ylabel('y');
        >>> ax2.legend()
        """
        self.X = X
        self.Y = Y
        self.C = self.X if C is None else C
        self.rbf = rbf
        self.verbose = verbose
        self.distmethod = distmethod
        self._assert_ndim_X(self.X)
        self._assert_ndim_X(self.C)
        self._assert_ndim_Y(self.Y)

        self.Rsq = None

    @staticmethod
    def _assert_ndim_X(X):
        assert X.ndim == 2, ("X or C not 2d array")

    @staticmethod
    def _assert_ndim_Y(Y):
        assert Y.ndim == 1, ("Y not 1d array")

    def calc_dist_mat(self):
        """Calculate self.Rsq for the training set.

        Can be used in conjunction w/ set_Y() to construct a network which
        calculates Rsq only once. Then it can be trained for different Y many
        times, using the same X grid w/o re-calculating Rsq over and over
        again.

        Examples
        --------
        >>> rbfi = rbf.RBFInt(X=X, Y=None)
        >>> rbfi.calc_dist_mat()
        >>> for Y in ...:
        >>> ... rbfi.set_Y(Y)
        >>> ... rbfi.fit()
        >>> ... YI=rbfi(XI)
        """
        self.Rsq = self.get_dist_mat(X=self.X, C=self.C)

    def set_Y(self, Y):
        self._assert_ndim_Y(Y)
        self.Y = Y

    def get_dist_mat(self, X=None, C=None):
        """Matrix of distance values r_ij = ||x_i - c_j|| with::

            x_i : X[i,:]
            c_i : C[i,:]

        Parameters
        ----------
        X : None or array (M,N) with N-dim points
            Training data or interpolation points. If None then self.Rsq is
            returned.
        C : 2d array (K,N), optional
            If None then self.C is used.

        Returns
        -------
        Rsq : (M,K), where K = M usually for training
        """
        # pure numpy:
        #     dist = X[:,None,...] - C[None,...]
        #     Rsq = (dist**2.0).sum(axis=-1)
        # where
        #     X:    (M,N)
        #     C:    (K,N)
        #     dist: (M,K,N) "matrix" of distance vectors (only for numpy case)
        #     Rsq:  (M,K)    matrix of squared distance values
        # Creates *big* temporary arrays if X is big (~1e4 points).
        #
        # training:
        #     If X == C, we could also use pdist(X), which would give us a 1d
        #     array of all distances. But we need the redundant square matrix
        #     form for get_rbf_mat() anyway, so there is no real point in
        #     special-casing that. These two are the same:
        #      >>> R = spatial.squareform(spatial.distances.pdist(X))
        #      >>> R = spatial.distances.cdist(X,X)
        #      >>> Rsq = R**2
        if X is not None:
            C = self.C if C is None else C
            if self.distmethod == 'spatial':
                Rsq = distance.cdist(X, C)**2.0
            elif self.distmethod == 'fortran':
                Rsq = num.distsq(X, C)
            elif self.distmethod == 'numpy':
                dist = X[:, None, ...] - C[None, ...]
                Rsq = (dist**2.0).sum(axis=-1)
            else:
                raise Exception("unknown value for distmethod: %s"
                                % self.distmethod)
            return Rsq
        else:
            assert self.Rsq is not None, ("self.Rsq is None")
            return self.Rsq

    def get_rbf_mat(self, Rsq=None):
        """Matrix of RBF values g_ij = rbf(||x_i - c_j||)."""
        Rsq = self.Rsq if Rsq is None else Rsq
        G = self.rbf(Rsq)
        return G

    def get_param(self, param):
        """Return `param` for RBF (to set self.rbf.param).

        Parameters
        ----------
        param : string 'est' or float
            If 'est', then return the mean distance of all points.
        """
        if param == 'est':
            return np.sqrt(self.get_dist_mat()).mean()
        else:
            return param

    def train(self, mode=None, **kwds):
        """Use :meth:`fit` instead."""
        if mode is not None:
            warnings.warn("train(mode=..) deprecated, use fit()",
                          DeprecationWarning)
        return self.fit(**kwds)

    
    def fit(self, param='est', solver='lstsq'):
        """Solve for the weights w_j:
            y_i = Sum_j g_ij w_j
        
        in case C == X (center vectors are all data points). Then G is
        quadratic w/ dim (M,M). Updates ``self.weights``.
        
        Parameters
        ----------
        param : 'est' or float
            see :meth:`get_param`
        solver : str
            'solve' : interpolation
                By definition, this always yields perfect interpolation at the points
                in X, but may oscillate between points if you have only a few. Use
                bigger param in that case.
            'lstsq' : fitting
                use least squares
        """
        # this test may be expensive for big data sets
        assert (self.C == self.X).all(), "C == X not fulfilled"
        if self.Rsq is None:
            self.Rsq = self.get_dist_mat(X=self.X, C=self.C)
        self.rbf.param = self.get_param(param)
        G = self.get_rbf_mat(Rsq=self.Rsq)
        if solver == 'solve':
            weights = getattr(linalg, solver)(G, self.Y)
        elif solver == 'lstsq':
            weights, res, rnk, svs = getattr(linalg, solver)(G, self.Y)
        else:
            raise Exception("unknown solver: {}".format(solver))
        self.weights = weights

    def fit_opt_param(self):
        def func(pvec):
            param = pvec[0]
            self.fit(param=param, solver='lstsq')
            res = self.Y - self.interpolate(self.X)
            err = np.dot(res,res)
            err = math.exp(abs(err)) if param < 0 else err
            if self.verbose:
                print('err={}, param={}'.format(err, param))
            return err 
        self.fit(param='est', solver='lstsq') 
        # self.rbf.param and self.weights are constantly updated in func(), the
        # last iteration already set the converged values, no need to assign
        # the result of fmin() 
        optimize.fmin(func, [self.rbf.param], disp=self.verbose)
        ##optimize.differential_evolution(func, bounds=[(0,3*self.rbf.param)], 
        ##                                disp=self.verbose)

    def interpolate(self, X):
        """Actually do interpolation. Return interpolated Y values at each
        point in X.

        Parameters
        ----------
        X : see :meth:`__call__`

        Returns
        -------
        YI : 1d array (L,)

        Notes
        -----
        Calculates
            Rsq = (X - C)**2      # squared distance matrix
            G = rbf(Rsq)          # RBF values
            YI = dot(G, w)        # interpolation y_i = Sum_j g_ij w_j
        """
        self._assert_ndim_X(X)
        Rsq = self.get_dist_mat(X=X, C=self.C)
        G = self.get_rbf_mat(Rsq)
        assert G.shape[1] == len(self.weights), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape),
                                                 str(self.weights.shape))
        # normalize weights
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        return np.dot(G, ww / maxw) * maxw

    def deriv(self, X):
        """Matrix of partial first derivatives.

        Implemented for
            RBFMultiquadric
            RBFGauss

        Parameters
        ----------
        X : see :meth:`__call__`

        Returns
        -------
        2d array (L,N)
            Each row holds the partial derivatives of one input point ``X[i,:] =
            [x_0, ..., x_N-1]``. For all points X: (L,N), we get the matrix::

                [[dy/dx0_0,   dy/dx0_1,   ..., dy/dx0_N-1],
                 [...],
                 [dy/dxL-1_0, dy/dxL-1_1, ..., dy/dxL-1_N-1]]
        """
        # For the implemented RBF types, the derivatives w.r.t. to the point
        # coords simplify to nice dot products, which can be evaluated
        # reasonably fast w/ numpy. We don't need to change the RBF's
        # implementations to provide a deriv() method. For that, they would
        # need to take X and C explicitly as args instead of squared
        # distances, which are calculated fast by Fortran outside.
        #
        # Speed:
        # We have one python loop over the L points (X.shape=(L,N)) left, so
        # this gets slow for many points.
        #
        # Loop versions (for RBFMultiquadric):
        #
        # # 3 loops:
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         for ii in range(len(self.weights)):
        #             D[zz,kk] += (X[zz,kk] - C[ii,kk]) / G[zz,ii] * \
        #                 self.weights[ii]
        #
        # # 2 loops:
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         vec = -1.0 * (C[:,kk] - X[zz,kk]) / G[zz,:]
        #         D[zz,kk] = np.dot(vec, self.weights)
        self._assert_ndim_X(X)
        L,N = X.shape
        C = self.C
        Rsq = self.get_dist_mat(X=X, C=C)
        G = self.get_rbf_mat(Rsq)
        D = np.empty((L,N), dtype=float)
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        if isinstance(self.rbf, RBFMultiquadric):
            for zz in range(L):
                D[zz,:] = -np.dot(((C - X[zz,:]) / G[zz,:][:,None]).T, ww / maxw) * maxw
        elif isinstance(self.rbf, RBFGauss):
            for zz in range(L):
                D[zz,:] = 1.0 / self.rbf.param**2.0 * \
                    np.dot(((C - X[zz,:]) * G[zz,:][:,None]).T, ww / maxw) * maxw
        else:
            raise Exception("derivative for rbf type not implemented")
        return D

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`interpolate` or :meth:`deriv`.

        Parameters
        ----------
        X : 2d array (L,N) or 1d array (N,) or None
            L data points in N-dim space. If None, then X = self.X, i.e. just
            evaluate the training set. If 1d, then this is just 1 point, which
            will be converted to shape (1,N) internally.
        der : int
            If == 1 return matrix of partial derivatives (see self.deriv()), else
            interpolated Y values (default).

        Returns
        -------
        YI : 1d array (L,)
            Interpolated values.
        or
        D : 2d array (L,N)
            1st partial derivatives.
        """
        if 'der' in list(kwargs.keys()):
            if kwargs['der'] != 1:
                raise Exception("only der=1 supported")
            kwargs.pop('der')
            return self.deriv(*args, **kwargs)
        else:
            return self.interpolate(*args, **kwargs)
