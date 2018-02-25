"""
Radial Basis Function N-dim fitting. See :ref:`rbf` for details.
"""

import warnings
import math
from scipy import optimize
import numpy as np
import scipy.linalg as linalg
from scipy.spatial import distance
from pwtools import num


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
            Note that with scipy.spatial, we get R instead of R**2 anyway.
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
    def __init__(self, points, values=None, centers=None, rbf=RBFMultiquadric(),
                 verbose=False):
        """
        Parameters
        ----------
        points : 2d array, (M,N)
            data points : M points in N-dim space, training set points
        values : 1d array, (M,)
            function values at training points
        centers : 2d array (K,N)
            K N-dim center vectors, for usual interpolation training points == centers
            (default)
        rbf : RBFFunction instance, optional
        verbose : bool, optional
            print some messages

        Examples
        --------
        >>> from pwtools import mpl, rbf
        >>> import numpy as np
        >>> # 1D example w/ derivatives. For 1d, we need to use points[:,None]
        >>> # input array containing training (dd.DY) and interpolation
        >>> # (ddi.XY) points must be 2d.
        >>> fig,ax = mpl.fig_ax()
        >>> x=linspace(0,10,20)     # shape (M,), M=20 points
        >>> z=sin(x)                # shape (M,)
        >>> rbfi=rbf.RBFInt(x[:,None],z)
        >>> rbfi.fit()
        >>> xi=linspace(0,10,100)   # shape (M,), M=100 points
        >>> ax.plot(x,z,'o', label='data')
        >>> ax.plot(xi, sin(xi), label='sin(x)')
        >>> ax.plot(xi, rbfi(xi[:,None]), label='rbf')
        >>> ax.plot(xi, cos(xi), label='cos(x)')
        >>> ax.plot(xi, rbfi(xi[:,None],der=1)[:,0], label='d(rbf)/dx')
        >>> ax.legend()
        >>> # 2D example
        >>> x = np.linspace(-3,3,10)
        >>> dd = mpl.Data2D(x=x, y=x)
        >>> dd.update(Z=np.sin(dd.X)+np.cos(dd.Y))
        >>> rbfi=rbf.RBFInt(dd.XY, dd.zz)
        >>> rbfi.fit()
        >>> xi=linspace(-3,3,50)
        >>> ddi = mpl.Data2D(x=xi, y=xi)
        >>> fig1,ax1 = mpl.fig_ax3d()
        >>> ax1.scatter(dd.xx, dd.yy, dd.zz, label='data', color='r')
        >>> ax1.plot_wireframe(ddi.X, ddi.Y, rbfi(ddi.XY).reshape(50,50), 
        ...                    label='rbf')
        >>> ax1.set_xlabel('x'); ax1.set_ylabel('y');
        >>> ax1.legend()
        >>> fig2,ax2 = mpl.fig_ax3d()
        >>> offset=2
        >>> ax2.plot_wireframe(ddi.X, ddi.Y, rbfi(ddi.XY).reshape(50,50), 
        ...                    label='rbf', color='b')
        >>> ax2.plot_wireframe(ddi.X, ddi.Y, 
        ...                    rbfi(ddi.XY, der=1)[:,0].reshape(50,50)+offset,
        ...                    color='g', label='d(rbf)/dx')
        >>> ax2.plot_wireframe(ddi.X, ddi.Y, 
        ...                    rbfi(ddi.XY, der=1)[:,1].reshape(50,50)+2*offset,
        ...                    color='r', label='d(rbf)/dy')
        >>> ax2.set_xlabel('x'); ax2.set_ylabel('y');
        >>> ax2.legend()
        """
        self.points = points
        self.values = values
        self.centers = self.points if centers is None else centers
        self.rbf = rbf
        self.verbose = verbose
        self._assert_ndim_points(self.points)
        self._assert_ndim_points(self.centers)
        self._assert_ndim_values(self.values)

        self.distsq = None

    @staticmethod
    def _assert_ndim_points(points):
        assert points.ndim == 2, ("points or centers not 2d array")

    @staticmethod
    def _assert_ndim_values(values):
        assert values.ndim == 1, ("values not 1d array")

    def calc_distsq(self):
        """Calculate self.distsq for the training set.

        Can be used in conjunction w/ set_values() to construct a network which
        calculates distsq only once. Then it can be trained for different
        values many times, using the same points grid w/o re-calculating distsq
        over and over again.

        Examples
        --------
        >>> rbfi = rbf.RBFInt(points=points, values=None)
        >>> rbfi.calc_distsq()
        >>> for values in ...:
        >>> ... rbfi.set_values(values)
        >>> ... rbfi.fit()
        >>> ... ZI=rbfi(XI)
        """
        self.distsq = self.get_distsq()

    def set_values(self, values):
        self._assert_ndim_values(values)
        self.values = values

    def get_distsq(self, points=None):
        """Matrix of distance values r_ij = ||x_i - c_j|| with::

            x_i : points[i,:]
            c_i : centers[i,:]

        Parameters
        ----------
        points : array (M,N) with N-dim points, optional
            If None then ``self.points`` is used.
        
        Returns
        -------
        distsq : (M,K), where K = M usually for training
        """
        # pure numpy:
        #     dist = points[:,None,...] - centers[None,...]
        #     distsq = (dist**2.0).sum(axis=-1)
        # where
        #     points:  (M,N)
        #     centers: (K,N)
        #     dist:    (M,K,N) "matrix" of distance vectors (only for numpy case)
        #     distsq:  (M,K)    matrix of squared distance values
        # Creates *big* temporary arrays if points is big (~1e4 points).
        #
        # training:
        #     If points == centers, we could also use pdist(points), which
        #     would give us a 1d array of all distances. But we need the
        #     redundant square matrix form for G=rbf(distsq) anyway, so there
        #     is no real point in special-casing that. These two are the same:
        #      >>> R = spatial.squareform(spatial.distances.pdist(points))
        #      >>> R = spatial.distances.cdist(points,points)
        #      >>> distsq = R**2
        if points is None:
            if self.distsq is None:
                return num.distsq(self.points, self.centers)
            else:
                return self.distsq
        else:
            return num.distsq(points, self.centers)

    def get_param(self, param):
        """Return `param` for RBF (to set self.rbf.param).

        Parameters
        ----------
        param : string 'est' or float
            If 'est', then return the mean distance of all points.
        """
        if param == 'est':
            return np.sqrt(self.get_distsq()).mean()
        else:
            return param

    def train(self, mode=None, **kwds):
        """Use :meth:`fit` instead."""
        if mode is not None:
            warnings.warn("train(mode=..) deprecated, use fit()",
                          DeprecationWarning)
        return self.fit(**kwds)

    
    def fit(self, param='est', solver='lstsq'):
        """Solve linear system for the weights w:
            G . w = z
        
        with centers == points (center vectors are all data points). Then G is
        quadratic. Updates ``self.weights``.
        
        Parameters
        ----------
        param : 'est' or float
            see :meth:`get_param`
        solver : str
            'solve' : interpolation
                Use ``scipy.linalg.solve()``. By definition, this always yields
                perfect interpolation at the data points. May be numerically
                unstable.
            'lstsq' : least squares regression (default)
                Use ``scipy.linalg.lstsq()``. Numerically more stable. Will
                mostly be the same as the interpolation result, but will not go
                thru all points for very noisy data.
        """
        # this test may be expensive for big data sets
        assert (self.centers == self.points).all(), "centers == points not fulfilled"
        # re-use self.distsq if possible
        if self.distsq is None:
            self.distsq = self.get_distsq()
        self.rbf.param = self.get_param(param)
        G = self.rbf(self.distsq)
        if solver == 'solve':
            weights = getattr(linalg, solver)(G, self.values)
        elif solver == 'lstsq':
            weights, res, rnk, svs = getattr(linalg, solver)(G, self.values)
        else:
            raise Exception("unknown solver: {}".format(solver))
        self.weights = weights

    def fit_opt_param(self):
        """Optimize ``rbf.param`` and weights simultaneously by minimizing the
        fit error with :meth:`fit` using ``solver='lstsq'``. Can be used in
        place of :meth:`fit`.
        """
        def func(pvec):
            param = pvec[0]
            self.fit(param=param, solver='lstsq')
            res = self.values - self.interpolate(self.points)
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

    def interpolate(self, points):
        """Actually do interpolation. Return interpolated values values at each
        point in points.

        Parameters
        ----------
        points : see :meth:`__call__`

        Returns
        -------
        vals : 1d array (L,)

        Notes
        -----
        Calculates
            distsq = (points - centers)**2  # squared distance matrix
            G = rbf(distsq)                 # RBF values
            zi = dot(G, w)                  # interpolation z_i = Sum_j g_ij w_j
        """
        self._assert_ndim_points(points)
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq)
        assert G.shape[1] == len(self.weights), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape),
                                                 str(self.weights.shape))
        # normalize weights
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        return np.dot(G, ww / maxw) * maxw

    def deriv(self, points):
        """Matrix of partial first derivatives.

        Implemented for
            RBFMultiquadric
            RBFGauss

        Parameters
        ----------
        points : see :meth:`__call__`

        Returns
        -------
        2d array (L,N)
            Each row holds the partial derivatives of one input point ``points[i,:] =
            [x_0, ..., x_N-1]``. For all points points: (L,N), we get the matrix::

                [[dy/dx0_0,   dy/dx0_1,   ..., dy/dx0_N-1],
                 [...],
                 [dy/dxL-1_0, dy/dxL-1_1, ..., dy/dxL-1_N-1]]
        """
        # For the implemented RBF types, the derivatives w.r.t. to the point
        # coords simplify to nice dot products, which can be evaluated
        # reasonably fast w/ numpy. We don't need to change the RBF's
        # implementations to provide a deriv() method. For that, they would
        # need to take points and centers explicitly as args instead of squared
        # distances, which are calculated fast by Fortran outside.
        #
        # Speed:
        # We have one python loop over the L points (points.shape=(L,N)) left, so
        # this gets slow for many points.
        #
        # Loop versions (for RBFMultiquadric):
        #
        # # 3 loops:
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         for ii in range(len(self.weights)):
        #             D[zz,kk] += (points[zz,kk] - centers[ii,kk]) / G[zz,ii] * \
        #                 self.weights[ii]
        #
        # # 2 loops:
        # D = np.zeros((L,N), dtype=float)
        # for zz in range(L):
        #     for kk in range(N):
        #         vec = -1.0 * (centers[:,kk] - points[zz,kk]) / G[zz,:]
        #         D[zz,kk] = np.dot(vec, self.weights)
        self._assert_ndim_points(points)
        L,N = points.shape
        centers = self.centers
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq)
        D = np.empty((L,N), dtype=float)
        ww = self.weights
        maxw = np.abs(ww).max()*1.0
        if isinstance(self.rbf, RBFMultiquadric):
            for zz in range(L):
                D[zz,:] = -np.dot(((centers - points[zz,:]) / G[zz,:][:,None]).T, ww / maxw) * maxw
        elif isinstance(self.rbf, RBFGauss):
            for zz in range(L):
                D[zz,:] = 1.0 / self.rbf.param**2.0 * \
                    np.dot(((centers - points[zz,:]) * G[zz,:][:,None]).T, ww / maxw) * maxw
        else:
            raise Exception("derivative for rbf type not implemented")
        return D

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`interpolate` or :meth:`deriv`.

        Parameters
        ----------
        points : 2d array (L,N) or 1d array (N,) or None
            L data points in N-dim space. If None, then points = self.points, i.e. just
            evaluate the training set. If 1d, then this is just 1 point, which
            will be converted to shape (1,N) internally.
        der : int
            If == 1 return matrix of partial derivatives (see self.deriv()), else
            interpolated values values (default).

        Returns
        -------
        vals : 1d array (L,)
            Interpolated values.
        or
        derivs : 2d array (L,N)
            1st partial derivatives.
        """
        if 'der' in list(kwargs.keys()):
            if kwargs['der'] != 1:
                raise Exception("only der=1 supported")
            kwargs.pop('der')
            return self.deriv(*args, **kwargs)
        else:
            return self.interpolate(*args, **kwargs)
