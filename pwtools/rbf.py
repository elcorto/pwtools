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


try:
    from sklearn.model_selection import RepeatedKFold
except ImportError:
    pass


def rbf_gauss(rsq, p):
    r"""Gaussian RBF :math:`\exp\left(-\frac{r^2}{2\,p^2}\right)`
    """
    return np.exp(-0.5*rsq / p**2.0)


def rbf_multi(rsq, p):
    r"""Multiquadric RBF :math:`\sqrt{r^2 + p^2}`"""
    return np.sqrt(rsq + p**2.0)


def rbf_inv_multi(rsq, p):
    r"""Inverse Multiquadric RBF :math:`\frac{1}{\sqrt{r^2 + p^2}}`"""
    return 1/rbf_multi(rsq, p)


rbf_dct = {
    'gauss': rbf_gauss,
    'multi': rbf_multi,
    'inv_multi': rbf_inv_multi,
    }


class Rbf:
    """Radial basis function network interpolation and fitting."""
    def __init__(self, points, values, centers=None, rbf='inv_multi',
                 r=None, p='mean', verbose=False, fit=True):
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
        rbf : str (see rbf_dct.keys()) or callable rbf(r**2, p)
        r : float or None
            regularization parameter, if None then we use a least squares
            solver
        p : float
            the RBF's free parameter
        verbose : bool
        fit : bool
            call self.fit() in __init__

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
        >>> rbfi=rbf.Rbf(x[:,None],z)
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
        >>> rbfi=rbf.Rbf(dd.XY, dd.zz)
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
        self.rbf = rbf_dct[rbf] if isinstance(rbf, str) else rbf
        self.verbose = verbose
        self._assert_ndim_points(self.points)
        self._assert_ndim_points(self.centers)
        self._assert_ndim_values(self.values)
        self.distsq = None
        if p == 'mean':
            self.distsq = self.get_distsq()
            self.p = np.sqrt(self.distsq).mean()
        else:
            self.p = p
        self.r = r
        if fit:
            self.fit()

    @staticmethod
    def _assert_ndim_points(points):
        assert points.ndim == 2, ("points or centers not 2d array")

    @staticmethod
    def _assert_ndim_values(values):
        assert values.ndim == 1, ("values not 1d array")

    def get_distsq(self, points=None):
        """Matrix of distance values r_ij = ||x_i - c_j|| with::

            x_i : points[i,:]
            c_i : centers[i,:]

        Parameters
        ----------
        points : array (M,N) with N-dim points, optional
            If None then ``self.points`` is used (training points).
        
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
    
    def get_params(self):
        return self.p, self.r

    def fit(self):
        """Solve linear system for the weights w:
            G . w = z
        
        with centers == points (center vectors are all data points). Then G is
        quadratic. Updates ``self.w`` and ``self.p``.
        
        Parameters
        ----------
        p : 'mean' or float
            see :meth:`get_p`
        
        Notes
        -----
        solver : 
             r != None: linear system solver
                Use ``scipy.linalg.solve()``. By definition, this always yields
                perfect interpolation at the data points for ``r=None``. May
                be numerically unstable in that case. Use `r` to increase
                stability and create smooth fitting (generate more stiff
                functions), similar to ``lstsq`` but appears to be numerically
                more stable (no small noise in solution) .. but it is another
                parameter that needs to be tuned.
            r=None: least squares solver (default)
                Use ``scipy.linalg.lstsq()``. Numerically more stable than
                direct solver w/o regularization. Will mostly be the same as
                the interpolation result, but will not go thru all points for
                very noisy data. May create small noise in solution (plot fit
                with high point density).
        r : None, float, optional
            regularization parameter for solver='solve' 
        """
        # this test may be expensive for big data sets
        assert (self.centers == self.points).all(), "centers == points not fulfilled"
        # re-use self.distsq if possible
        if self.distsq is None:
            self.distsq = self.get_distsq()
        G = self.rbf(self.distsq, self.p)
        if self.r is None:
            self.w, res, rnk, svs = linalg.lstsq(G, self.values)
        else:
            self.w = linalg.solve(G + np.identity(G.shape[0])*self.r, self.values)

    def interpolate(self, points):
        """Actually do interpolation. Return interpolated values at each
        point in points.

        Parameters
        ----------
        points : see :meth:`__call__`

        Returns
        -------
        vals : 1d array (points.shape[0],)

        Notes
        -----
        Calculates
            distsq = (points - centers)**2  # squared distance matrix
            G = rbf(distsq)                 # RBF values
            zi = dot(G, w)                  # interpolation z_i = Sum_j g_ij w_j
        """
        self._assert_ndim_points(points)
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq, self.p)
        assert G.shape[1] == len(self.w), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape),
                                                 str(self.w.shape))
        # normalize w
        ww = self.w
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

                [[dz/dx0_0,   dz/dx0_1,   ..., dz/dx0_N-1],
                 [...],
                 [dz/dxL-1_0, dz/dxL-1_1, ..., dz/dxL-1_N-1]]
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
        # for ll in range(L):
        #     for kk in range(N):
        #         for jj in range(len(self.w)):
        #             D[ll,kk] += (points[ll,kk] - centers[jj,kk]) / G[ll,jj] * \
        #                 self.w[jj]
        #
        # # 2 loops:
        # D = np.zeros((L,N), dtype=float)
        # for ll in range(L):
        #     for kk in range(N):
        #         vec = -1.0 * (centers[:,kk] - points[ll,kk]) / G[ll,:]
        #         D[ll,kk] = np.dot(vec, self.w)
        self._assert_ndim_points(points)
        L,N = points.shape
        centers = self.centers
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq, self.p)
        D = np.empty((L,N), dtype=float)
        ww = self.w
        maxw = np.abs(ww).max()*1.0
        fname = self.rbf.__name__
        if fname == 'rbf_multi':
            for zz in range(L):
                D[zz,:] = -np.dot(((centers - points[zz,:]) / G[zz,:][:,None]).T, ww / maxw) * maxw
        elif fname == 'rbf_inv_multi':
            for zz in range(L):
                D[zz,:] = np.dot(((centers - points[zz,:]) * (G[zz,:]**3.0)[:,None]).T, ww / maxw) * maxw
        elif fname == 'rbf_gauss':
            for zz in range(L):
                D[zz,:] = 1.0 / self.p**2.0 * \
                    np.dot(((centers - points[zz,:]) * G[zz,:][:,None]).T, ww / maxw) * maxw
        else:
            raise Exception(f"derivative not implemented for function: {fname}")
        return D

    def fit_error(self, points, values):
        """Sum of squared fit errors whith penality on negative p."""
        res = values - self(points)
        err = np.dot(res,res) / len(res)
        return math.exp(abs(err)) if self.p < 0 else err

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
