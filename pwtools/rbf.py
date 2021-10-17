"""
Radial Basis Function regression. See :ref:`rbf` for details.
"""

import math

import numpy as np
from scipy import optimize
import scipy.linalg as linalg
from scipy.spatial.distance import cdist

try:
    from sklearn.model_selection import RepeatedKFold
except ImportError:
    class RepeatedKFold:
        def __init__(*args, **kwds):
            raise NotImplementedError("Failed to import RepeatedKFold from "
                                      "sklearn, not installed?")


def rbf_gauss(rsq, p):
    r"""Gaussian RBF :math:`\exp\left(-\frac{r^2}{2\,p^2}\right)`

    Parameters
    ----------
    rsq : float
        squared distance :math:`r^2`
    p : float
        width
    """
    return np.exp(-0.5*rsq / p**2.0)


def rbf_multi(rsq, p):
    r"""Multiquadric RBF :math:`\sqrt{r^2 + p^2}`

    Parameters
    ----------
    rsq : float
        squared distance :math:`r^2`
    p : float
        width
    """
    return np.sqrt(rsq + p**2.0)


def rbf_inv_multi(rsq, p):
    r"""Inverse Multiquadric RBF :math:`\frac{1}{\sqrt{r^2 + p^2}}`

    Parameters
    ----------
    rsq : float
        squared distance :math:`r^2`
    p : float
        width
    """
    return 1/rbf_multi(rsq, p)


def squared_dists(aa, bb):
    return cdist(aa, bb, metric="sqeuclidean")


def euclidean_dists(aa, bb):
    return cdist(aa, bb, metric="euclidean")


rbf_dct = {
    'gauss': rbf_gauss,
    'multi': rbf_multi,
    'inv_multi': rbf_inv_multi,
    }


class Rbf:
    """Radial basis function network interpolation and fitting."""
    def __init__(self, points, values, rbf='inv_multi',
                 r=None, p='mean', fit=True, lin_solver='dsysv'):
        """
        Parameters
        ----------
        points : 2d array, (M,N)
            data points : M points in N-dim space, training set points
        values : 1d array, (M,)
            function values at training points
        rbf : str (see rbf_dct.keys()) or callable rbf(r**2, p)
        r : float or None
            regularization parameter, if None then we use a least squares
            solver
        p : 'mean' or 'scipy' (see :func:`estimate_p`) or float
            the RBF's free parameter
        lin_solver : str
            Linear solver method in case `r` is given

            | solve : :func:`scipy.linalg.solve`
            | dsysv : :func:`scipy.linalg.lapack.dsysv` (symmetric G)
            | dposv : :func:`scipy.linalg.lapack.dposv` (positive definite G)
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
        self.rbf = rbf_dct[rbf] if isinstance(rbf, str) else rbf
        self._assert_ndim_points(self.points)
        self._assert_ndim_values(self.values)
        self.distsq = None
        if isinstance(p, str):
            if p == 'mean':
                # re-implement the 'mean' case here again since we can re-use
                # distsq later (training data distance matrix)
                self.distsq = self.get_distsq()
                self.p = np.sqrt(self.distsq).mean()
            elif p == 'scipy':
                self.p = estimate_p(points, 'scipy')
            else:
                raise ValueError("p is not 'mean' or 'scipy'")
        else:
            self.p = p
        self.r = r
        self.lin_solver = lin_solver
        self._lin_solvers = dict(
                solve=self._solve_general,
                dsysv=self._solve_dsysv,
                dposv=self._solve_dposv,
                lstsq=self._solve_lstsq,
                )
        if fit:
            self.fit()

    @staticmethod
    def _assert_ndim_points(points):
        assert points.ndim == 2, ("points not 2d array")

    @staticmethod
    def _assert_ndim_values(values):
        assert values.ndim == 1, ("values not 1d array")

    def get_distsq(self, points=None):
        r"""Matrix of distance values :math:`R_{ij} = |\mathbf x_i - \mathbf
        c_j|`.

            | :math:`\mathbf x_i` : ``points[i,:]``      (points)
            | :math:`\mathbf c_j` : ``self.points[j,:]`` (centers)

        Parameters
        ----------
        points : array (K,N) with N-dim points, optional
            If None then ``self.points`` is used (training points).

        Returns
        -------
        distsq : (M,K), where K = M for training
        """
        # training:
        #     If points == centers, we could also use
        #     scipy.spatial.distance.pdist(points), which would give us a 1d
        #     array of all distances. But we need the redundant square matrix
        #     form for G=rbf(distsq) anyway, so there is no real point in
        #     special-casing that. These two are the same:
        #       spatial.squareform(spatial.distances.pdist(points,
        #                                                  metric="sqeuclidean"))
        #       spatial.distances.cdist(points, points, metric="sqeuclidean")
        # speed: see examples/benchmarks/distmat_speed.py
        if points is None:
            if self.distsq is None:
                return squared_dists(self.points, self.points)
            else:
                return self.distsq
        else:
            return squared_dists(points, self.points)

    def get_params(self):
        """Return ``(p,r)``.
        """
        return self.p, self.r

    def fit(self):
        r"""Solve linear system for the weights.

        The weights  `self.w` (:math:`\mathbf w`) are found from: :math:`\mathbf
        G\,\mathbf w = \mathbf z` or if :math:`r` is given :math:`(\mathbf G +
        r\,\mathbf I)\,\mathbf w = \mathbf z`.

        with centers == points (center vectors are all data points). Then G is
        quadratic. Updates ``self.w``.

        Notes
        -----
        ``self.r != None`` : linear system solver
            Use `lin_solver`. For :math:`r=0`, this always yields
            perfect interpolation at the data points. May be numerically
            unstable in that case. Use :math:`r>0` to increase stability (try
            small values such as ``1e-10`` first) or create smooth fitting (generate
            more stiff functions with higher `r`). Behaves similar to ``lstsq``
            but appears to be numerically more stable (no small noise in
            solution) .. but `r` it is another parameter that needs to be
            tuned.
        ``self.r = None`` : least squares solver
            Use :func:`scipy.linalg.lstsq`. Numerically more stable than
            direct solver w/o regularization. Will mostly be the same as
            the interpolation result, but will not go thru all points for
            very noisy data. May create small noise in solution (plot fit
            with high point density). Also slower that e.g. ``dsysv``.
        """
        G = self.rbf(self.get_distsq(), self.p)
        assert G.shape == (self.points.shape[0],)*2
        if self.r is None:
            self.w = self._lin_solvers['lstsq'](G)
        else:
            self.w = self._lin_solvers[self.lin_solver](G + np.eye(G.shape[0])*self.r)

    def _solve_lstsq(self, G):
        x, res, rnk, svs = linalg.lstsq(G, self.values)
        return x

    # Generally shaped G
    def _solve_general(self, Gr):
        return linalg.solve(Gr, self.values)

    # G symmetric
    def _solve_dsysv(self, Gr):
        udut,ipiv,x,info = linalg.lapack.dsysv(Gr, self.values)
        if info > 0:
            raise Exception(f"info={info}: singular matrix")
        elif info < 0:
            raise Exception(f"illegal input for {info}-th argument")
        return x

    # G positive definite
    def _solve_dposv(self, Gr):
        c,x,info = linalg.lapack.dposv(Gr, self.values)
        if info > 0:
            raise Exception(f"info={info}: not positive definite")
        elif info < 0:
            raise Exception(f"illegal input for {info}-th argument")
        return x

    def interpolate(self, points):
        """Evaluate interpolant at `points`.

        Parameters
        ----------
        points : see :meth:`__call__`

        Returns
        -------
        vals : 1d array (points.shape[0],)
        """
        self._assert_ndim_points(points)
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq, self.p)
        assert G.shape[0] == points.shape[0]
        assert G.shape[1] == len(self.w), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape),
                                                 str(self.w.shape))
        # normalize w
        maxw = np.abs(self.w).max()*1.0
        return np.dot(G, self.w / maxw) * maxw

    def deriv(self, points):
        r"""Matrix of partial first derivatives.

        Parameters
        ----------
        points : 2d array (L,N)
            See also :meth:`__call__`

        Returns
        -------
        2d array (L,N)
            Each row holds the gradient vector :math:`\partial f/\partial\mathbf x_i`
            where :math:`\mathbf x_i = \texttt{points[i,:]
            = [xi_0, ..., xi_N-1]}`. For all points points (L,N) we get the
            matrix::

                [[df/dx0_0,   df/dx0_1,   ..., df/dx0_N-1],
                 [...],
                 [df/dxL-1_0, df/dxL-1_1, ..., df/dxL-1_N-1]]
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
        centers = self.points
        distsq = self.get_distsq(points=points)
        G = self.rbf(distsq, self.p)
        D = np.empty((L,N), dtype=float)
        maxw = np.abs(self.w).max()*1.0
        fname = self.rbf.__name__
        if fname == 'rbf_multi':
            for zz in range(L):
                D[zz,:] = -np.dot(((centers - points[zz,:]) / G[zz,:][:,None]).T, self.w / maxw) * maxw
        elif fname == 'rbf_inv_multi':
            for zz in range(L):
                D[zz,:] = np.dot(((centers - points[zz,:]) * (G[zz,:]**3.0)[:,None]).T, self.w / maxw) * maxw
        elif fname == 'rbf_gauss':
            for zz in range(L):
                D[zz,:] = 1.0 / self.p**2.0 * \
                    np.dot(((centers - points[zz,:]) * G[zz,:][:,None]).T, self.w / maxw) * maxw
        else:
            raise Exception(f"derivative not implemented for function: {fname}")
        return D

    def fit_error(self, points, values):
        """Sum of squared fit errors with penalty on negative `p`."""
        res = values - self(points)
        err = np.dot(res,res) / len(res)
        return math.exp(abs(err)) if self.p < 0 else err

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`interpolate` or :meth:`deriv`.

        Parameters
        ----------
        points : 2d array (L,N)
            L N-dim points to evaluate the interpolant on.
        der : int
            If == 1 return matrix of partial derivatives (see :meth:`deriv`), else
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


def estimate_p(points, method='mean'):
    r"""Estimate :math:`p`.

    Parameters
    ----------
    method : str
        | 'mean' : :math:`1/M^2\,\sum_{ij} R_{ij}; M=\texttt{points.shape[0]}`
        | 'scipy' : mean nearest neighbor distance
    """
    if method == 'mean':
        return euclidean_dists(points, points).mean()
    elif method == 'scipy':
        xi = points.T
        ximax = np.amax(xi, axis=1)
        ximin = np.amin(xi, axis=1)
        edges = ximax - ximin
        edges = edges[np.nonzero(edges)]
        return np.power(np.prod(edges)/xi.shape[-1], 1.0/edges.size)
    else:
        raise Exception(f"illegal method: {method}")


class FitError:
    """Direct or cross-validation (CV) fit error of :class:`Rbf` for a
    parameter set ``[p,r]`` or just ``[p]``.

    All methods accept a sequence `params` with either only `p` (length 1) or
    `p` and `r` (length 2) to build a :class:`Rbf` model and fit it.

    examples:

        | r = None (default in :class:`Rbf`) -> linear least squares solver
        |     params = [1.5]
        |     params = [1.5, None]

        | r != None -> normal linear solver
        |     params = [1.5, 0]       -> no regularization (r=0)
        |     params = [1.5, 1e-8]    -> with regularization

    Use :meth:`err_cv` or :meth:`err_direct` as error metric for `param`. Or
    use :meth:`__call__` which will call one or the other, depending on
    `params`.
    """
    def __init__(self, points, values, rbf_kwds=dict(),
                 cv_kwds=dict(n_splits=5, n_repeats=1)):
        """
        Parameters
        ----------
        points, values : see :class:`Rbf`
        rbf_kwds : dict
            for ``Rbf(points, values, **rbf_kwds)``
        cv_kwds : {dict, None}, optional
            cross-validation parameters for
            :class:`sklearn.model_selection.RepeatedKFold`), if None then
            :meth:`__call__` will use :meth:`err_direct`, else :meth:`err_cv`
        """
        self.points = points
        self.values = values
        self.rbf_kwds = rbf_kwds
        self.cv_kwds = cv_kwds

    def __call__(self, params):
        if self.cv_kwds is None:
            return self.err_direct(params)
        else:
            return self.err_cv(params)

    def _get_rbfi(self, params, points=None, values=None):
        points = self.points if points is None else points
        values = self.values if values is None else values
        if len(params) == 1:
            assert 'p' not in self.rbf_kwds.keys(), "'p' in kwds"
            return Rbf(points, values, p=params[0], **self.rbf_kwds)
        elif len(params) == 2:
            for kw in ['p', 'r']:
                assert kw not in self.rbf_kwds.keys(), f"'{kw}' in kwds"
            return Rbf(points, values, p=params[0], r=params[1], **self.rbf_kwds)
        else:
            raise Exception("length of params can only be 1 or 2, got "
                            "{}".format(len(params)))

    def cv(self, params):
        """K-fold repeated CV.

        Split data (points, values) randomly into K parts ("folds", K =
        ``n_splits`` in ``self.cv_kwds``) along axis 0 and use each part once
        as test set, the rest as training set. For example `ns=5`: split in 5
        parts at random indices, use 5 times 4/5 data for train, 1/5 for test
        (each of the folds), so 5 fits total -> 5 fit errors. Optionally repeat
        ``n_repeats`` times with different random splits. So, `n_repeats` *
        `n_splits` fit errors total.

        Each time, build an Rbf interpolator with ``self.rbf_kwds``, fit,
        return the fit error (scalar sum of squares from
        :meth:`Rbf.fit_error`).

        Parameters
        ----------
        params : seq length 1 or 2
            | params[0] = p
            | params[1] = r (optional)

        Returns
        -------
        errs : 1d array (n_repeats * n_splits,)
            direct fit error from each fold
        """
        ns = self.cv_kwds['n_splits']
        nr = self.cv_kwds['n_repeats']
        errs = np.empty((ns*nr,), dtype=float)
        folds = RepeatedKFold(**self.cv_kwds)
        for ii, tup in enumerate(folds.split(self.points)):
            idxs_train, idxs_test = tup
            rbfi = self._get_rbfi(params,
                                  self.points[idxs_train,...],
                                  self.values[idxs_train,...])
            errs[ii] = rbfi.fit_error(self.points[idxs_test,...],
                                      self.values[idxs_test,...])
        return errs

    def err_cv(self, params):
        """Median of :meth:`cv`."""
        return np.median(self.cv(params))

    def err_direct(self, params):
        """Normal fit error w/o CV. Uses :meth:`Rbf.fit_error`.

        Build and Rbf interpolator with ``self.rbf_kwds``, fit, return the fit
        error (scalar, sum of squares). Should be zero for interpolation, i.e. no
        regularization ``r=0``.
        """
        return self._get_rbfi(params).fit_error(self.points, self.values)


def fit_opt(points, values, method='de', what='pr', cv_kwds=dict(n_splits=5,
            n_repeats=1), opt_kwds=dict(), rbf_kwds=dict()):
    """Optimize :class:`Rbf`'s hyper-parameter :math:`p` or both :math:`(p,r)`.

    Use a cross validation error metric or the direct fit error if `cv_kwds` is
    None. Uses :class:`FitError`.

    Parameters
    ----------
    points, values : see :class:`Rbf`
    method : str
        | 'de' : :func:`scipy.optimize.differential_evolution`
        | 'fmin': :func:`scipy.optimize.fmin`
        | 'brute': :func:`scipy.optimize.brute`
    what : str
        'p' (optimize only `p` and set `r=None`) or 'pr' (optimize `p` and `r`)
    cv_kwds, rbf_kwds : see :class:`FitError`
    opt_kwds : dict
        kwds for the optimizer (see `method`)

    Returns
    -------
    rbfi : :class:`Rbf`
        Rbf instance initialized with `points`, `values` and found optimal `p` (and `r`).
    """
    assert what in ['p', 'pr'], (f"unknown `what` value: {what}")
    assert method in ['de', 'fmin', 'brute'], (f"unknown `method` value: {method}")
    fit_err = FitError(points, values, cv_kwds=cv_kwds, rbf_kwds=rbf_kwds)
    p0 = estimate_p(points)
    disp = opt_kwds.pop('disp', False)
    if method == 'fmin':
        if what == 'p':
            x0 = opt_kwds.pop('x0', [p0])
        elif what == 'pr':
            x0 = opt_kwds.pop('x0', [p0, 1e-8])
        xopt = optimize.fmin(fit_err, x0,
                             disp=disp, **opt_kwds)
    elif method in ['de', 'brute']:
        if what == 'p':
            _bounds = [(0, 5*p0)]
        elif what == 'pr':
            _bounds = [(0, 5*p0), (1e-12, 1e-1)]
        if method == 'de':
            bounds = opt_kwds.pop('bounds', _bounds)
            ret = optimize.differential_evolution(fit_err,
                                                  bounds=bounds,
                                                  disp=disp,
                                                  **opt_kwds)
            xopt = ret.x
        elif method == 'brute':
            bounds = opt_kwds.pop('ranges', _bounds)
            xopt = optimize.brute(fit_err,
                                  ranges=bounds,
                                  disp=disp,
                                  **opt_kwds)
    if what == 'pr':
        rbfi = Rbf(points, values, p=xopt[0], r=xopt[1])
    else:
        rbfi = Rbf(points, values, p=xopt[0])
    return rbfi
