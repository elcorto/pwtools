"""
Radial Basis Function regression. See :ref:`rbf` for details.
"""

import math

from pwtools import config
JAX_MODE = config.use_jax

if JAX_MODE:
    import jax.numpy as np
    from jax.config import config
    # Need double prec, else, the analytic derivs in Rbf.deriv() as well as
    # the autodiff version Rbf.deriv_jax() are rubbish.
    config.update("jax_enable_x64", True)
    from jax import grad, vmap, jit
    import jax.scipy.linalg as jax_linalg
else:
    import numpy as np
    from scipy.spatial.distance import cdist

from scipy import optimize
import scipy.linalg as linalg

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


rbf_dct = {
    'gauss': rbf_gauss,
    'multi': rbf_multi,
    'inv_multi': rbf_inv_multi,
    }


# Consider using jax-md space.py here
def _np_distsq(aa, bb):
    """(Slow) pure numpy squared distance matrix.

    Need that in JAX_MODE b/c we cannot diff thru
    scipy.spatial.distance.cdist(). BUT: jax.jit() is crazy good (factor 40
    faster than the numpy expression, factor almost 4 better than cdist (with
    64 bit)!!

    >>> f=lambda aa, bb: ((aa[:,None,:] - bb[None,...])**2.0).sum(-1)
    >>> jf=jit(f)
    >>> %timeit f(x,x)
    39.3 ms ± 438 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    >>> %timeit scipy.spatial.distance.cdist(x,x)
    3.43 ms ± 9.25 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit jf(x,x)
    1.03 ms ± 53.9 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    return ((aa[:,None,:] - bb[None,...])**2.0).sum(-1)

if JAX_MODE:
    _jax_np_distsq = jit(_np_distsq)
    _jax_np_dist = jit(lambda aa, bb: np.sqrt(_np_distsq(aa, bb)))

def squared_dists(aa, bb):
    if JAX_MODE:
        return _jax_np_distsq(aa, bb)
    else:
        return cdist(aa, bb, metric="sqeuclidean")


def euclidean_dists(aa, bb):
    if JAX_MODE:
        return _jax_np_dist(aa, bb)
    else:
        return cdist(aa, bb, metric="euclidean")


class Rbf:
    r"""Radial basis function network interpolation and regression.

    Notes
    -----
    Array shape API is as in :class:`~pwtools.num.PolyFit`.

    :math:`\texttt{f}: \mathbb R^{m\times n} \rightarrow \mathbb R^m` if
    ``points.ndim=2``, which is the setting when training and vectorized
    evaluation.

    >>> X.shape
    (m,n)
    >>> y.shape
    (m,)
    >>> f=Rbf(X,y)
    >>> f(X).shape
    (m,)

    :math:`\texttt{f}: \mathbb R^n \rightarrow \mathbb R` if
    ``points.ndim=1``.

    >>> x.shape
    (n,)
    >>> f(x).shape
    ()

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
    def __init__(self, points, values, rbf='inv_multi',
                 r=None, p='mean', fit=True):
        r"""
        Parameters
        ----------
        points : 2d array, (M,N)
            data points : M points in N-dim space, training set points
        values : 1d array, (M,)
            function values at training points
        rbf : str (see rbf_dct.keys()) or callable rbf(r**2, p)
            RBF definition
        r : float or None
            regularization parameter, if None then we use a least squares
            solver
        p : 'mean' or 'scipy' (see :func:`estimate_p`) or float
            the RBF's free parameter
        fit : bool
            call :meth:`fit` in :meth:`__init__`
        """
        assert points.ndim == 2, ("points must be 2d array")
        assert values.ndim == 1, ("values must be 1d array")
        self.npoints = points.shape[0]
        self.ndim = points.shape[1]
        assert len(values) == self.npoints, (f"{len(values)=} != {self.npoints=}")
        self.points = points
        self.values = values
        self.rbf = rbf_dct[rbf] if isinstance(rbf, str) else rbf
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

        if fit:
            self.fit()

    def _rectify_points_shape(self, points):
        ret = np.atleast_2d(points)
        assert ret.shape[1] == self.ndim, (f"{ret.shape[1]=} != {self.ndim}")
        return ret

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
        #
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
            For :math:`r=0`, this always yields perfect interpolation at the
            data points. May be numerically unstable in that case. Use
            :math:`r>0` to increase stability (try small values such as
            ``1e-10`` first) or create smooth fitting (generate more stiff
            functions with higher `r`). Behaves similar to ``lstsq`` but
            appears to be numerically more stable (no small noise in solution)
            .. but `r` it is another parameter that needs to be tuned.
        ``self.r = None`` : least squares solver
            Use :func:`scipy.linalg.lstsq`. Numerically more stable than
            direct solver w/o regularization. Will mostly be the same as
            the interpolation result, but will not go thru all points for
            very noisy data. May create small noise in solution (plot fit
            with high point density). Much (up to 10x) slower that normal
            linear solver when ``self.r != None``.
        """
        G = self.rbf(self.get_distsq(), self.p)
        assert G.shape == (self.npoints,)*2
        # The lstsq solver is ~10x slower than jax' solver and ~4x slower than
        # the symmetric scipy solver
        if self.r is None:
            self.w = self._solve_lstsq(G)
        else:
            self.w = self._solve(G + np.eye(G.shape[0])*self.r)

    def _solve_lstsq(self, G):
        x, res, rnk, svs = linalg.lstsq(G, self.values)
        return x

    def _solve(self, Gr):
        if JAX_MODE:
            la = jax_linalg
            # jax.scipy.linalg.solve() doesn't have assume_a kwd, but is still
            # ~3 times faster than scipy.linalg.solve(..., assume_a="sym")
            # which calls scipy.linalg.lapack.dsysv()
            kwds = {}
        else:
            la = linalg
            kwds = dict(assume_a="sym")
        return la.solve(Gr, self.values, **kwds)

    def predict(self, points):
        """Evaluate model at `points`.

        Parameters
        ----------
        points : 2d array (L,N) or (N,)

        Returns
        -------
        vals : 1d array (L,) or scalar
        """
        _got_single_point = points.ndim==1
        points = self._rectify_points_shape(points)
        assert points.shape[1] == self.ndim, ("wrong ndim")
        G = self.rbf(self.get_distsq(points=points), self.p)
        assert G.shape[0] == points.shape[0]
        assert G.shape[1] == len(self.w), \
               "shape mismatch between g_ij: %s and w_j: %s, 2nd dim of "\
               "g_ij must match length of w_j" %(str(G.shape),
                                                 str(self.w.shape))
        # normalize w
        maxw = np.abs(self.w).max()*1.0
        values = np.dot(G, self.w / maxw) * maxw
        return values[0] if _got_single_point else values

    def deriv_jax(self, points):
        """Partial derivs from jax.

        Same API as :meth:`deriv`: ``grad`` for 1d input or ``vmap(grad)`` for
        2d input.

            >>> x.shape
            (n,)
            >>> grad(f)(x).shape
            (n,)

            >>> X.shape
            (m,n)
            >>> vmap(grad(self))(X).shape
            (m,n)

        Parameters
        ----------
        points : 2d array (L,N) or (N,)

        Returns
        -------
        grads :  2d array (L,N) or (N,)

        See Also
        --------
        :func:`deriv`
        """
        if JAX_MODE:
            if points.ndim == 1:
                assert len(points) == self.ndim, ("wrong ndim")
                return jit(grad(self))(points)
            elif points.ndim == 2:
                assert points.shape[1] == self.ndim, ("wrong ndim")
                return jit(vmap(grad(self)))(points)
            else:
                raise Exception("points has wrong shape")
        else:
            raise NotImplementedError

    def deriv(self, points):
        r"""Analytic first partial derivatives.

        Analytic reference implementation of ``jax`` ``grad`` for 1d input or
        ``vmap(grad)`` for 2d input.

            >>> x.shape
            (n,)
            >>> grad(f)(x).shape
            (n,)

            >>> X.shape
            (m,n)
            >>> vmap(grad(self))(X).shape
            (m,n)

        Parameters
        ----------
        points : 2d array (L,N) or (N,)

        Returns
        -------
        2d array (L,N) or (N,)
            Each row holds the gradient vector :math:`\partial f/\partial\mathbf x_i`
            where :math:`\mathbf x_i = \texttt{points[i,:]
            = [xi_0, ..., xi_N-1]}`. For all points points (L,N) we get the
            matrix::

                [[df/dx0_0,   df/dx0_1,   ..., df/dx0_N-1],
                 [...],
                 [df/dxL-1_0, df/dxL-1_1, ..., df/dxL-1_N-1]]

        See Also
        --------
        :func:`deriv_jax`
        """
        # For the implemented RBF types, the derivatives w.r.t. to the point
        # coords simplify to nice dot products, which can be evaluated
        # reasonably fast w/ numpy. We don't need to change the RBF's
        # implementations to provide a deriv() method. For that, they would
        # need to take points and centers explicitly as args instead of squared
        # distances, which are calculated fast by cdist().

        # Also, this implementation of analytic derivs is nice, but we want to
        # play with the cool kids and also use jax. This method here serves as
        # reference mostly.

        # Speed:
        # We have one python loop over the L points (points.shape=(L,N)) left, so
        # this gets slow for many points.
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
        _got_single_point = points.ndim==1
        points = self._rectify_points_shape(points)
        assert points.shape[1] == self.ndim, ("wrong ndim")
        L,N = points.shape
        centers = self.points
        G = self.rbf(self.get_distsq(points=points), self.p)
        maxw = np.abs(self.w).max()*1.0
        fname = self.rbf.__name__

        # analytic deriv funcs for the inner loop
        D_zz = dict(rbf_multi=lambda zz: -np.dot(((centers - points[zz,:]) / \
                                         G[zz,:][:,None]).T, self.w / maxw) * maxw,
                    rbf_inv_multi=lambda zz: np.dot(((centers - points[zz,:]) * \
                                             (G[zz,:]**3.0)[:,None]).T, self.w / maxw) * maxw,
                    rbf_gauss=lambda zz: 1.0 / self.p**2.0 * np.dot(((centers - points[zz,:]) \
                                         * G[zz,:][:,None]).T, self.w / maxw) * maxw,
                )
        assert fname in D_zz.keys(), (f"{fname} not in {D_zz.keys()}")
        func = D_zz[fname]
        if JAX_MODE:
            # Of course don't call this method when doing jax autodiff. Still
            # when in JAX_MODE, np = jax.numpy and thus its limitations apply.
            # In this case, we use the code here only as reference
            # implementation, but it must anyway work under jax.numpy .
            #
            # Because of that, we must use slow list comp b/c jax' functional
            # workaround for in-place ops
            #   D.at[zz,:].set(func(zz))
            # still doesn't do inplace (despite its name, but in sync w/ docs)
            # unless we jax.jit stuff, then docs say. Instead it returns a
            # copy, in this case the full D matrix, with just one line changed.
            # Then it is cheaper to just list comp.
            #
            # We tried to jit this method but D.at[zz,:].set(func(zz)) still
            # doesn't update D and returns D all zero. The in-place state of
            # mind has no place in jax land.
            #
            ##D = np.empty((L,N), dtype=float)
            ##for zz in range(L):
            ##    D.at[zz,:].set(func(zz))
            D = np.array([func(zz) for zz in range(L)])
        else:
            D = np.empty((L,N), dtype=float)
            for zz in range(L):
                D[zz,:] = func(zz)
        return D[0] if _got_single_point else D

    def fit_error(self, points, values):
        """Sum of squared fit errors with penalty on negative `p`."""
        res = values - self(points)
        err = np.dot(res,res) / len(res)
        return math.exp(abs(err)) if self.p < 0 else err

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`predict` or :meth:`deriv`.

        Parameters
        ----------
        points : 2d array (L,N) or (N,)
            L N-dim points to evaluate the model on.
        der : int
            If 1 return (matrix of) partial derivatives (see :meth:`deriv` and
            :meth:`deriv_jax`), else model prediction values (default).

        Returns
        -------
        vals : 1d array (L,) or scalar
            Interpolated values.
        or
        derivs : 2d array (L,N) or (N,)
            1st partial derivatives.
        """
        # We hard-code only the 1st deriv using jax, mainly as an example, and
        # for easy comparison to our analytic derivs.
        #
        # Higher order derivs can be implemented by the user outside, e.g.
        #
        # >>> X=rand(100,2); y=rand(100)
        # >>> f=Rbf(X,y)
        # >>> jax.hessian(f)(rand(2)).shape
        # (4,4)
        if 'der' in list(kwargs.keys()):
            if kwargs['der'] != 1:
                raise Exception("only der=1 supported")
            kwargs.pop('der')
            if JAX_MODE:
                return self.deriv_jax(*args, **kwargs)
            else:
                return self.deriv(*args, **kwargs)
        else:
            return self.predict(*args, **kwargs)


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

        Each time, build an Rbf model with ``self.rbf_kwds``, fit,
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

        Build and Rbf model with ``self.rbf_kwds``, fit, return the fit
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
