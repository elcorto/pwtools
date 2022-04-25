import warnings
import copy

import numpy as np

from scipy import optimize

from pwtools.rbf.core import Rbf, estimate_p

try:
    from sklearn.model_selection import RepeatedKFold
except ImportError:

    class RepeatedKFold:
        def __init__(*args, **kwds):
            raise NotImplementedError(
                "Failed to import RepeatedKFold from "
                "sklearn, not installed?"
            )


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

    def __init__(
        self,
        points,
        values,
        rbf_kwds=dict(),
        cv=dict(n_splits=5, n_repeats=1),
        cv_kwds=None,
    ):
        """
        Parameters
        ----------
        points, values : see :class:`Rbf`
        rbf_kwds : dict
            for ``Rbf(points, values, **rbf_kwds)``
        cv : {dict, :class:`sklearn.model_selection.BaseCrossValidator` instance
            or anything with that API, None}, optional, if dict then
            cross-validation parameters for
            :class:`sklearn.model_selection.RepeatedKFold`, if
            ``BaseCrossValidator``-like class then this is used to split data,
            if None then :meth:`__call__` will use :meth:`err_direct`, else
            :meth:`err_cv`
        cv_kwds : deprecated: dict of kwds for default RepeatedKFold
        """
        self.points = points
        self.values = values
        self.rbf_kwds = rbf_kwds
        _default_cv_cls = RepeatedKFold

        self._use_cv = (cv_kwds is not None) or (cv is not None)
        if cv_kwds is not None:
            warnings.warn(
                "use `cv` keyword, `cv_kwds` deprecated", DeprecationWarning
            )
            self.cv_cls = _default_cv_cls(**cv_kwds)
        else:
            if isinstance(cv, dict):
                self.cv_cls = _default_cv_cls(**cv)
            else:
                self.cv_cls = cv

    def __call__(self, params):
        if self._use_cv:
            return self.err_cv(params)
        else:
            return self.err_direct(params)

    def _get_rbfi(self, params, points=None, values=None):
        points = self.points if points is None else points
        values = self.values if values is None else values
        if len(params) == 1:
            assert "p" not in self.rbf_kwds.keys(), "'p' in kwds"
            return Rbf(points, values, p=params[0], **self.rbf_kwds)
        elif len(params) == 2:
            for kw in ["p", "r"]:
                assert kw not in self.rbf_kwds.keys(), f"'{kw}' in kwds"
            return Rbf(
                points, values, p=params[0], r=params[1], **self.rbf_kwds
            )
        else:
            raise Exception(
                "length of params can only be 1 or 2, got "
                "{}".format(len(params))
            )

    def cv(self, params):
        """Cross validation fit errors.

        Default is RepeatedKFold (if `cv` is a dict, else `cv` itself is used):
        Split data (points, values) randomly into K parts ("folds", K =
        ``n_splits``) along axis 0 and use each part once as test set, the rest
        as training set. For example `ns=5`: split in 5 parts at random
        indices, use 5 times 4/5 data for train, 1/5 for test (each of the
        folds), so 5 fits total -> 5 fit errors. Optionally repeat
        ``n_repeats`` times with different random splits. So, `n_repeats` *
        `n_splits` fit errors total.

        Each time, build an Rbf model with ``self.rbf_kwds``, fit, return the
        fit error (scalar sum of squares from :meth:`Rbf.fit_error`).

        Parameters
        ----------
        params : seq length 1 or 2
            | params[0] = p
            | params[1] = r (optional)

        Returns
        -------
        errs : 1d array
            direct fit errors on the test set from each split
        """
        errs = np.empty((self.cv_cls.get_n_splits(self.points),), dtype=float)
        for ii, tup in enumerate(self.cv_cls.split(self.points)):
            idxs_train, idxs_test = tup
            rbfi = self._get_rbfi(
                params,
                self.points[idxs_train, ...],
                self.values[idxs_train, ...],
            )
            errs[ii] = rbfi.fit_error(
                self.points[idxs_test, ...], self.values[idxs_test, ...]
            )
        return errs

    def err_cv(self, params):
        """Mean of :meth:`cv`."""
        return self.cv(params).mean()

    def err_direct(self, params):
        """Normal fit error w/o CV. Uses :meth:`Rbf.fit_error`.

        Build and Rbf model with ``self.rbf_kwds``, fit, return the fit
        error (scalar, sum of squares). Should be zero for interpolation, i.e. no
        regularization ``r=0``.
        """
        return self._get_rbfi(params).fit_error(self.points, self.values)


def fit_opt(
    points,
    values,
    method="de",
    what="pr",
    cv=dict(n_splits=5, n_repeats=1),
    cv_kwds=None,
    opt_kwds=dict(),
    rbf_kwds=dict(),
):
    """Optimize :class:`Rbf`'s hyper-parameter :math:`p` or both :math:`(p,r)`.

    Use a cross validation error metric or the direct fit error if `cv` is
    None. Uses :class:`FitError`.

    Note: While we do have some defaults for initial guess or bounds, depending
    on the optimizer, you are strongly advised to set your own in `opt_kwds`.

    Parameters
    ----------
    points, values :
        see :class:`Rbf`
    method : str
        * 'de' : :func:`scipy.optimize.differential_evolution`
        * 'fmin': :func:`scipy.optimize.fmin`
        * 'brute': :func:`scipy.optimize.brute`
    what : str
        * 'p' : optimize only `p`, set fixed `r` in `rbf_kwds` in this case, else
          we'll use :class:`Rbf`'s default)
        * 'pr' : optimize `p` and `r`
    cv, rbf_kwds :
        see :class:`FitError`
    opt_kwds : dict
        kwds for the optimizer (see `method`)
    rbf_kwds : dict
        Constant params for :class:`Rbf`

    Returns
    -------
    rbfi : :class:`Rbf`
        Rbf instance initialized with `points`, `values` and found optimal `p` (and `r`).
    """
    assert what in (
        what_ok := ["p", "pr"]
    ), f"unknown value {what=}, allowed is {what_ok}"
    what_lst = list(what)
    for kw in what_lst:
        assert (
            kw not in rbf_kwds.keys()
        ), f"{kw} in {rbf_kwds=} while {kw} should be optimized"
    assert method in [
        "de",
        "fmin",
        "brute",
    ], f"unknown {method=}"
    fit_err = FitError(
        points, values, cv=cv, cv_kwds=cv_kwds, rbf_kwds=rbf_kwds
    )
    _get_p0 = lambda: estimate_p(points)
    disp = opt_kwds.pop("disp", False)
    if method == "fmin":
        if what == "p":
            x0 = opt_kwds.pop("x0", [_get_p0()])
        elif what == "pr":
            x0 = opt_kwds.pop("x0", [_get_p0(), 1e-8])
        xopt = optimize.fmin(fit_err, x0, disp=disp, **opt_kwds)
    elif method in ["de", "brute"]:
        if what == "p":
            _bounds = [(0, 5 * _get_p0())]
        elif what == "pr":
            _bounds = [(0, 5 * _get_p0()), (1e-12, 1e-1)]
        if method == "de":
            bounds = opt_kwds.pop("bounds", _bounds)
            ret = optimize.differential_evolution(
                fit_err, bounds=bounds, disp=disp, **opt_kwds
            )
            xopt = ret.x
        elif method == "brute":
            bounds = opt_kwds.pop("ranges", _bounds)
            xopt = optimize.brute(
                fit_err, ranges=bounds, disp=disp, **opt_kwds
            )
    rbf_kwds_opt = copy.copy(rbf_kwds)
    for idx, symbol in enumerate(what_lst):
        rbf_kwds_opt[symbol] = xopt[idx]
    return Rbf(points, values, **rbf_kwds_opt)
